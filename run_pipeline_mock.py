import torch
import yaml
import numpy as np

from src.perception import CameraViewAligner, LeWMEncoder
from src.imagination import MuDreamerPredictor, MuDreamerActorCritic, MuDreamerValueTrainer
from src.lifelong import LegionComponentAllocator, LegionRewardNormalizer
from src.telemetry import JepaDreamerLogger

# Assuming all previously defined classes are imported/loaded in your runtime environment:
# CameraViewAligner, LeWMEncoder, MuDreamerPredictor, MuDreamerActorCritic,
# LegionComponentAllocator, LegionRewardNormalizer, JepaDreamerLogger, MuDreamerValueTrainer

def run_legion_jepa_dreamer_loop(config_path="config/legion_jepa_dreamer.yaml"):
    # 1. Load Configurations
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting JEPA-Dreamer on device: {device}")
    
    # 2. Initialize Shared Base Components
    stn_aligner = CameraViewAligner().to(device)
    base_encoder = LeWMEncoder(embed_dim=cfg["lewm_encoder"]["embed_dim"]).to(device)
    base_encoder.eval() # Keep the LeWM base frozen as configured
    
    # 3. Initialize Lifelong Management Modules
    allocator = LegionComponentAllocator(threshold=cfg["legion"]["knowledge_space"]["expansion_threshold"],
                                         max_components=cfg["legion"]["knowledge_space"]["max_components"])
    reward_normalizer = LegionRewardNormalizer(
        clip_range=tuple(cfg["environment"]["reward_mapping"]["clip_range"])
    )
    logger = JepaDreamerLogger(log_dir=cfg["system"]["checkpoint_dir"] + "/logs")
    
    # Repositories for task-specific networks
    task_predictors = {}
    task_actor_critics = {}
    task_value_trainers = {}
    
    # Mock environment loop parameters for structure verification
    total_steps = 5000
    horizon = cfg["mudreamer_core"]["horizon"]
    embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]
    
    # 4. Main Lifelong RL Loop Simulation
    for step in range(total_steps):
        # Create a mock batch step mimicking image arrays from Metaworld-KUKA-IIWA-R800
        # (Batch=4, Channels=3, Height=224, Width=224)
        mock_raw_pixels = torch.randn(4, 3, 224, 224).to(device)
        mock_raw_reward = np.random.uniform(-2.0, 5.0)
        
        # Phase A: Correct Camera Shift and Extract Stable Latents
        with torch.no_grad():
            aligned_pixels = stn_aligner(mock_raw_pixels)
            stable_latents = base_encoder(aligned_pixels)
            
        # Phase B: Run LEGION Allocation Check via Latent Space Profiles
        # Evaluates the first vector elements in the batch to decide task routing
        task_id, spawned_new = allocator.evaluate_task_assignment(stable_latents[0], step, logger)
        
        if spawned_new:
            print(f"[!] LEGION triggered allocation! Spawning new MuDreamer Task Head ID: {task_id}")
            task_predictors[task_id] = MuDreamerPredictor(embed_dim, action_dim).to(device)
            task_actor_critics[task_id] = MuDreamerActorCritic(embed_dim, action_dim).to(device)
            task_value_trainers[task_id] = MuDreamerValueTrainer(
                critic_net=task_actor_critics[task_id].critic,
                lr=cfg["mudreamer_core"]["optimizer"]["lr_critic"],
                discount=cfg["mudreamer_core"]["discount"],
                lambda_=cfg["mudreamer_core"]["lambda_"]
            )
            
        # Phase C: Normalize Environmental Feedbacks
        reward_normalizer.update_statistics([mock_raw_reward])
        clean_reward_tensor = reward_normalizer.normalize(
            torch.full((horizon, 4), mock_raw_reward, dtype=torch.float32).to(device)
        )
        
        # Phase D: Execute MuDreamer Imaginative Trajectory Planning
        active_predictor = task_predictors[task_id]
        active_ac = task_actor_critics[task_id]
        active_trainer = task_value_trainers[task_id]
        
        # Generate internal rollouts directly from current stable latent anchor
        current_latent = stable_latents.clone()
        imagined_trajectory = [current_latent]
        
        for h_step in range(horizon):
            actions = active_ac.get_action(current_latent)
            next_latent = active_predictor(current_latent, actions)
            imagined_trajectory.append(next_latent)
            current_latent = next_latent
            
        imagined_trajectory_tensor = torch.stack(imagined_trajectory) # Shape: (H+1, B, Embed_Dim)
        
        # Phase E: Optimization Steps (Backpropagation inside the non-reconstructive imagination)
        critic_loss = active_trainer.train_step(imagined_trajectory_tensor, clean_reward_tensor)
        
        # Mock values for logging telemetry data
        actor_loss = critic_loss * 0.4 
        pred_loss = 0.05
        policy_entropy = 1.2
        
        # Telemetry updates
        if step % cfg["system"]["log_interval"] == 0:
            logger.log_imagination_step(
                step=step,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                pred_loss=pred_loss,
                policy_entropy=policy_entropy,
                rewards=clean_reward_tensor
            )
            print(f"Step {step}/{total_steps} | Task ID: {task_id} | Critic Loss: {critic_loss:.4f}")
            
    logger.close()
    print("[*] Unified pipeline run completed successfully.")

if __name__ == "__main__":
    # Create configuration folder and run template verification loop
    import os
    os.makedirs("config", exist_ok=True)
    os.makedirs("./checkpoints/legion_jepa_dreamer", exist_ok=True)
    
    # Execute training runtime directly
    run_legion_jepa_dreamer_loop()
