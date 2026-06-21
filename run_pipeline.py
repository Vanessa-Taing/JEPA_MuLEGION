import os
import yaml
import numpy as np
import torch
import gymnasium as gym

# Force PyTorch memory optimization for your RTX 2080 Ti
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Import Meta-World benchmark suites
import metaworld

from src.perception import CameraViewAligner, LeWMEncoder
from src.imagination import MuDreamerPredictor, MuDreamerActorCritic, MuDreamerValueTrainer
from src.lifelong import LegionComponentAllocator, LegionRewardNormalizer, LegionReplayBuffer
from src.telemetry import JepaDreamerLogger

def run_legion_jepa_dreamer_loop(config_path="config/legion_jepa_dreamer.yaml"):
    # 1. Load Configurations
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        print(cfg["legion"])
        
    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting JEPA-Dreamer on device: {device}")
    
    # 2. Initialize Meta-World Environment Suite
    # We load MT1 (Single Task instance) for clean testing, like 'assembly-v2' or 'reach-v2'
    print("[*] Initializing Meta-World Robotic Environment...")
    task_name = "reach-v3"
    mt1 = metaworld.MT1(task_name)

    env = mt1.train_classes[task_name](
        render_mode="rgb_array"
    )

    task = mt1.train_tasks[0]
    env.set_task(task)
    
    # Reset environment to get initial observation and configure rendering
    obs, info = env.reset()
    
    # 3. Initialize Shared Base Components
    stn_aligner = CameraViewAligner().to(device)
    base_encoder = LeWMEncoder(embed_dim=cfg["lewm_encoder"]["embed_dim"]).to(device)
    base_encoder.eval() 
    
    # 4. Initialize Lifelong Management Modules
    buffer = LegionReplayBuffer(max_size=cfg["legion"]["replay_buffer"]["max_size"])
    allocator = LegionComponentAllocator(threshold=cfg["legion"]["knowledge_space"]["expansion_threshold"])
    reward_normalizer = LegionRewardNormalizer(
        clip_range=tuple(cfg["environment"]["reward_mapping"]["clip_range"])
    )
    logger = JepaDreamerLogger(log_dir=cfg["system"]["checkpoint_dir"] + "/logs")
    
    task_predictors = {}
    task_actor_critics = {}
    task_value_trainers = {}
    
    total_steps = cfg["system"].get("total_steps", 5000)
    horizon = cfg["mudreamer_core"]["horizon"]
    embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]
    
    # 5. Main Lifelong RL Loop over Real Robot Actions
    for step in range(total_steps):
        
        # Phase A: Render Visual Camera Stream from MuJoCo Simulator
        # Render yields a (H, W, C) uint8 numpy array, e.g., (480, 480, 3)
        raw_frame = env.render()

        if raw_frame is None:
            raise RuntimeError(
                "Render returned None. Check that env was created with render_mode='rgb_array'"
            )

        img_tensor = torch.from_numpy(raw_frame.copy()).float()

        # HWC -> CHW
        img_tensor = img_tensor.permute(2,0,1)

        # normalize
        img_tensor = img_tensor / 255.0

        # add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(device)

        pixel_input = torch.nn.functional.interpolate(
            img_tensor,
            size=(112,112),
            mode="bilinear",
            align_corners=False
        )
        # Phase B: Correct Camera Shift and Extract Stable Latents
        with torch.no_grad():
            aligned_pixels = stn_aligner(pixel_input)
            stable_latents = base_encoder(aligned_pixels)
            
        # Phase C: Run LEGION Allocation Check via Latent Space Profiles
        # Use consistent evaluations for stable profile convergence during testing
        eval_latent = stable_latents[0]
            
        task_id, spawned_new = allocator.evaluate_task_assignment(eval_latent, step, logger)
        
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
            
        active_predictor = task_predictors[task_id]
        active_ac = task_actor_critics[task_id]
        active_trainer = task_value_trainers[task_id]
        
        # Phase D: Sample Real Action from Policy and Advance Robot Simulation State
        with torch.no_grad():
            # Extract continuous action vector from current latent anchor
            action_tensor = active_ac.get_action(stable_latents)
            # Convert to a flat NumPy array matching Meta-World control constraints
            real_action = action_tensor.cpu().numpy().flatten()[:action_dim]
            
        # Step the physical MuJoCo environment engine forward
        next_obs, reward, terminated, truncated, info = env.step(real_action)

        buffer.add(
            stable_latents.cpu(),
            real_action,
            reward,
            next_obs,
            terminated or truncated
        )
        
        # Phase E: Normalize Environmental Feedbacks
        reward_normalizer.update_statistics([reward])
        clean_reward_tensor = reward_normalizer.normalize(
            torch.full(
                (horizon, 1),
                reward,
                dtype=torch.float32,
                device=device
            )
        )
        
        # Phase F: Execute MuDreamer Imaginative Trajectory Planning
        current_latent = stable_latents.clone()
        imagined_trajectory = [current_latent]
        
        for h_step in range(horizon):
            actions = active_ac.get_action(current_latent)
            next_latent = active_predictor(current_latent, actions)
            imagined_trajectory.append(next_latent)
            current_latent = next_latent
            
        imagined_trajectory_tensor = torch.stack(imagined_trajectory)
        
        # Phase G: Value Training Backpropagation step
        critic_loss = active_trainer.train_step(imagined_trajectory_tensor, clean_reward_tensor)
        
        if step % 1000 == 0:
            print(f"Step {step}/{total_steps} | Task ID: {task_id} | Real Reward: {reward:.4f} | Critic Loss: {critic_loss:.4f}")
            
        if terminated or truncated:
            env.reset()
            
    env.close()
    print("[*] Real robotic environment execution completed successfully.")

if __name__ == "__main__":
    run_legion_jepa_dreamer_loop()
