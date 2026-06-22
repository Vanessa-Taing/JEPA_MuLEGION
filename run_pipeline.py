import os
import yaml
import numpy as np
import torch

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import metaworld

from src.encoders import build_encoder
from src.imagination import (
    MuDreamerPredictor,
    MuDreamerActorCritic,
    MuDreamerValueTrainer,
    MuDreamerWorldModelTrainer,
    MuDreamerActorTrainer,
)
from src.lifelong import LegionComponentAllocator, LegionRewardNormalizer, LegionReplayBuffer
from src.telemetry import JepaDreamerLogger


def run_legion_jepa_dreamer_loop(config_path="config/legion_jepa_dreamer.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    encoder_type = cfg["lewm_encoder"].get("encoder_type", "lewm")
    print(f"[*] Starting JEPA-Dreamer on device: {device} | Encoder: {encoder_type}")

    print("[*] Initializing Meta-World Robotic Environment...")
    task_name = "reach-v3"
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="rgb_array")
    task = mt1.train_tasks[0]
    env.set_task(task)
    obs, info = env.reset()

    embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]

    stn_aligner, base_encoder, is_frozen, is_trainable_online = build_encoder(
        encoder_type=encoder_type,
        embed_dim=embed_dim,
        device=device,
        checkpoint_path=cfg["lewm_encoder"].get("pretrained_weights"),
    )

    buffer = LegionReplayBuffer(max_size=cfg["legion"]["replay_buffer"]["max_size"])
    allocator = LegionComponentAllocator(
        alpha=cfg["legion"]["knowledge_space"]["expansion_threshold"]
    )
    reward_normalizer = LegionRewardNormalizer(
        clip_range=tuple(cfg["environment"]["reward_mapping"]["clip_range"])
    )
    logger = JepaDreamerLogger(log_dir=cfg["system"]["checkpoint_dir"] + "/logs")

    task_predictors = {}
    task_actor_critics = {}
    task_value_trainers = {}
    task_world_model_trainers = {}
    task_actor_trainers = {}

    # If encoder trains online alongside world model (mudreamer mode),
    # each task's world model trainer needs to also update encoder params.
    encoder_optimizer = None
    if is_trainable_online:
        encoder_optimizer = torch.optim.Adam(base_encoder.parameters(), lr=1e-4)
        print("[*] Encoder will train online alongside world model (MuDreamer mode).")

    total_steps = cfg["system"].get("total_steps", 5000)
    horizon = cfg["mudreamer_core"]["horizon"]
    batch_size = cfg["mudreamer_core"].get("batch_size", 64)
    min_buffer_size = max(batch_size * 4, 256)

    def get_aligned_pixels():
        raw_frame = env.render()
        if raw_frame is None:
            raise RuntimeError("Render returned None. Check render_mode='rgb_array'")
        img_tensor = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        pixel_input = torch.nn.functional.interpolate(
            img_tensor, size=(112, 112), mode="bilinear", align_corners=False
        )
        if is_frozen:
            with torch.no_grad():
                aligned = stn_aligner(pixel_input)
        else:
            aligned = stn_aligner(pixel_input)
        return aligned

    def encode(pixels):
        if is_frozen:
            with torch.no_grad():
                return base_encoder(pixels)
        else:
            # In online mode, encoder is updated during world model training;
            # for acting we still detach to avoid unnecessary graph accumulation
            with torch.no_grad():
                return base_encoder(pixels)

    def ensure_task_modules(task_id):
        if task_id not in task_predictors:
            print(f"[!] LEGION triggered allocation! Spawning MuDreamer Task Head ID: {task_id}")
            task_predictors[task_id] = MuDreamerPredictor(embed_dim, action_dim).to(device)
            task_actor_critics[task_id] = MuDreamerActorCritic(embed_dim, action_dim).to(device)
            task_value_trainers[task_id] = MuDreamerValueTrainer(
                critic_net=task_actor_critics[task_id].critic,
                lr=cfg["mudreamer_core"]["optimizer"]["lr_critic"],
                discount=cfg["mudreamer_core"]["discount"],
                lambda_=cfg["mudreamer_core"]["lambda_"],
            )
            task_world_model_trainers[task_id] = MuDreamerWorldModelTrainer(
                predictor=task_predictors[task_id],
                lr=cfg["mudreamer_core"]["optimizer"]["lr_predictor"],
            )
            task_actor_trainers[task_id] = MuDreamerActorTrainer(
                actor_critic=task_actor_critics[task_id],
                lr=cfg["mudreamer_core"]["optimizer"]["lr_actor"],
            )

    current_pixels = get_aligned_pixels()
    stable_latents = encode(current_pixels)

    for step in range(total_steps):
        eval_latent = stable_latents[0]

        task_id, spawned_new = allocator.evaluate_task_assignment(eval_latent, step, logger)
        ensure_task_modules(task_id)

        active_predictor = task_predictors[task_id]
        active_ac = task_actor_critics[task_id]
        active_value_trainer = task_value_trainers[task_id]
        active_wm_trainer = task_world_model_trainers[task_id]
        active_actor_trainer = task_actor_trainers[task_id]

        # ---- Act ----
        with torch.no_grad():
            action_tensor = active_ac.get_action(stable_latents)
            real_action = action_tensor.cpu().numpy().flatten()[:action_dim]

        next_obs, reward, terminated, truncated, info = env.step(real_action)
        done = float(terminated or truncated)

        next_pixels = get_aligned_pixels()
        next_stable_latents = encode(next_pixels)

        buffer.add(stable_latents, real_action, reward, next_stable_latents, done)
        reward_normalizer.update_statistics([reward])

        # ---- Train world model on real transitions ----
        wm_metrics = {}
        if len(buffer) >= min_buffer_size:
            batch = buffer.sample(batch_size)
            states_b = torch.cat([b[0] for b in batch], dim=0).to(device)
            actions_b = torch.tensor(
                np.stack([b[1] for b in batch]), dtype=torch.float32
            ).to(device)
            rewards_b = torch.tensor(
                [b[2] for b in batch], dtype=torch.float32
            ).view(-1, 1).to(device)
            next_states_b = torch.cat([b[3] for b in batch], dim=0).to(device)
            dones_b = torch.tensor(
                [b[4] for b in batch], dtype=torch.float32
            ).view(-1, 1).to(device)

            # In mudreamer mode, zero the encoder optimizer before wm step
            # so gradients flow through both predictor and encoder together
            if is_trainable_online and encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            wm_metrics = active_wm_trainer.train_step(
                states_b, actions_b, next_states_b, rewards_b, dones_b
            )

            if is_trainable_online and encoder_optimizer is not None:
                encoder_optimizer.step()

        # ---- Imagine rollout ----
        current_latent = stable_latents.clone()
        imagined_latents = [current_latent]
        imagined_rewards = []
        imagined_continues = []
        imagined_log_probs = []
        imagined_entropy = []

        for h_step in range(horizon):
            dist = active_ac.get_action_dist(current_latent)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            next_latent, pred_reward, pred_continue_logit = active_predictor(
                current_latent, action
            )
            pred_continue = torch.sigmoid(pred_continue_logit)

            imagined_latents.append(next_latent)
            imagined_rewards.append(pred_reward.squeeze(-1))
            imagined_continues.append(pred_continue.squeeze(-1))
            imagined_log_probs.append(log_prob)
            imagined_entropy.append(entropy)

            current_latent = next_latent

        imagined_latents_t = torch.stack(imagined_latents)
        imagined_rewards_t = torch.stack(imagined_rewards)
        imagined_continues_t = torch.stack(imagined_continues)
        imagined_log_probs_t = torch.stack(imagined_log_probs)
        imagined_entropy_t = torch.stack(imagined_entropy)

        critic_loss, lambda_returns, traj_values = active_value_trainer.train_step(
            imagined_latents_t.detach(),
            imagined_rewards_t.detach(),
            imagined_continues_t.detach(),
        )

        actor_metrics = active_actor_trainer.train_step(
            imagined_latents_t[:-1].detach(),
            imagined_log_probs_t,
            imagined_entropy_t,
            lambda_returns,
            traj_values,
        )

        if step % cfg["system"]["log_interval"] == 0:
            logger.log_imagination_step(
                step=step,
                actor_loss=actor_metrics["actor_loss"],
                critic_loss=critic_loss,
                pred_loss=wm_metrics.get("world_model_loss", 0.0),
                policy_entropy=imagined_entropy_t.mean().item(),
                rewards=imagined_rewards_t,
            )
            print(
                f"Step {step}/{total_steps} | Task ID: {task_id} | "
                f"Real Reward: {reward:.4f} | Critic Loss: {critic_loss:.4f} | "
                f"Actor Loss: {actor_metrics['actor_loss']:.4f} | "
                f"WM Loss: {wm_metrics.get('world_model_loss', float('nan')):.4f}"
            )

        if terminated or truncated:
            env.reset()
            current_pixels = get_aligned_pixels()
            stable_latents = encode(current_pixels)
        else:
            current_pixels = next_pixels
            stable_latents = next_stable_latents

    env.close()
    logger.close()
    print("[*] Execution completed successfully.")


if __name__ == "__main__":
    run_legion_jepa_dreamer_loop()