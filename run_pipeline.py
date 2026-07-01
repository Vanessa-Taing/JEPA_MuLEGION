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
    symexp,
)
from src.btn import MinimalBTN, BTNTrainer
from src.lifelong import LegionComponentAllocator, LegionRewardNormalizer, LegionReplayBuffer
from src.telemetry import JepaDreamerLogger


def evaluate_policy(
    env_class,
    task,
    actor_critic,
    stn_aligner,
    base_encoder,
    action_dim,
    device,
    is_frozen,
    btn=None,
    n_episodes=5,
    resize=(112, 112),
):
    eval_env = env_class(render_mode="rgb_array")
    eval_env.set_task(task)

    total_reward = 0.0
    total_success = 0

    was_training = base_encoder.training
    base_encoder.eval()

    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        ep_reward = 0.0
        done = False
        last_info = {}
        wm_hidden = None  # BTN working memory reset per episode

        while not done:
            raw_frame = eval_env.render()
            img_tensor = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            pixel_input = torch.nn.functional.interpolate(
                img_tensor, size=resize, mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                aligned = stn_aligner(pixel_input)
                latent = base_encoder(aligned)

                conditioning = None
                if btn is not None:
                    conditioning, wm_hidden, _ = btn(latent, wm_hidden)

                action = actor_critic.get_action(latent, conditioning, deterministic=True)
                real_action = action.cpu().numpy().flatten()[:action_dim]

            obs, reward, terminated, truncated, info = eval_env.step(real_action)
            ep_reward += reward
            last_info = info
            done = terminated or truncated

        total_reward += ep_reward
        total_success += int(last_info.get("success", 0.0) > 0.5)

    eval_env.close()

    if was_training:
        base_encoder.train()

    return {
        "mean_episode_reward": total_reward / n_episodes,
        "success_rate": total_success / n_episodes,
    }


def run_legion_jepa_dreamer_loop(config_path="config/legion_jepa_dreamer.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    encoder_type = cfg["lewm_encoder"].get("encoder_type", "lewm")
    use_btn = cfg.get("btn", {}).get("enabled", False)
    print(
        f"[*] Starting JEPA-Dreamer on device: {device} | "
        f"Encoder: {encoder_type} | BTN: {use_btn}"
    )

    print("[*] Initializing Meta-World Robotic Environment...")
    task_name = "reach-v3"
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="rgb_array")
    task = mt1.train_tasks[0]
    env.set_task(task)
    obs, info = env.reset()

    embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]
    conditioning_dim = cfg["btn"]["conditioning_dim"] if use_btn else 0

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
    task_btns = {}
    task_btn_trainers = {}

    encoder_optimizer = None
    if is_trainable_online:
        encoder_optimizer = torch.optim.Adam(base_encoder.parameters(), lr=1e-4)
        print("[*] Encoder will train online alongside world model (MuDreamer mode).")

    # Encoder warmup: freeze encoder gradient updates for first N steps
    # even in mudreamer mode, giving predictor/critic time to stabilize
    # before their input distribution starts shifting
    encoder_warmup_steps = cfg.get("encoder_warmup_steps", 2000) if is_trainable_online else 0

    total_steps = cfg["system"].get("total_steps", 5000)
    horizon = cfg["mudreamer_core"]["horizon"]
    batch_size = cfg["mudreamer_core"].get("batch_size", 64)
    min_buffer_size = max(batch_size * 4, 256)
    eval_interval = cfg["system"].get("eval_interval", 1000)
    eval_episodes = cfg["system"].get("eval_episodes", 5)

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
            was_training = base_encoder.training
            base_encoder.eval()
            with torch.no_grad():
                latent = base_encoder(pixels)
            if was_training:
                base_encoder.train()
            return latent

    def ensure_task_modules(task_id):
        if task_id not in task_predictors:
            print(f"[!] LEGION triggered allocation! Spawning MuDreamer Task Head ID: {task_id}")
            task_predictors[task_id] = MuDreamerPredictor(
                embed_dim, action_dim,
                hidden_dim=cfg["mudreamer_core"]["networks"]["hidden_dim"]
            ).to(device)
            task_actor_critics[task_id] = MuDreamerActorCritic(
                embed_dim, action_dim, conditioning_dim=conditioning_dim
            ).to(device)
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
            if use_btn:
                btn_cfg = cfg["btn"]
                task_btns[task_id] = MinimalBTN(
                    embed_dim=embed_dim,
                    n_subjects=btn_cfg.get("n_subjects", 30),
                    n_predicates=btn_cfg.get("n_predicates", 15),
                    n_objects=btn_cfg.get("n_objects", 30),
                    vocab_embed_dim=btn_cfg.get("vocab_embed_dim", 64),
                    conditioning_dim=btn_cfg["conditioning_dim"],
                ).to(device)
                task_btn_trainers[task_id] = BTNTrainer(
                    btn=task_btns[task_id],
                    lr=btn_cfg.get("lr", 3e-4),
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
        active_btn = task_btns.get(task_id, None)
        active_btn_trainer = task_btn_trainers.get(task_id, None)

        # ---- Act ----
        with torch.no_grad():
            live_conditioning = None
            if active_btn is not None:
                # Use a fresh working memory state for the live action step
                # (not the imagination rollout's working memory)
                live_conditioning, _, _ = active_btn(stable_latents, wm_hidden=None)
            action_tensor = active_ac.get_action(stable_latents, live_conditioning)
            real_action = action_tensor.cpu().numpy().flatten()[:action_dim]

        next_obs, reward, terminated, truncated, info = env.step(real_action)
        done = float(terminated or truncated)

        next_pixels = get_aligned_pixels()
        next_stable_latents = encode(next_pixels)

        buffer.add(stable_latents, real_action, reward, next_stable_latents, done)
        reward_normalizer.update_statistics([reward])

        # ---- Train world model on real transitions ----
        wm_metrics = {}
        btn_metrics = {}
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

            # Encoder warmup: only allow encoder gradient updates after warmup_steps
            encoder_active = (
                is_trainable_online
                and encoder_optimizer is not None
                and step >= encoder_warmup_steps
            )

            if encoder_active:
                base_encoder.train()
                encoder_optimizer.zero_grad()

            wm_metrics = active_wm_trainer.train_step(
                states_b, actions_b, next_states_b, rewards_b, dones_b
            )

            if encoder_active:
                encoder_optimizer.step()
                base_encoder.eval()

            # BTN self-supervised training on buffer latents
            if active_btn_trainer is not None:
                btn_metrics = active_btn_trainer.train_step(states_b)

        # ---- Imagination rollout seeded from replayed buffer states ----
        if len(buffer) >= min_buffer_size:
            seed_batch = buffer.sample(batch_size)
            seed_latents = torch.cat([b[0] for b in seed_batch], dim=0).to(device)
        else:
            seed_latents = stable_latents.expand(batch_size, -1).clone()

        current_latent = seed_latents           # (B, D)
        # GRU hidden state — initialized to zeros, carried across imagination horizon
        h_imagination = torch.zeros(
            batch_size, active_predictor.hidden_dim, device=device
        )
        # BTN working memory — initialized to zeros, carried across imagination horizon
        wm_imagination = None

        imagined_latents = [current_latent]
        imagined_rewards = []
        imagined_continues = []
        imagined_log_probs = []
        imagined_entropy = []

        for h_step in range(horizon):
            # Get BTN conditioning for this imagination step
            imagination_conditioning = None
            if active_btn is not None:
                imagination_conditioning, wm_imagination, _ = active_btn(
                    current_latent, wm_imagination
                )

            dist = active_ac.get_action_dist(current_latent, imagination_conditioning)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # GRU hidden state is carried — this is what gives the world model memory
            next_latent, pred_reward, pred_continue_logit, h_imagination = (
                active_predictor(current_latent, action, h_imagination)
            )
            pred_continue = torch.sigmoid(pred_continue_logit)

            imagined_latents.append(next_latent)
            imagined_rewards.append(pred_reward.squeeze(-1))
            imagined_continues.append(pred_continue.squeeze(-1))
            imagined_log_probs.append(log_prob)
            imagined_entropy.append(entropy)

            current_latent = next_latent

        imagined_latents_t = torch.stack(imagined_latents)     # (H+1, B, D)
        imagined_rewards_t = torch.stack(imagined_rewards)     # (H, B)
        imagined_continues_t = torch.stack(imagined_continues) # (H, B)
        imagined_log_probs_t = torch.stack(imagined_log_probs) # (H, B)
        imagined_entropy_t = torch.stack(imagined_entropy)     # (H, B)

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
                rewards=symexp(imagined_rewards_t).detach(),
            )
            encoder_frozen_str = (
                f"[warmup {step}/{encoder_warmup_steps}]"
                if is_trainable_online and step < encoder_warmup_steps
                else ""
            )
            print(
                f"Step {step}/{total_steps} | Task ID: {task_id} | "
                f"Real Reward: {reward:.4f} | Critic Loss: {critic_loss:.4f} | "
                f"Actor Loss: {actor_metrics['actor_loss']:.4f} | "
                f"WM Loss: {wm_metrics.get('world_model_loss', float('nan')):.4f} | "
                f"BTN Loss: {btn_metrics.get('btn_loss', float('nan')):.4f} "
                f"{encoder_frozen_str}"
            )

        if step % eval_interval == 0 and step > 0:
            eval_metrics = evaluate_policy(
                env_class=mt1.train_classes[task_name],
                task=task,
                actor_critic=active_ac,
                stn_aligner=stn_aligner,
                base_encoder=base_encoder,
                action_dim=action_dim,
                device=device,
                is_frozen=is_frozen,
                btn=active_btn,
                n_episodes=eval_episodes,
            )
            logger.writer.add_scalar(
                "Eval/MeanEpisodeReward", eval_metrics["mean_episode_reward"], step
            )
            logger.writer.add_scalar(
                "Eval/SuccessRate", eval_metrics["success_rate"], step
            )
            print(
                f"  [EVAL] Mean Episode Reward: {eval_metrics['mean_episode_reward']:.4f} | "
                f"Success Rate: {eval_metrics['success_rate']:.2f}"
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