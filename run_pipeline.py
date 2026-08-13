import os
import sys
import yaml
import numpy as np
import torch
from datetime import datetime

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import metaworld

from src.encoders import build_encoder
from src.imagination import (
    RSSMCore,
    MuDreamerActorCritic,
    MuDreamerValueTrainer,
    MuDreamerWorldModelTrainer,
    MuDreamerActorTrainer,
    symexp,
)
from src.btn import MinimalBTN, BTNTrainer
from src.lifelong import (
    LegionComponentAllocator,
    LegionRewardNormalizer,
    LegionReplayBuffer,
    SequenceReplayBuffer,
)
from src.telemetry import JepaDreamerLogger


def evaluate_policy(
    env_class,
    task,
    rssm,
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
    final_distances = []
    # Diagnostics for the "tanh saturation trap" and "BTN conditioning acts
    # as a near-constant bias" hypotheses: if final_distance is frozen while
    # other metrics move, we need to know whether the ACTIONS being sent to
    # the env are pinned near the tanh boundary (+/-1), and whether the BTN
    # conditioning vector is actually varying across the episode or
    # collapsing to something close to constant regardless of state.
    all_action_abs_means = []       # per-step mean |action| across episodes
    all_action_saturated_fracs = [] # per-step fraction of |action| > 0.99
    per_episode_conditioning_stds = []  # std of conditioning vector ACROSS TIME within an episode

    was_training = base_encoder.training
    base_encoder.eval()
    rssm.eval()

    for ep in range(n_episodes):
        obs, info = eval_env.reset()
        ep_reward = 0.0
        done = False
        last_info = {}

        h, z = rssm.initial_state(1, device)
        wm_hidden = None
        prev_action = torch.zeros(1, action_dim, device=device)
        episode_conditioning_vectors = []

        while not done:
            raw_frame = eval_env.render()
            img_tensor = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            pixel_input = torch.nn.functional.interpolate(
                img_tensor, size=resize, mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                aligned = stn_aligner(pixel_input)
                obs_embed = base_encoder(aligned)
                # Eval uses posterior step — carry h across episode
                rssm.posterior_net.eval()
                h, z, _, _ = rssm.observe_step(h, z, prev_action, obs_embed)
                state = rssm.get_state_features(h, z)

                conditioning = None
                if btn is not None:
                    conditioning, wm_hidden, _ = btn(state, wm_hidden)
                    episode_conditioning_vectors.append(conditioning.cpu().numpy().flatten())

                action = actor_critic.get_action(state, conditioning, deterministic=True)
                real_action = action.cpu().numpy().flatten()[:action_dim]

                all_action_abs_means.append(float(abs(real_action).mean()))
                all_action_saturated_fracs.append(
                    float((abs(real_action) > 0.99).mean())
                )

            obs, reward, terminated, truncated, info = eval_env.step(real_action)
            ep_reward += reward
            last_info = info
            done = terminated or truncated
            prev_action = action

        if len(episode_conditioning_vectors) > 1:
            cond_arr = np.stack(episode_conditioning_vectors)  # (T, conditioning_dim)
            # Mean std across the conditioning dim, computed across TIME —
            # near-zero means the BTN is emitting roughly the same vector
            # every step regardless of how the state actually changed.
            per_episode_conditioning_stds.append(float(cond_arr.std(axis=0).mean()))

        total_reward += ep_reward
        total_success += int(last_info.get("success", 0.0) > 0.5)

        # MetaWorld envs commonly expose a gripper/object-to-target distance
        # in info under one of these keys depending on env version. This is
        # the metric that distinguishes "policy moves toward the goal but
        # can't close the last few cm" from "policy isn't heading there at
        # all" — something reward/success alone can't tell apart. We check
        # several known key names and skip silently if none are present so
        # this never breaks eval on an unexpected MetaWorld version.
        for dist_key in ("obj_to_target", "distance_to_target", "reachDist", "target_to_obj"):
            if dist_key in last_info:
                final_distances.append(float(last_info[dist_key]))
                break

    eval_env.close()
    rssm.train()
    if was_training:
        base_encoder.train()

    result = {
        "mean_episode_reward": total_reward / n_episodes,
        "success_rate": total_success / n_episodes,
    }
    if final_distances:
        result["mean_final_distance"] = sum(final_distances) / len(final_distances)
    if all_action_abs_means:
        result["mean_abs_action"] = sum(all_action_abs_means) / len(all_action_abs_means)
        result["frac_saturated_action"] = (
            sum(all_action_saturated_fracs) / len(all_action_saturated_fracs)
        )
    if per_episode_conditioning_stds:
        result["mean_conditioning_std_over_time"] = (
            sum(per_episode_conditioning_stds) / len(per_episode_conditioning_stds)
        )
    return result


class _Tee:
    """
    Minimal stdout duplicator: every print() still shows up in the console
    exactly as before, AND is simultaneously written to a timestamped log
    file. This is a pure side-channel — it doesn't touch TensorBoard or any
    training logic, it just captures the same console output you've been
    manually copy-pasting back for review, so every run is saved
    automatically without needing to remember to redirect output yourself.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_legion_jepa_dreamer_loop(config_path="config/legion_jepa_dreamer.yaml"):
    # Save every run's console output to runs/run_<timestamp>.txt, in
    # addition to printing normally. Directory and file are created fresh
    # each call, so concurrent/repeated runs never overwrite each other.
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(runs_dir, f"run_{run_timestamp}.txt")
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"[*] Logging this run's console output to: {log_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    encoder_type = cfg["lewm_encoder"].get("encoder_type", "lewm")
    use_btn = cfg.get("btn", {}).get("enabled", False)
    print(
        f"[*] Starting JEPA-MuLEGION on device: {device} | "
        f"Encoder: {encoder_type} | BTN: {use_btn}"
    )

    print("[*] Initializing Meta-World environment...")
    task_name = "reach-v3"
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="rgb_array")
    task = mt1.train_tasks[0]
    env.set_task(task)
    obs, info = env.reset()

    obs_embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]
    deter_dim = cfg["mudreamer_core"]["rssm"]["deter_dim"]
    stoch_dim = cfg["mudreamer_core"]["rssm"]["stoch_dim"]
    stoch_classes = cfg["mudreamer_core"]["rssm"]["stoch_classes"]
    hidden_dim = cfg["mudreamer_core"]["networks"]["hidden_dim"]
    conditioning_dim = cfg["btn"]["conditioning_dim"] if use_btn else 0
    state_dim = deter_dim + stoch_dim * stoch_classes
    seq_len = cfg["mudreamer_core"].get("sequence_length", 50)
    batch_size = cfg["mudreamer_core"].get("batch_size", 16)

    stn_aligner, base_encoder, is_frozen, is_trainable_online = build_encoder(
        encoder_type=encoder_type,
        embed_dim=obs_embed_dim,
        device=device,
        checkpoint_path=cfg["lewm_encoder"].get("pretrained_weights"),
    )

    # Sequence buffer: for RSSM training (sequential, propagates h).
    # Episodes are tagged with a majority-vote task_id so sample_sequences()
    # can be filtered per cluster (see lifelong.py).
    seq_buffer = SequenceReplayBuffer(
        max_episodes=cfg["mudreamer_core"].get("max_episodes", 500),
        sequence_length=seq_len,
    )
    # Flat buffer: for actor-critic imagination seeding. Transitions are
    # tagged per-step with task_id so sample() can be filtered per cluster.
    flat_buffer = LegionReplayBuffer(
        max_size=cfg["legion"]["replay_buffer"]["max_size"]
    )

    allocator = LegionComponentAllocator(
        alpha=cfg["legion"]["knowledge_space"]["expansion_threshold"]
    )
    reward_normalizer = LegionRewardNormalizer(
        clip_range=tuple(cfg["environment"]["reward_mapping"]["clip_range"])
    )
    logger = JepaDreamerLogger(log_dir=cfg["system"]["checkpoint_dir"] + "/logs")

    task_rssms = {}
    task_actor_critics = {}
    task_value_trainers = {}
    task_wm_trainers = {}
    task_actor_trainers = {}
    task_btns = {}
    task_btn_trainers = {}

    encoder_optimizer = None
    encoder_warmup_steps = 0
    if is_trainable_online:
        encoder_optimizer = torch.optim.Adam(base_encoder.parameters(), lr=1e-4)
        encoder_warmup_steps = cfg.get("encoder_warmup_steps", 2000)
        print(f"[*] Encoder trains online. Warmup: {encoder_warmup_steps} steps.")

    total_steps = cfg["system"].get("total_steps", 50000)
    horizon = cfg["mudreamer_core"]["horizon"]
    # min episodes before we start RSSM training — need enough sequences
    min_episodes = cfg["mudreamer_core"].get("min_episodes_before_train", 10)
    eval_interval = cfg["system"].get("eval_interval", 1000)
    eval_episodes = cfg["system"].get("eval_episodes", 5)

    # ---- Actor/critic warmup gate ----
    # The world model receives ZERO real gradient updates until seq_buffer
    # has min_episodes full episodes (world_model_loss is nan before that —
    # see the diagnostic run). Prior to this fix, the actor and critic
    # trained every step regardless, imagining against a completely
    # untrained/random RSSM for that entire stretch. The instant the RSSM
    # gets its first real update, its reward/dynamics predictions shift
    # discontinuously, and an actor/critic pair that has already converged
    # toward exploiting the old (random) dynamics gets hit with a target
    # shift it can't absorb — this is what produced the Critic:10.3 /
    # Actor:-75.4 spike and the subsequent frozen, collapsed policy.
    # Fix: don't let the actor/critic take gradient steps until the world
    # model has accumulated `wm_warmup_updates` real training steps of its
    # own. Imagination rollouts still run (needed to keep the code path
    # exercised and for logging), but their gradients are discarded during
    # warmup so the policy can't be shaped by an untrained model.
    wm_warmup_updates = cfg["mudreamer_core"].get("wm_warmup_updates", 200)
    task_wm_update_counts = {}

    def get_aligned_pixels():
        raw_frame = env.render()
        if raw_frame is None:
            raise RuntimeError("Render returned None.")
        img = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).to(device)
        img = torch.nn.functional.interpolate(
            img, size=(112, 112), mode="bilinear", align_corners=False
        )
        if is_frozen:
            with torch.no_grad():
                return stn_aligner(img)
        return stn_aligner(img)

    def encode(pixels):
        if is_frozen:
            with torch.no_grad():
                return base_encoder(pixels)
        was_training = base_encoder.training
        base_encoder.eval()
        with torch.no_grad():
            latent = base_encoder(pixels)
        if was_training:
            base_encoder.train()
        return latent

    def ensure_task_modules(task_id):
        if task_id in task_rssms:
            return
        print(f"[!] LEGION: Spawning RSSM + Actor-Critic for Task Head ID: {task_id}")

        rssm = RSSMCore(
            obs_embed_dim=obs_embed_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            hidden_dim=hidden_dim,
        ).to(device)

        ac = MuDreamerActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            conditioning_dim=conditioning_dim,
        ).to(device)

        task_rssms[task_id] = rssm
        task_actor_critics[task_id] = ac

        task_value_trainers[task_id] = MuDreamerValueTrainer(
            critic_net=ac.critic,
            lr=cfg["mudreamer_core"]["optimizer"]["lr_critic"],
            discount=cfg["mudreamer_core"]["discount"],
            lambda_=cfg["mudreamer_core"]["lambda_"],
        )
        task_wm_trainers[task_id] = MuDreamerWorldModelTrainer(
            rssm=rssm,
            lr=cfg["mudreamer_core"]["optimizer"]["lr_predictor"],
            kl_weight=cfg["mudreamer_core"].get("kl_weight", 1.0),
        )

        # BTN is constructed BEFORE the actor trainer now, so its parameters
        # can be passed in as an auxiliary param group. This is the fix for
        # the optimizer-scoping bug: previously the actor's backward pass
        # genuinely computed gradients on the BTN's parameters (since the
        # BTN sits inside the imagination rollout feeding the actor), but
        # nothing ever applied them — BTNTrainer's own optimizer only
        # covered btn.parameters() via a SEPARATE, entropy-only forward
        # pass, and its zero_grad() wiped the actor-derived gradients before
        # they were ever used. Now MuDreamerActorTrainer's optimizer
        # includes btn.parameters() as a second param group (at a reduced
        # lr, see auxiliary_lr_scale), so the policy-gradient signal that
        # was already being computed actually gets applied.
        btn = None
        if use_btn:
            btn_cfg = cfg["btn"]
            btn = MinimalBTN(
                embed_dim=state_dim,
                n_subjects=btn_cfg.get("n_subjects", 30),
                n_predicates=btn_cfg.get("n_predicates", 15),
                n_objects=btn_cfg.get("n_objects", 30),
                vocab_embed_dim=btn_cfg.get("vocab_embed_dim", 64),
                conditioning_dim=btn_cfg["conditioning_dim"],
            ).to(device)
            task_btns[task_id] = btn
            # BTNTrainer's entropy-maximization loss remains as a separate,
            # complementary stabilizer (prevents the vocabulary collapsing
            # to a single triple), but is no longer the ONLY signal the BTN
            # receives.
            task_btn_trainers[task_id] = BTNTrainer(
                btn=btn, lr=btn_cfg.get("lr", 3e-4)
            )

        task_actor_trainers[task_id] = MuDreamerActorTrainer(
            actor_critic=ac,
            lr=cfg["mudreamer_core"]["optimizer"]["lr_actor"],
            auxiliary_params=btn.parameters() if btn is not None else None,
            auxiliary_lr_scale=cfg["mudreamer_core"].get("btn_policy_lr_scale", 0.5),
            action_mean_penalty=cfg["mudreamer_core"].get("action_mean_penalty", 3e-5),
        )

    # Live RSSM state — carried across steps within an episode
    current_pixels = get_aligned_pixels()
    current_obs_embed = encode(current_pixels)
    live_h = None
    live_z = None
    prev_action = torch.zeros(1, action_dim, device=device)
    current_task_id = None
    bootstrap_state = torch.zeros(1, state_dim, device=device)

    for step in range(total_steps):

        # ---- Task allocation ----
        if live_h is not None and current_task_id is not None:
            alloc_state = task_rssms[current_task_id].get_state_features(live_h, live_z)
        else:
            alloc_state = bootstrap_state

        task_id, _ = allocator.evaluate_task_assignment(alloc_state[0], step, logger)
        ensure_task_modules(task_id)

        if task_id != current_task_id or live_h is None:
            live_h, live_z = task_rssms[task_id].initial_state(1, device)
            prev_action = torch.zeros(1, action_dim, device=device)
        current_task_id = task_id

        active_rssm = task_rssms[task_id]
        active_ac = task_actor_critics[task_id]
        active_value_trainer = task_value_trainers[task_id]
        active_wm_trainer = task_wm_trainers[task_id]
        active_actor_trainer = task_actor_trainers[task_id]
        active_btn = task_btns.get(task_id, None)
        active_btn_trainer = task_btn_trainers.get(task_id, None)

        # ---- Posterior step: update live RSSM with current observation ----
        active_rssm.posterior_net.eval()
        with torch.no_grad():
            live_h, live_z, _, _ = active_rssm.observe_step(
                live_h, live_z, prev_action, current_obs_embed
            )
        active_rssm.posterior_net.train()
        live_state = active_rssm.get_state_features(live_h, live_z)

        # ---- Act ----
        with torch.no_grad():
            live_conditioning = None
            if active_btn is not None:
                live_conditioning, _, _ = active_btn(live_state, wm_hidden=None)
            action_tensor = active_ac.get_action(live_state, live_conditioning)
            real_action = action_tensor.cpu().numpy().flatten()[:action_dim]

        next_obs, reward, terminated, truncated, info = env.step(real_action)
        done = float(terminated or truncated)

        next_pixels = get_aligned_pixels()
        next_obs_embed = encode(next_pixels)

        # ---- Store in BOTH buffers, tagged with the current DPMM cluster ----
        # Sequence buffer: for sequential RSSM training
        seq_buffer.add_step(current_obs_embed, real_action, reward, done, task_id=task_id)
        # Flat buffer: for imagination seeding (actor/critic)
        flat_buffer.add(
            current_obs_embed, real_action, reward, next_obs_embed, done, task_id=task_id
        )

        reward_normalizer.update_statistics([reward])
        prev_action = action_tensor

        # ---- Train RSSM sequentially on episode chunks ----
        # Sampling is filtered to THIS cluster's episodes where possible (falls
        # back to the full episode pool if the cluster is too new — see
        # SequenceReplayBuffer.sample_sequences). Without this filter, a
        # task's RSSM would be trained on dynamics collected under a
        # different cluster's transitions, undermining per-task world models.
        wm_metrics = {}
        btn_metrics = {}
        if len(seq_buffer) >= min_episodes:
            result = seq_buffer.sample_sequences(batch_size, task_id=task_id)
            if result is not None:
                obs_b, act_b, rew_b, done_b = result
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                rew_b = rew_b.to(device)
                done_b = done_b.to(device)

                encoder_active = (
                    is_trainable_online
                    and encoder_optimizer is not None
                    and step >= encoder_warmup_steps
                )
                if encoder_active:
                    base_encoder.train()
                    encoder_optimizer.zero_grad()

                # Sequential RSSM training — h propagated through L steps
                wm_metrics = active_wm_trainer.train_step_sequence(
                    obs_b, act_b, rew_b, done_b
                )
                task_wm_update_counts[task_id] = task_wm_update_counts.get(task_id, 0) + 1

                if encoder_active:
                    encoder_optimizer.step()
                    base_encoder.eval()

                # BTN training on states derived from sequential RSSM pass
                if active_btn_trainer is not None:
                    with torch.no_grad():
                        active_rssm.posterior_net.eval()
                        # Use middle of sequence for BTN training (h is warm by then)
                        mid = seq_len // 2
                        h_b, z_b = active_rssm.initial_state(batch_size, device)
                        pa = torch.zeros(batch_size, action_dim, device=device)
                        for t in range(mid):
                            h_b, z_b, _, _ = active_rssm.observe_step(
                                h_b, z_b, pa, obs_b[:, t, :]
                            )
                            pa = act_b[:, t, :]
                        active_rssm.posterior_net.train()
                        states_for_btn = active_rssm.get_state_features(h_b, z_b)
                    btn_metrics = active_btn_trainer.train_step(states_for_btn)

        # ---- Imagination rollout seeded from flat buffer ----
        # Seeding is filtered to THIS cluster's transitions where possible
        # (falls back to the full buffer if the cluster is too new — see
        # LegionReplayBuffer.sample). This is the actor-critic-side half of
        # the same isolation fix as above.
        flat_min = batch_size * 4
        if len(flat_buffer) >= flat_min:
            seed_batch = flat_buffer.sample(batch_size, task_id=task_id)
            seed_obs = torch.cat([b[0] for b in seed_batch], dim=0).to(device)
            seed_act = torch.tensor(
                np.stack([b[1] for b in seed_batch]), dtype=torch.float32
            ).to(device)
        else:
            seed_obs = current_obs_embed.expand(batch_size, -1).clone()
            seed_act = torch.zeros(batch_size, action_dim, device=device)

        # Seed RSSM from flat buffer obs
        with torch.no_grad():
            h_seed, z_seed = active_rssm.initial_state(batch_size, device)
            active_rssm.posterior_net.eval()
            h_seed, z_seed, _, _ = active_rssm.observe_step(
                h_seed, z_seed, seed_act, seed_obs
            )
            active_rssm.posterior_net.train()

        current_h = h_seed
        current_z = z_seed
        wm_hidden_img = None

        imagined_states = []
        imagined_rewards = []
        imagined_continues = []
        imagined_log_probs = []
        imagined_entropy = []
        imagined_raw_means = []

        s_current = active_rssm.get_state_features(current_h, current_z)
        imagined_states.append(s_current)

        for h_step in range(horizon):
            imagination_conditioning = None
            if active_btn is not None:
                imagination_conditioning, wm_hidden_img, _ = active_btn(
                    s_current, wm_hidden_img
                )

            dist = active_ac.get_action_dist(s_current, imagination_conditioning)
            raw_action = dist.rsample()
            action_img = torch.tanh(raw_action)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            current_h, current_z, _, reward_symlog, continue_logit = (
                active_rssm.imagine_step(current_h, current_z, action_img)
            )
            pred_continue = torch.sigmoid(continue_logit)
            s_current = active_rssm.get_state_features(current_h, current_z)

            imagined_states.append(s_current)
            imagined_rewards.append(reward_symlog.squeeze(-1))
            imagined_continues.append(pred_continue.squeeze(-1))
            imagined_log_probs.append(log_prob)
            imagined_entropy.append(entropy)
            imagined_raw_means.append(dist.mean)

        imagined_states_t = torch.stack(imagined_states)
        imagined_rewards_t = torch.stack(imagined_rewards)
        imagined_continues_t = torch.stack(imagined_continues)
        imagined_log_probs_t = torch.stack(imagined_log_probs)
        imagined_entropy_t = torch.stack(imagined_entropy)
        imagined_raw_means_t = torch.stack(imagined_raw_means)

        # ---- Actor/critic warmup gate ----
        # Only take gradient steps once THIS task's world model has accrued
        # wm_warmup_updates real training updates of its own (see comment at
        # wm_warmup_updates definition above for why this matters). Before
        # that, imagination still runs so the code path stays exercised and
        # the loop's timing/shape stays consistent, but no optimizer step is
        # taken — the policy simply doesn't move until the dynamics it's
        # being shaped against are themselves non-random.
        wm_updates_so_far = task_wm_update_counts.get(task_id, 0)
        actor_critic_ready = wm_updates_so_far >= wm_warmup_updates

        if actor_critic_ready:
            critic_loss, lambda_returns, traj_values = active_value_trainer.train_step(
                imagined_states_t.detach(),
                imagined_rewards_t.detach(),
                imagined_continues_t.detach(),
            )

            actor_metrics = active_actor_trainer.train_step(
                imagined_states_t[:-1].detach(),
                imagined_log_probs_t,
                imagined_entropy_t,
                lambda_returns,
                traj_values,
                imagined_raw_means=imagined_raw_means_t,
            )
        else:
            critic_loss = float("nan")
            actor_metrics = {"actor_loss": float("nan"), "policy_loss": float("nan")}

        if step % cfg["system"]["log_interval"] == 0:
            logger.log_imagination_step(
                step=step,
                actor_loss=actor_metrics["actor_loss"],
                critic_loss=critic_loss,
                pred_loss=wm_metrics.get("world_model_loss", 0.0),
                policy_entropy=imagined_entropy_t.mean().item(),
                rewards=symexp(imagined_rewards_t).detach(),
            )
            if btn_metrics:
                logger.writer.add_scalar("BTN/Loss", btn_metrics["btn_loss"], step)
                logger.writer.add_scalar("BTN/Entropy", btn_metrics["btn_entropy"], step)
                logger.writer.add_scalar(
                    "BTN/EntropyWeight", btn_metrics["btn_entropy_weight"], step
                )
            if "mean_penalty_loss" in actor_metrics:
                logger.writer.add_scalar(
                    "MuDreamer/Loss/ActionMeanPenalty",
                    actor_metrics["mean_penalty_loss"],
                    step,
                )

            # Diagnostic: how many DPMM clusters are active, and how the two
            # buffers are split across them. On a single-task run (e.g.
            # reach-v3 only) num_components should stay at 1 — if it climbs
            # above 1, the DPMM is spuriously fragmenting one task into
            # several undertrained heads.
            active_ids = list(task_rssms.keys())
            flat_sizes = {tid: flat_buffer.cluster_size(tid) for tid in active_ids}
            seq_sizes = {tid: seq_buffer.cluster_episode_count(tid) for tid in active_ids}
            logger.log_cluster_diagnostics(
                step=step,
                num_components=len(active_ids),
                flat_cluster_sizes=flat_sizes,
                seq_cluster_episode_counts=seq_sizes,
            )
            logger.writer.add_scalar(
                "Diagnostic/ActorCriticWarmupActive", int(not actor_critic_ready), step
            )
            logger.writer.add_scalar(
                "Diagnostic/WMUpdatesForActiveTask", wm_updates_so_far, step
            )
            print(
                f"  [DIAG] Active clusters: {len(active_ids)} | "
                f"flat sizes: {flat_sizes} | seq episode counts: {seq_sizes}"
            )

            warmup_tag = (
                f"[enc warmup {step}/{encoder_warmup_steps}]"
                if is_trainable_online and step < encoder_warmup_steps else ""
            )
            ac_tag = (
                f"[AC warmup: WM updates {wm_updates_so_far}/{wm_warmup_updates}]"
                if not actor_critic_ready else ""
            )
            print(
                f"Step {step}/{total_steps} | Task: {task_id} | "
                f"Reward: {reward:.4f} | Critic: {critic_loss:.4f} | "
                f"Actor: {actor_metrics['actor_loss']:.4f} | "
                f"WM: {wm_metrics.get('world_model_loss', float('nan')):.4f} | "
                f"KL: {wm_metrics.get('kl_loss', float('nan')):.4f} | "
                f"Episodes: {len(seq_buffer)} "
                f"{warmup_tag} {ac_tag}"
            )

        if step % eval_interval == 0 and step > 0:
            eval_metrics = evaluate_policy(
                env_class=mt1.train_classes[task_name],
                task=task,
                rssm=active_rssm,
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
            dist_str = ""
            if "mean_final_distance" in eval_metrics:
                logger.log_eval_precision(step, eval_metrics["mean_final_distance"])
                dist_str = f" | Mean Final Distance: {eval_metrics['mean_final_distance']:.4f}"

            sat_str = ""
            if "mean_abs_action" in eval_metrics:
                logger.writer.add_scalar(
                    "Eval/Diagnostic/MeanAbsAction", eval_metrics["mean_abs_action"], step
                )
                logger.writer.add_scalar(
                    "Eval/Diagnostic/FracSaturatedAction",
                    eval_metrics["frac_saturated_action"],
                    step,
                )
                sat_str = (
                    f" | MeanAbsAction: {eval_metrics['mean_abs_action']:.4f} "
                    f"| FracSaturated: {eval_metrics['frac_saturated_action']:.2f}"
                )

            cond_str = ""
            if "mean_conditioning_std_over_time" in eval_metrics:
                logger.writer.add_scalar(
                    "Eval/Diagnostic/ConditioningStdOverTime",
                    eval_metrics["mean_conditioning_std_over_time"],
                    step,
                )
                cond_str = (
                    f" | BTN Cond StdOverTime: "
                    f"{eval_metrics['mean_conditioning_std_over_time']:.5f}"
                )

            print(
                f"  [EVAL] Mean Episode Reward: {eval_metrics['mean_episode_reward']:.4f} | "
                f"Success Rate: {eval_metrics['success_rate']:.2f}"
                f"{dist_str}{sat_str}{cond_str}"
            )

        if terminated or truncated:
            env.reset()
            current_pixels = get_aligned_pixels()
            current_obs_embed = encode(current_pixels)
            live_h, live_z = task_rssms[task_id].initial_state(1, device)
            prev_action = torch.zeros(1, action_dim, device=device)
        else:
            current_pixels = next_pixels
            current_obs_embed = next_obs_embed

    env.close()
    logger.close()
    print("[*] Execution completed successfully.")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    run_legion_jepa_dreamer_loop()