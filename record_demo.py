"""
Loads a saved checkpoint (defaults to checkpoint_best.pt) and records an
evaluation rollout as an MP4, with the BTN's decoded semantic triple
("inner monologue") overlaid as live text on each frame.

This is a standalone script — it does NOT touch training state, buffers,
or optimizers. It only needs the RSSM, actor-critic, and BTN weights for
one task cluster, loaded read-only in eval mode.

Usage:
    python record_demo.py
    python record_demo.py --checkpoint checkpoints/legion_jepa_dreamer/checkpoint_best.pt --episodes 5 --out pitch_demo.mp4

Requires (install once, on your machine, not needed for training itself):
    pip install imageio imageio-ffmpeg pillow
"""

import argparse
import os
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
import imageio

import metaworld

from src.encoders import build_encoder
from src.imagination import RSSMCore, MuDreamerActorCritic
from src.btn import MinimalBTN, DEFAULT_VOCAB


def load_checkpoint_for_task(
    ckpt_path, task_id, obs_embed_dim, action_dim, deter_dim, stoch_dim,
    stoch_classes, hidden_dim, state_dim, conditioning_dim, use_btn,
    btn_cfg, device,
):
    # weights_only=False: this is our own checkpoint (saved by run_pipeline.py),
    # not a third-party download — same reasoning as the fix in run_pipeline.py.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if task_id not in ckpt["tasks"]:
        available = list(ckpt["tasks"].keys())
        raise ValueError(
            f"task_id {task_id} not found in checkpoint. Available task IDs: {available}"
        )
    task_state = ckpt["tasks"][task_id]

    rssm = RSSMCore(
        obs_embed_dim=obs_embed_dim,
        action_dim=action_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        stoch_classes=stoch_classes,
        hidden_dim=hidden_dim,
    ).to(device)
    rssm.load_state_dict(task_state["rssm"])
    rssm.eval()

    ac = MuDreamerActorCritic(
        state_dim=state_dim, action_dim=action_dim, conditioning_dim=conditioning_dim
    ).to(device)
    ac.load_state_dict(task_state["actor_critic"])
    ac.eval()

    btn = None
    if use_btn and "btn" in task_state:
        btn = MinimalBTN(
            embed_dim=state_dim,
            n_subjects=btn_cfg.get("n_subjects", 30),
            n_predicates=btn_cfg.get("n_predicates", 15),
            n_objects=btn_cfg.get("n_objects", 30),
            vocab_embed_dim=btn_cfg.get("vocab_embed_dim", 64),
            conditioning_dim=btn_cfg["conditioning_dim"],
        ).to(device)
        btn.load_state_dict(task_state["btn"])
        btn.eval()

    return rssm, ac, btn, ckpt.get("step"), ckpt.get("best_metric_value")


def draw_narration(frame_np, text, success_so_far):
    """
    Burns a text bar into the bottom of the frame showing the BTN's
    currently decoded (subject, predicate, object) triple — the "inner
    monologue" narration. Turns green once MetaWorld's own success flag
    has fired at any point in the episode, so a viewer can see the exact
    moment of success on the video without needing extra UI.
    """
    img = Image.fromarray(frame_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        # Falls back gracefully if that specific font isn't installed —
        # still produces a working video, just with a smaller default font.
        font = ImageFont.load_default()
    bar_h = 36
    draw.rectangle([0, img.height - bar_h, img.width, img.height], fill=(0, 0, 0))
    color = (80, 220, 120) if success_so_far else (255, 255, 255)
    draw.text((8, img.height - bar_h + 8), text, fill=color, font=font)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/legion_jepa_dreamer.yaml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Defaults to <checkpoint_dir>/checkpoint_best.pt from the config.",
    )
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--task_name", default="reach-v3")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", default="demo_rollout.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resize", type=int, nargs=2, default=[112, 112])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["system"]["device"] if torch.cuda.is_available() else "cpu")
    encoder_type = cfg["lewm_encoder"].get("encoder_type", "lewm")
    use_btn = cfg.get("btn", {}).get("enabled", False)

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["system"]["checkpoint_dir"], "checkpoint_best.pt"
    )
    print(f"[*] Loading checkpoint: {checkpoint_path}")

    obs_embed_dim = cfg["lewm_encoder"]["embed_dim"]
    action_dim = cfg["environment"]["action_dim"]
    deter_dim = cfg["mudreamer_core"]["rssm"]["deter_dim"]
    stoch_dim = cfg["mudreamer_core"]["rssm"]["stoch_dim"]
    stoch_classes = cfg["mudreamer_core"]["rssm"]["stoch_classes"]
    hidden_dim = cfg["mudreamer_core"]["networks"]["hidden_dim"]
    conditioning_dim = cfg["btn"]["conditioning_dim"] if use_btn else 0
    state_dim = deter_dim + stoch_dim * stoch_classes

    stn_aligner, base_encoder, is_frozen, _ = build_encoder(
        encoder_type=encoder_type,
        embed_dim=obs_embed_dim,
        device=device,
        checkpoint_path=cfg["lewm_encoder"].get("pretrained_weights"),
    )
    base_encoder.eval()
    stn_aligner.eval()

    rssm, ac, btn, ckpt_step, best_metric = load_checkpoint_for_task(
        checkpoint_path, args.task_id, obs_embed_dim, action_dim, deter_dim,
        stoch_dim, stoch_classes, hidden_dim, state_dim, conditioning_dim,
        use_btn, cfg.get("btn", {}), device,
    )
    print(f"[*] Loaded checkpoint from training step {ckpt_step}, "
          f"best_metric_value at save time: {best_metric}")

    mt1 = metaworld.MT1(args.task_name)
    env_class = mt1.train_classes[args.task_name]
    task = mt1.train_tasks[0]

    subjects = DEFAULT_VOCAB["subjects"]
    predicates = DEFAULT_VOCAB["predicates"]
    objects_ = DEFAULT_VOCAB["objects"]

    frames = []
    total_success = 0
    total_reward = 0.0
    final_distances = []
    resize = tuple(args.resize)

    for ep in range(args.episodes):
        env = env_class(render_mode="rgb_array")
        env.set_task(task)
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        last_info = {}
        success_so_far = False

        h, z = rssm.initial_state(1, device)
        wm_hidden = None
        prev_action = torch.zeros(1, action_dim, device=device)

        while not done:
            raw_frame = env.render()
            img_tensor = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            pixel_input = torch.nn.functional.interpolate(
                img_tensor, size=resize, mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                aligned = stn_aligner(pixel_input)
                obs_embed = base_encoder(aligned)
                rssm.posterior_net.eval()
                h, z, _, _ = rssm.observe_step(h, z, prev_action, obs_embed)
                state = rssm.get_state_features(h, z)

                conditioning = None
                narration = ""
                if btn is not None:
                    conditioning, wm_hidden, _ = btn(state, wm_hidden)
                    s_idx, p_idx, o_idx = btn.decode_triple_hard(state)
                    subj = subjects[int(s_idx.item()) % len(subjects)]
                    pred = predicates[int(p_idx.item()) % len(predicates)]
                    obj = objects_[int(o_idx.item()) % len(objects_)]
                    narration = f"({subj}, {pred}, {obj})"

                action = ac.get_action(state, conditioning, deterministic=True)
                real_action = action.cpu().numpy().flatten()[:action_dim]

            frames.append(draw_narration(raw_frame.copy(), narration, success_so_far))

            obs, reward, terminated, truncated, info = env.step(real_action)
            ep_reward += reward
            last_info = info
            if info.get("success", 0.0) > 0.5:
                success_so_far = True
            done = terminated or truncated
            prev_action = action

        env.close()
        total_reward += ep_reward
        total_success += int(last_info.get("success", 0.0) > 0.5)
        for dist_key in ("obj_to_target", "distance_to_target", "reachDist", "target_to_obj"):
            if dist_key in last_info:
                final_distances.append(float(last_info[dist_key]))
                break
        print(f"[*] Episode {ep + 1}/{args.episodes}: "
              f"reward={ep_reward:.2f}, success={last_info.get('success', 0.0)}")

    print(f"[*] Mean episode reward: {total_reward / args.episodes:.2f}")
    print(f"[*] Success rate: {total_success / args.episodes:.2f}")
    if final_distances:
        print(f"[*] Mean final distance: {sum(final_distances) / len(final_distances):.4f}")

    print(f"[*] Writing {len(frames)} frames to {args.out} at {args.fps} fps...")
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8)
    print(f"[*] Saved demo video to {args.out}")


if __name__ == "__main__":
    main()