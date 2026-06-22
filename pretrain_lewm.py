import os
import argparse
import numpy as np
import torch

import metaworld

from src.perception import CameraViewAligner, LeWMEncoder, LeWMPredictor, LeWMTrainer
from src.lifelong import PixelPairBuffer


def collect_random_pixel_pairs(env, action_dim, stn_aligner, device, num_steps, resize=(112, 112)):
    """
    Rolls out random actions in the environment to collect diverse
    (current_pixels, action, next_pixels) pairs for offline JEPA pretraining.
    Random actions are sufficient here since LeWM's objective is task-agnostic
    physical dynamics, not reward-driven behavior.
    """
    pairs = []
    obs, info = env.reset()

    def get_aligned_pixels():
        raw_frame = env.render()
        if raw_frame is None:
            raise RuntimeError("Render returned None. Check render_mode='rgb_array'")
        img_tensor = torch.from_numpy(raw_frame.copy()).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        pixel_input = torch.nn.functional.interpolate(
            img_tensor, size=resize, mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            aligned = stn_aligner(pixel_input)
        return aligned

    current_pixels = get_aligned_pixels()

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_pixels = get_aligned_pixels()

        action_padded = np.zeros(action_dim, dtype=np.float32)
        n = min(len(action), action_dim)
        action_padded[:n] = action[:n]

        pairs.append((current_pixels.cpu(), action_padded, next_pixels.cpu()))

        if terminated or truncated:
            env.reset()
            current_pixels = get_aligned_pixels()
        else:
            current_pixels = next_pixels

        if (step + 1) % 500 == 0:
            print(f"  collected {step + 1}/{num_steps} pixel pairs")

    return pairs


def pretrain(
    tasks,
    embed_dim=512,
    action_dim=4,
    steps_per_task=2000,
    train_epochs=20,
    batch_size=64,
    lr=3e-4,
    ema_decay=0.996,
    sigreg_weight=1.0,
    pred_weight=1.0,
    device="cuda",
    out_path="weights/lewm_kuka_shared_base.pt",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[*] Pretraining LeWM on device: {device}")

    stn_aligner = CameraViewAligner().to(device)
    encoder = LeWMEncoder(embed_dim=embed_dim).to(device)
    predictor = LeWMPredictor(embed_dim=embed_dim, action_dim=action_dim).to(device)

    trainer = LeWMTrainer(
        online_encoder=encoder,
        predictor=predictor,
        action_dim=action_dim,
        lr=lr,
        ema_decay=ema_decay,
        sigreg_weight=sigreg_weight,
        pred_weight=pred_weight,
    )

    buffer = PixelPairBuffer(max_size=steps_per_task * len(tasks) + 100)

    # ---- Phase 1a: collect diverse pixel pairs across all tasks ----
    for task_name in tasks:
        print(f"[*] Collecting rollouts for task: {task_name}")
        mt1 = metaworld.MT1(task_name)
        env = mt1.train_classes[task_name](render_mode="rgb_array")
        task = mt1.train_tasks[0]
        env.set_task(task)

        pairs = collect_random_pixel_pairs(
            env, action_dim, stn_aligner, device, num_steps=steps_per_task
        )
        for p in pairs:
            buffer.add(*p)

        env.close()

    print(f"[*] Collected {len(buffer)} total pixel pairs across {len(tasks)} tasks")

    # ---- Phase 1b: train encoder+predictor offline ----
    steps_per_epoch = max(len(buffer) // batch_size, 1)
    print(f"[*] Training for {train_epochs} epochs x {steps_per_epoch} steps/epoch")

    for epoch in range(train_epochs):
        epoch_losses = []
        epoch_pred_losses = []
        epoch_sigreg_losses = []

        for _ in range(steps_per_epoch):
            batch = buffer.sample(batch_size)
            cur_px_b = torch.cat([b[0] for b in batch], dim=0).to(device)
            act_b = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).to(device)
            next_px_b = torch.cat([b[2] for b in batch], dim=0).to(device)

            metrics = trainer.train_step(cur_px_b, act_b, next_px_b)
            epoch_losses.append(metrics["lewm_loss"])
            epoch_pred_losses.append(metrics["lewm_pred_loss"])
            epoch_sigreg_losses.append(metrics["lewm_sigreg_loss"])

        print(
            f"Epoch {epoch + 1}/{train_epochs} | "
            f"Loss: {np.mean(epoch_losses):.5f} | "
            f"Pred: {np.mean(epoch_pred_losses):.5f} | "
            f"SIGReg: {np.mean(epoch_sigreg_losses):.5f}"
        )

    # ---- Save checkpoint ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "aligner_state_dict": stn_aligner.state_dict(),
            "embed_dim": embed_dim,
            "action_dim": action_dim,
        },
        out_path,
    )
    print(f"[*] Saved pretrained LeWM checkpoint to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["reach-v3", "push-v3", "pick-place-v3", "door-open-v3"],
        help="MetaWorld task names to collect rollouts from",
    )
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--steps_per_task", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_path", type=str, default="weights/lewm_kuka_shared_base.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pretrain(
        tasks=args.tasks,
        embed_dim=args.embed_dim,
        action_dim=args.action_dim,
        steps_per_task=args.steps_per_task,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out_path=args.out_path,
    )