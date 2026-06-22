"""
Pluggable encoder registry for JEPA-Dreamer ablation study.

Each encoder must implement:
    encode(pixel_tensor: Tensor (B, C, H, W)) -> Tensor (B, embed_dim)

Available encoders:
    "lewm"      : Pretrained JEPA + SIGReg encoder (default, two-phase)
    "mudreamer"  : Plain CNN with BatchNorm (original MuDreamer encoder style)
    "random"    : Randomly initialized frozen CNN (baseline / lower bound)
"""

import torch
import torch.nn as nn
import os


class PlainCNNEncoder(nn.Module):
    """
    Reconstruction-free CNN encoder in the style of the original MuDreamer
    representation network — no JEPA, no SIGReg, just a standard CNN with
    BatchNorm to prevent collapse (as used in the MuDreamer paper).
    Trained end-to-end via the world model's prediction losses, not offline.
    """
    def __init__(self, c_in=3, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(embed_dim),
            # BatchNorm1d here is the critical MuDreamer addition that prevents
            # representation collapse without a reconstruction decoder
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class RandomFrozenEncoder(nn.Module):
    """
    Randomly initialized, permanently frozen CNN. Used as a lower-bound
    ablation — establishes what DPMM + MuDreamer achieves with zero perceptual
    learning, attributing any gains purely to the downstream RL components.
    """
    def __init__(self, c_in=3, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(embed_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)


def build_encoder(encoder_type, embed_dim, device, checkpoint_path=None):
    """
    Factory function. Returns (aligner, encoder, is_frozen, is_trainable_online).

    is_frozen: True means encoder.eval() + no grad — perception never updates
               during Phase 2 (run_pipeline.py).
    is_trainable_online: True means the encoder should receive gradient updates
               during Phase 2 alongside the world model. Only applies to
               "mudreamer" mode where encoder and world model train jointly.
    """
    from src.perception import CameraViewAligner, LeWMEncoder

    aligner = CameraViewAligner().to(device)

    if encoder_type == "lewm":
        encoder = LeWMEncoder(embed_dim=embed_dim).to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading pretrained LeWM weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            aligner.load_state_dict(ckpt["aligner_state_dict"])
        else:
            print(
                f"[!!! WARNING !!!] LeWM encoder selected but no checkpoint found at "
                f"'{checkpoint_path}'. Weights are RANDOM. Run pretrain_lewm.py first."
            )
        encoder.eval()
        aligner.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        for p in aligner.parameters():
            p.requires_grad = False
        return aligner, encoder, True, False

    elif encoder_type == "mudreamer":
        encoder = PlainCNNEncoder(embed_dim=embed_dim).to(device)
        # Not frozen — trains jointly with world model via prediction losses
        # (BatchNorm inside prevents collapse, as per MuDreamer paper)
        return aligner, encoder, False, True

    elif encoder_type == "random":
        encoder = RandomFrozenEncoder(embed_dim=embed_dim).to(device)
        encoder.eval()
        aligner.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        for p in aligner.parameters():
            p.requires_grad = False
        return aligner, encoder, True, False

    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. "
            f"Choose from: 'lewm', 'mudreamer', 'random'"
        )