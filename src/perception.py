import torch
import torch.nn.functional as F
import torch.nn as nn
import copy


class CameraViewAligner(nn.Module):
    """
    A lightweight spatial transformer module that scales, rotates, and shifts
    disparate KUKA simulator visual renders into a uniform viewpoint.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Linear(32, 2 * 3)
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x):
        features = self.localization(x)
        theta = self.fc_loc(features).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aligned = F.grid_sample(x, grid, align_corners=False)
        return x_aligned


class LeWMEncoder(nn.Module):
    """
    Lightweight pixel encoder, trained via JEPA-style predictive loss + SIGReg
    (instead of being left randomly initialized and frozen).
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
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class LeWMPredictor(nn.Module):
    """
    JEPA predictor head: predicts the target encoder's embedding of the NEXT
    frame, given the online encoder's embedding of the CURRENT frame and the
    action taken. This is what gives the encoder a non-reconstructive training
    signal instead of sitting randomly initialized forever.
    """
    def __init__(self, embed_dim=512, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, online_embed, action):
        return self.net(torch.cat([online_embed, action], dim=-1))


class UnifiedVisionPipeline(nn.Module):
    """
    Combines perspective correction with stable feature encoding.
    """
    def __init__(self, frozen_lewm_encoder):
        super().__init__()
        self.aligner = CameraViewAligner()
        self.encoder = frozen_lewm_encoder

    def forward(self, raw_camera_pixels):
        standardized_pixels = self.aligner(raw_camera_pixels)
        return self.encoder(standardized_pixels)


def sigreg_loss(embeddings, num_sketches=64):
    """
    Simplified Sketched Isotropic Gaussian Regularizer.

    Projects the batch of embeddings onto `num_sketches` random 1D directions,
    then penalizes the deviation of each projected distribution's mean/std
    from a standard normal (mean 0, std 1). This discourages collapse
    (all embeddings identical -> std -> 0) and discourages low-rank
    degenerate solutions, without requiring an explicit covariance matrix
    inversion or eigendecomposition.

    embeddings: (B, D)
    """
    B, D = embeddings.shape
    device = embeddings.device

    # Random unit-norm projection directions, regenerated each call
    # (cheap, avoids needing to store/persist a fixed sketch matrix)
    directions = torch.randn(D, num_sketches, device=device)
    directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-8)

    projections = embeddings @ directions  # (B, num_sketches)

    proj_mean = projections.mean(dim=0)          # (num_sketches,)
    proj_std = projections.std(dim=0) + 1e-6      # (num_sketches,)

    mean_penalty = torch.mean(proj_mean ** 2)
    std_penalty = torch.mean((proj_std - 1.0) ** 2)

    return mean_penalty + std_penalty


class LeWMTrainer:
    """
    Trains the LeWM encoder + predictor using:
      - JEPA predictive loss (online encoder + predictor -> target encoder embedding)
      - SIGReg regularization (keeps embedding distribution isotropic, avoids collapse)

    The target encoder is an EMA copy of the online encoder, updated after
    every training step. Only the online encoder (and predictor) receive
    gradients; the target encoder never does (mirrors BYOL/JEPA practice).
    """
    def __init__(
        self,
        online_encoder,
        predictor,
        action_dim,
        lr=3e-4,
        ema_decay=0.996,
        sigreg_weight=1.0,
        pred_weight=1.0,
    ):
        self.online_encoder = online_encoder
        self.predictor = predictor
        self.action_dim = action_dim
        self.ema_decay = ema_decay
        self.sigreg_weight = sigreg_weight
        self.pred_weight = pred_weight

        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        self.optimizer = torch.optim.Adam(
            list(self.online_encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr,
        )

    @torch.no_grad()
    def _update_target_encoder(self):
        for online_p, target_p in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1 - self.ema_decay)

    def train_step(self, current_pixels, actions, next_pixels):
        """
        current_pixels, next_pixels: (B, C, H, W) aligned/normalized image tensors
        actions: (B, action_dim)
        """
        self.online_encoder.train()
        self.optimizer.zero_grad()

        online_embed = self.online_encoder(current_pixels)            # (B, D), grad flows
        with torch.no_grad():
            target_embed = self.target_encoder(next_pixels)            # (B, D), no grad

        predicted_embed = self.predictor(online_embed, actions)

        # JEPA predictive loss: match predictor output to target encoder's embedding
        pred_loss = F.mse_loss(predicted_embed, target_embed)

        # SIGReg on the online embeddings (prevents the online encoder from collapsing)
        reg_loss = sigreg_loss(online_embed)

        loss = self.pred_weight * pred_loss + self.sigreg_weight * reg_loss
        loss.backward()
        self.optimizer.step()

        self._update_target_encoder()
        self.online_encoder.eval()  # back to eval mode for downstream RL use

        return {
            "lewm_loss": loss.item(),
            "lewm_pred_loss": pred_loss.item(),
            "lewm_sigreg_loss": reg_loss.item(),
        }