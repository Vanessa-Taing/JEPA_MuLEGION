import torch
import torch.nn.functional as F
import torch.nn as nn

class CameraViewAligner(nn.Module):
    """
    A lightweight spatial transformer module that scales, rotates, and shifts 
    disparate KUKA simulator visual renders into a uniform viewpoint.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Localization network to estimate the affine transformation frame matrix
        # self.localization = nn.Sequential(
        #     nn.Conv2d(in_channels, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Flatten(),
        #     nn.Linear(10 * 22 * 22, 32), # Size matches 112x112 inputs
        #     nn.ReLU(True)
        # )
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
        
        # Regressor head initialized to output an Identity Transform mapping matrix
        self.fc_loc = nn.Linear(32, 2 * 3)
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x):
        # 1. Evaluate perspective offsets 
        features = self.localization(x)
        theta = self.fc_loc(features).view(-1, 2, 3)
        
        # 2. Construct transformation sampling meshgrid
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # 3. Sample pixels with bi-linear interpolation to normalize view geometry
        x_aligned = F.grid_sample(x, grid, align_corners=False)
        return x_aligned
    
class LeWMEncoder(nn.Module):
    """
    Conceptual extraction of LeWM's lightweight pixel encoder 
    Stabilized natively via SIGReg Gaussian regularization.
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
        # x shape: (B, C, H, W)
        return self.net(x)

class UnifiedVisionPipeline(nn.Module):
    """
    Combines perspective correction with stable feature encoding.
    """
    def __init__(self, frozen_lewm_encoder):
        super().__init__()
        self.aligner = CameraViewAligner()
        self.encoder = frozen_lewm_encoder # Static LeWorldModel block
        
    def forward(self, raw_camera_pixels):
        # View shifts are corrected prior to calculating latents
        standardized_pixels = self.aligner(raw_camera_pixels)
        return self.encoder(standardized_pixels)
