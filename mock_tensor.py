import torch
import os

# 1. Create directory structure
os.makedirs("weights", exist_ok=True)

# 2. Build dummy tensor dictionary mirroring a 512-dim LeWM base
mock_weights = {
    "encoder.weight": torch.randn(512, 3, 64, 64),
    "encoder.bias": torch.zeros(512),
    "feature_projection.weight": torch.randn(512, 512)
}

# 3. Save mock weights to target file path
torch.save(mock_weights, "weights/lewm_kuka_shared_base.pt")
print("Saved temporary mock weights for testing pipeline execution!")
