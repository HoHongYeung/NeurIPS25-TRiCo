import torch
import torch.nn as nn


class ViTSupervisedBackbone(nn.Module):
    """Frozen supervised ViT-B/16 baseline with optional pretrained weights."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import vit_b_16

            encoder = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception as exc:  # pragma: no cover
            print("[Warning] torchvision ViT-B/16 unavailable, using identity backbone:", exc)
            encoder = nn.Identity()
        for param in encoder.parameters() if hasattr(encoder, "parameters") else []:
            param.requires_grad = False
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            return self.encoder.forward_features(x)
        return self.encoder(x)
