import torch
import torch.nn as nn


class ResNetSupervisedBackbone(nn.Module):
    """Frozen supervised ResNet-50 baseline with identity classification head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet50

            encoder = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception as exc:  # pragma: no cover
            print("[Warning] torchvision ResNet50 unavailable, using identity backbone:", exc)
            encoder = nn.Identity()
        if hasattr(encoder, "fc") and isinstance(encoder.fc, nn.Module):
            encoder.fc = nn.Identity()
        for param in encoder.parameters() if hasattr(encoder, "parameters") else []:
            param.requires_grad = False
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
