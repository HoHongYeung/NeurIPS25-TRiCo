import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 Vision Transformer encoder with safe fallbacks."""

    def __init__(self, model_name: str = "dinov2_vits14", pretrained: bool = True):
        super().__init__()
        encoder = None
        try:
            import torch.hub  # noqa: WPS433 (third-party optional dependency)

            encoder = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=pretrained,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            print("[Warning] DINOv2 load failed, fallback to ViT-B/16:", exc)
            try:
                from torchvision.models import vit_b_16

                encoder = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
            except Exception as vit_exc:  # pragma: no cover
                print("[Warning] torchvision ViT-B/16 unavailable, using identity backbone:", vit_exc)
                encoder = nn.Identity()
        for param in encoder.parameters() if hasattr(encoder, "parameters") else []:
            param.requires_grad = False
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            return self.encoder.forward_features(x)
        return self.encoder(x)
