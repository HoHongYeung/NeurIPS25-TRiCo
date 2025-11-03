import torch
import torch.nn as nn


class CLIPBackbone(nn.Module):
    """Frozen CLIP ViT-B/16 encoder with CPU/GPU support and safe fallback."""

    def __init__(self, model_name: str = "ViT-B/16", pretrained: bool = True):
        super().__init__()
        encoder = None
        if pretrained:
            try:
                import clip  # noqa: WPS433

                device = "cuda" if torch.cuda.is_available() else "cpu"
                encoder, _ = clip.load(model_name, device=device, jit=False)
                encoder = encoder.eval().float()
            except Exception as exc:  # pragma: no cover
                print("[Warning] CLIP unavailable, fallback to ViT-B/16:", exc)
        if encoder is None:
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
        if hasattr(self.encoder, "encode_image"):
            return self.encoder.encode_image(x)
        if hasattr(self.encoder, "forward_features"):
            return self.encoder.forward_features(x)
        return self.encoder(x)
