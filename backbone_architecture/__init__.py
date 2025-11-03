from .dino import DINOv2Backbone
from .mae import MAEBackbone
from .simclr import SimCLRBackbone
from .moco import MoCoV3Backbone
from .byol import BYOLBackbone
from .swav import SwAVBackbone
from .clip_encoder import CLIPBackbone
from .vit_supervised import ViTSupervisedBackbone
from .resnet_supervised import ResNetSupervisedBackbone

ENCODER_FACTORY = {
    "dinov2": DINOv2Backbone,
    "mae": MAEBackbone,
    "simclr": SimCLRBackbone,
    "moco": MoCoV3Backbone,
    "mocov3": MoCoV3Backbone,
    "byol": BYOLBackbone,
    "swav": SwAVBackbone,
    "clip": CLIPBackbone,
    "vit": ViTSupervisedBackbone,
    "vit-supervised": ViTSupervisedBackbone,
    "vit_b16": ViTSupervisedBackbone,
    "resnet": ResNetSupervisedBackbone,
    "resnet50": ResNetSupervisedBackbone,
}

__all__ = [
    "DINOv2Backbone",
    "MAEBackbone",
    "SimCLRBackbone",
    "MoCoV3Backbone",
    "BYOLBackbone",
    "SwAVBackbone",
    "CLIPBackbone",
    "ViTSupervisedBackbone",
    "ResNetSupervisedBackbone",
    "ENCODER_FACTORY",
]
