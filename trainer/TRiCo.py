#!coding:utf-8
"""Trainer shim that wraps the TRiCo game-theoretic co-training routine."""

from __future__ import annotations

import inspect
from typing import Iterable, Optional

import torch

from backbone_architecture import ENCODER_FACTORY
from methods.trico import train_trico


class Trainer:
    def __init__(self, device, config, num_classes: int):
        print("TRiCo Trainer")
        self.device = torch.device(device)
        self.config = config
        self.num_classes = num_classes
        self.pretrained = getattr(config, "encoder_pretrained", True)
        self.encoder1 = self._build_encoder(config.encoder1, index=1)
        self.encoder2 = self._build_encoder(config.encoder2, index=2)
        self.students: Optional[tuple[torch.nn.Module, torch.nn.Module]] = None
        self.teacher: Optional[torch.nn.Module] = None

    def _build_encoder(self, name: Optional[str], index: int):
        if not name:
            raise ValueError(
                f"encoder{index} must be specified for TRiCo (available: {sorted(ENCODER_FACTORY.keys())})"
            )
        parts = str(name).split(":", 1)
        key = parts[0].lower()
        extra = parts[1] if len(parts) > 1 else None
        if key not in ENCODER_FACTORY:
            raise ValueError(
                f"Unknown encoder '{name}'. Supported options: {sorted(ENCODER_FACTORY.keys())}"
            )
        encoder_cls = ENCODER_FACTORY[key]
        kwargs = {"pretrained": self.pretrained}
        if extra is not None:
            signature = inspect.signature(encoder_cls.__init__)
            if "model_name" in signature.parameters:
                kwargs["model_name"] = extra
            elif "variant" in signature.parameters:
                kwargs["variant"] = extra
            else:  # pragma: no cover - defensive branch
                print(f"[Warning] extra encoder argument '{extra}' ignored for {name}")
        encoder = encoder_cls(**kwargs)
        return encoder

    def train(self, *_args, **_kwargs):  # pragma: no cover - interface placeholder
        raise NotImplementedError("TRiCo Trainer does not expose step-wise training.")

    def test(self, *_args, **_kwargs):  # pragma: no cover - interface placeholder
        raise NotImplementedError("TRiCo Trainer does not expose separate evaluation loop.")

    def loop(
        self,
        epochs: int,
        label_loader: Iterable,
        unlabel_loader: Iterable,
        eval_loader: Iterable,
        scheduler=None,
    ) -> None:
        del scheduler  # Not used, but kept for interface compatibility.
        students = train_trico(
            self.encoder1,
            self.encoder2,
            label_loader,
            unlabel_loader,
            eval_loader,
            num_classes=self.num_classes,
            epochs=epochs,
            K=getattr(self.config, "trico_k", 5),
            eps=getattr(self.config, "trico_eps", 1.0),
            lr_student=getattr(self.config, "trico_lr_student", 3e-2),
            lr_teacher=getattr(self.config, "trico_lr_teacher", 1e-2),
            device=str(self.device),
            feature_dim=getattr(self.config, "feature_dim", 768),
            mi_gamma=getattr(self.config, "trico_mi_gamma", 0.5),
        )
        student1, student2, teacher = students
        self.students = (student1, student2)
        self.teacher = teacher
        print("TRiCo training completed.")
