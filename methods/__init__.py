"""Method registry stubs for custom semi-supervised algorithms."""

from .trico import train_trico

METHODS = {
    "trico": train_trico,
}

__all__ = ["METHODS", "train_trico"]
