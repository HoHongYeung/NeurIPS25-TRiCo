import torch
import torch.nn as nn
import torch.nn.functional as F


class Student(nn.Module):
    """Lightweight MLP student operating on frozen encoder features."""

    def __init__(self, in_dim=768, num_classes=10, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, dropout_pass=False):
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=dropout_pass or self.training)
        return self.fc2(x)


def build_students(feature_dim, num_classes, **kwargs):
    """Factory helper to instantiate a pair of students with shared config."""
    kwargs.setdefault("in_dim", feature_dim)
    kwargs.setdefault("num_classes", num_classes)
    model_kwargs = {k: v for k, v in kwargs.items() if k in {"in_dim", "num_classes", "hidden_dim", "dropout"}}
    student_a = Student(**model_kwargs)
    student_b = Student(**model_kwargs)
    return student_a, student_b
