import torch
import torch.nn as nn


class Teacher(nn.Module):
    """Meta-learned strategy scheduler controlling tau_MI, lambda_u, lambda_adv."""

    def __init__(self, init_tau=0.05, init_lambda_u=0.5, init_lambda_adv=0.5):
        super().__init__()
        init = torch.tensor([init_tau, init_lambda_u, init_lambda_adv], dtype=torch.float32)
        self.params = nn.Parameter(torch.logit(init.clamp(1e-3, 1 - 1e-3)))

    def forward(self):
        tau, lambda_u, lambda_adv = torch.sigmoid(self.params)
        return tau, lambda_u, lambda_adv

    def meta_update(self, val_loss):
        scale = float(val_loss.detach()) if torch.is_tensor(val_loss) else float(val_loss)
        for p in self.parameters():
            if p.grad is None:
                continue
            p.grad.mul_(scale)
