import torch
import torch.nn.functional as F


def mutual_information(logits_list, eps: float = 1e-8):
    """Estimate epistemic MI(x) = H[p_bar(y)] - E[H[p(y|theta)]]."""
    if len(logits_list) == 0:
        raise ValueError("logits_list must contain at least one tensor")
    probs = torch.stack([F.softmax(l, dim=-1) for l in logits_list])  # [K, B, C]
    p_mean = probs.mean(0)
    log_p_mean = torch.log(p_mean.clamp_min(eps))
    H_mean = -(p_mean * log_p_mean).sum(-1)
    log_probs = torch.log(probs.clamp_min(eps))
    H_exp = -(probs * log_probs).sum(-1).mean(0)
    return (H_mean - H_exp).detach()


def entropy(logits, eps: float = 1e-8):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(eps))).sum(-1).mean()
