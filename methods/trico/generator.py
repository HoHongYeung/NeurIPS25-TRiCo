import torch
import torch.nn.functional as F

from .utils import mutual_information, entropy


def entropy_guided_perturb(x, model, eps=1.0, gamma=0.5, mi_passes=4):
    """Generate embedding-space perturbation that maximizes entropy + MI."""
    if mi_passes < 1:
        raise ValueError("mi_passes must be >= 1")
    x_adv = x.detach().clone().requires_grad_(True)
    logits_samples = [model(x_adv, dropout_pass=True) for _ in range(mi_passes)]
    ent = entropy(logits_samples[0])
    mi = mutual_information(logits_samples)
    loss = ent + gamma * mi.mean()
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    delta = eps * grad.sign()
    return (x_adv + delta).detach()
