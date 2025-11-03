import itertools
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import build_students
from .teacher import Teacher
from .generator import entropy_guided_perturb
from .utils import mutual_information, entropy


Encoder = Callable[[torch.Tensor], torch.Tensor]


def _as_callable(encoder, device: torch.device) -> Encoder:
    if isinstance(encoder, torch.nn.Module):
        module = encoder.to(device)
        module.eval()

        @torch.no_grad()
        def forward(x):
            return module(x)

        return forward
    if callable(encoder):
        return encoder
    raise TypeError("encoder must be a callable or torch.nn.Module")


def _dataloader_len(loader: Iterable) -> int:
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return 0


def train_trico(
    V1,
    V2,
    train_l,
    train_u,
    val_loader,
    num_classes: int = 10,
    epochs: int = 512,
    K: int = 5,
    eps: float = 1.0,
    lr_student: float = 0.03,
    lr_teacher: float = 0.01,
    device: Optional[str] = None,
    feature_dim: int = 768,
    mi_gamma: float = 0.5,
    writer: Optional["torch.utils.tensorboard.SummaryWriter"] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, Teacher]:
    """Triadic Game-Theoretic Co-Training loop.

    Parameters
    ----------
    V1, V2:
        Frozen encoders returning embeddings for labelled/unlabelled inputs.
    train_l, train_u:
        Dataloaders producing labelled and unlabelled minibatches respectively.
    val_loader:
        Validation dataloader used for teacher meta-update.
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder1 = _as_callable(V1, device)
    encoder2 = _as_callable(V2, device)

    # Students share architecture but keep independent parameters.
    student1, student2 = build_students(feature_dim, num_classes)
    student1, student2 = student1.to(device), student2.to(device)
    teacher = Teacher().to(device)

    opt_students = torch.optim.SGD(
        itertools.chain(student1.parameters(), student2.parameters()),
        lr=lr_student,
        momentum=0.9,
    )

    # Teacher uses manually accumulated gradients scaled via validation loss
    teacher_optimizer = torch.optim.SGD(teacher.parameters(), lr=lr_teacher)

    labelled_len = _dataloader_len(train_l)
    unlabelled_len = _dataloader_len(train_u)
    loop_total = min(labelled_len, unlabelled_len)

    global_step = 0
    for epoch in range(epochs):
        student1.train()
        student2.train()
        teacher_optimizer.zero_grad(set_to_none=True)

        iterator = zip(train_l, train_u)
        if loop_total:
            iterator = tqdm(iterator, total=loop_total, desc=f"Epoch {epoch}")

        for (xl, yl), (xu, _) in iterator:
            xl = xl.to(device)
            yl = yl.to(device)
            xu = xu.to(device)

            with torch.no_grad():
                x1 = encoder1(xl)
                x2 = encoder2(xl)
                xu1 = encoder1(xu)
                xu2 = encoder2(xu)

            # Monte-Carlo dropout samples provide uncertainty estimates for MI gating.
            logits1_samples = [student1(xu1, dropout_pass=True) for _ in range(K)]
            logits2_samples = [student2(xu2, dropout_pass=True) for _ in range(K)]

            MI1 = mutual_information(logits1_samples)
            MI2 = mutual_information(logits2_samples)
            tau, lambda_u, lambda_adv = teacher()

            mask1 = MI1 > tau
            mask2 = MI2 > tau
            pseudo_y1 = logits1_samples[0].argmax(-1)
            pseudo_y2 = logits2_samples[0].argmax(-1)

            # Cross pseudo-labelling: each student trusts the other when MI is high.
            unsup_loss = torch.zeros(1, device=device)
            if mask1.any():
                unsup_loss = unsup_loss + F.cross_entropy(
                    student1(xu1[mask1]), pseudo_y2[mask1]
                )
            if mask2.any():
                unsup_loss = unsup_loss + F.cross_entropy(
                    student2(xu2[mask2]), pseudo_y1[mask2]
                )

            perturbed_xu1 = entropy_guided_perturb(xu1, student1, eps=eps, gamma=mi_gamma, mi_passes=K)
            perturbed_xu2 = entropy_guided_perturb(xu2, student2, eps=eps, gamma=mi_gamma, mi_passes=K)
            adv_loss = entropy(student1(perturbed_xu1)) + entropy(student2(perturbed_xu2))

            supervised_loss = F.cross_entropy(student1(x1), yl) + F.cross_entropy(student2(x2), yl)
            total_loss = supervised_loss + lambda_u * unsup_loss + lambda_adv * adv_loss

            opt_students.zero_grad()
            total_loss.backward()
            opt_students.step()

            global_step += 1
            if writer is not None:
                writer.add_scalar("loss/supervised", supervised_loss.item(), global_step)
                writer.add_scalar("loss/unsupervised", unsup_loss.item(), global_step)
                writer.add_scalar("loss/adversarial", adv_loss.item(), global_step)
                writer.add_scalar("loss/total", total_loss.item(), global_step)

        with torch.no_grad():
            val_loss = torch.zeros(1, device=device)
            correct = 0
            count = 0
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                xv1 = encoder1(x_val)
                xv2 = encoder2(x_val)
                logits_v1 = student1(xv1)
                logits_v2 = student2(xv2)
                val_loss += F.cross_entropy(logits_v1, y_val) + F.cross_entropy(logits_v2, y_val)
                correct += (logits_v1.argmax(dim=-1) == y_val).sum()
                correct += (logits_v2.argmax(dim=-1) == y_val).sum()
                count += 2 * y_val.size(0)
            val_loss /= max(len(val_loader), 1)
            val_accuracy = correct.float() / max(count, 1)

        # Scale teacher gradients collected during inner loop with validation loss signal.
        teacher.meta_update(val_loss)
        teacher_optimizer.step()
        teacher_optimizer.zero_grad(set_to_none=True)

        if writer is not None:
            writer.add_scalar("teacher/tau_MI", tau.item(), epoch)
            writer.add_scalar("teacher/lambda_u", lambda_u.item(), epoch)
            writer.add_scalar("teacher/lambda_adv", lambda_adv.item(), epoch)
            writer.add_scalar("val/loss", val_loss.item(), epoch)
            writer.add_scalar("val/accuracy", val_accuracy.item(), epoch)

        print(
            f"[Epoch {epoch}] tau={tau.item():.3f} lambda_u={lambda_u.item():.3f} "
            f"lambda_adv={lambda_adv.item():.3f} val_loss={val_loss.item():.4f} val_acc={val_accuracy.item():.4f}"
        )

    return student1, student2, teacher
