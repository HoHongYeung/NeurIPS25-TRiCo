<p align="center">
  <img src="https://arxiv.org/abs/2509.21526"/>
  <img src="https://img.shields.io/badge/Semi--Supervised%20Learning-TRiCo-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Coming%20Soon-FFA500?style=for-the-badge&logo=github"/>
</p>

<h1 align="center" style="font-size:40px; font-weight:bold;">
🌌 <span style="color:#3C9EE7;">TRiCo</span>: <span style="color:#8A2BE2;">Triadic Game-Theoretic Co-Training</span> for Robust Semi-Supervised Learning
</h1>

<p align="center">
  <b>Official PyTorch implementation</b> of our NeurIPS 2025 paper:<br>
  <i>"TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning"</i><br>
  <a href="https://openreview.net/forum?id=6732_TRiCo_Triadic_Game_Theore">📄 Paper (OpenReview)</a> |
  <a href="#">🧠 Project Page (Coming Soon)</a> |
  <a href="#">🧪 Demo (In Preparation)</a>
</p>

---

## 🚀 Overview

**TRiCo** redefines semi-supervised learning as a *three-player Stackelberg game* between:
- 🧑‍🎓 Two *student classifiers* trained on complementary frozen encoders (e.g., DINOv2, MAE, CLIP, MoCo-v3, etc.),
- 🧑‍🏫 A *meta-learned teacher* that dynamically adjusts pseudo-label thresholds (τ<sub>MI</sub>) and loss weights (λ<sub>u</sub>, λ<sub>adv</sub>),
- 🎲 A *non-parametric generator* that perturbs embeddings to expose decision boundary weaknesses.

Together, these components form a stable triadic optimization process, achieving superior robustness and generalization across CIFAR-10, SVHN, STL-10, and ImageNet subsets.

---

## 🧩 Repository Structure

