# 04 — Deep Learning Foundations

> Back to [README](README.md) · Roadmap: [01](01_Learning_Roadmap.md) · Math: [02](02_Mathematics_Foundations.md) · ML: [03](03_Machine_Learning.md)
>
> This is the internals of neural networks, from a single neuron to the CNN/Transformer building blocks your papers combine. The **goal** is that you can (1) compute a forward pass by hand, (2) compute its gradients by hand, (3) predict what PyTorch's `backward()` will produce, and (4) explain why each architectural component exists.

## 1. Topic map

| Block | Topic | Pri. | Papers that need it | Lesson |
|---|---|---|---|---|
| **Basic networks** | neuron, affine map, activation, depth, shapes, forward pass, training vs. inference | E | all | L01, L02 |
| **Backprop** | chain rule on graphs; per-parameter gradients; vectorised backprop for an MLP; autograd; `requires_grad`, `no_grad`, `detach` | E | all, especially frozen-encoder methods (7–11, 24) | L01 (intro), **L02** |
| | vanishing/exploding gradients; gradient flow through residuals | E | 16 (PGI), deep nets | L02, L05 |
| **Losses** | CE, BCE, focal, asymmetric, smooth-L1, IoU-family, DFL, contrastive (InfoNCE), triplet, LM loss, KD | E | see the papers | L01, L06, L08 |
| **Activations** | ReLU, LeakyReLU, SiLU/Swish, Mish, GELU, sigmoid, softmax | E | YOLOv4 (Mish), YOLOv5+ (SiLU), Transformers (GELU) | L02 |
| **Init & normalisation** | Xavier/Kaiming init; BatchNorm (train vs. eval stats), LayerNorm, CmBN | E | 5, 14, all | L02, L05 |
| **Regularisation** | dropout, DropBlock, weight decay, augmentation (Mosaic, MixUp), label smoothing | E | 5, 14, 15 | L03, L13 |
| **Training loop** | data loading, forward, loss, backward, step, zero_grad, eval mode, checkpoints | E | all | L02, L03 |
| **Optimisers** | SGD+momentum, AdamW, warm-up/cosine, EMA of weights | E | YOLO family (SGD + EMA), Transformers/VLMs (AdamW) | L03 |
| **Practicalities** | GPU memory (activations vs. params), mixed precision (FP16/BF16, loss scaling), gradient accumulation, debugging (overfit one batch) | E | all | L03 |
| **CNNs** | convolution, kernels, stride, padding, dilation, channels, depthwise/pointwise conv, pooling, receptive field | E | 2, 4, 14–19, 23–24 | **L05** |
| | residual networks; CSP/ELAN-style aggregation; SPP/SPPF | E | 4, 14–16, 20 | L05, L13 |
| | multi-scale features: FPN, PAN; detection heads (coupled/decoupled, anchor-based/free) | E | 4, 14–19, 24, 27 | L06, L13 |
| **Sequence models** | RNN, gating (LSTM/GRU intuition), why parallelism matters | E (for 5, 26, 27) | 5, 26, 27 | L07, L18 |
| | attention as soft lookup; self vs. cross attention; masking | E | ~20 papers | L07 |
| **Embeddings** | token embeddings, positional encodings, learned queries, label embeddings | E | 5, 7, 9, 24 | L07 |

## 2. The minimum you must be able to do by hand (exit tests for Stage 0)

1. **Forward pass:** given $x\in\mathbb R^2$, $W_1\in\mathbb R^{3\times2}$, ReLU, $W_2\in\mathbb R^{2\times3}$ and softmax, compute $p$ with a pen.
2. **Backward pass:** for the same network with CE loss, compute $\nabla_{W_2}L$, $\nabla_{W_1}L$, $\nabla_x L$ and check them with `torch.autograd`.
3. **Shape audit:** for a conv layer with $C_{in}=64$, $C_{out}=128$, $k=3$, $s=2$, $p=1$ on a $64\times80\times80$ input, give the output shape, parameter count and MACs.
4. **BN semantics:** explain what changes between `model.train()` and `model.eval()` for BatchNorm and dropout, and why fine-tuning with tiny batches can break BN.
5. **Frozen modules:** for CoOp, list which tensors are leaves with `requires_grad=True`, which receive `.grad`, and which only *propagate* gradients.

## 3. Component "why" table (fill in as you progress)

For every component, the course asks: *what problem did it solve, what does it assume, when does it fail?*

| Component | Problem it solves | Assumes / costs | Fails when | Status |
|---|---|---|---|---|
| Residual connection | degradation of deep plain nets; gives an identity path for gradients | shapes must match (or a projection) | — | Not started |
| BatchNorm | internal covariate shift (original claim) / smoother loss landscape (later view); allows higher LR | reasonably large i.i.d. batches | tiny batches, distribution shift between train and test | Not started |
| LayerNorm | per-token normalisation, batch-independent | — | — | Not started |
| Attention | content-based, long-range mixing | $O(N^2)$ memory/compute | very long sequences / high-res maps (→ 19, 26–28) | Not started |
| Positional encoding | attention is permutation-equivariant | — | — | Not started |
| FPN / PAN | objects at multiple scales | extra compute | — | Not started |
| NMS | duplicate detections from one-to-many assignment | IoU threshold choice | crowded scenes; adds latency (→ 17, 18) | Not started |

## 4. PyTorch ↔ mathematics bridge

| You write | Mathematically | Covered in |
|---|---|---|
| `nn.Linear(n, m)` | $y = xW^\top + b$, $W\in\mathbb R^{m\times n}$ | L01 |
| `F.cross_entropy(z, c)` | $-\log\mathrm{softmax}(z)_c$ (computed stably via log-sum-exp) | **L01** |
| `F.normalize(x, dim=-1)` | $x/\|x\|_2$ | L01 |
| `loss.backward()` | reverse-mode accumulation of $\partial L/\partial\theta$ into `.grad` for leaves with `requires_grad` | L02 |
| `p.requires_grad_(False)` | $\theta$ treated as a constant; gradients still pass *through* ops that use it | L01, L10 |
| `torch.no_grad()` | no graph is recorded, so nothing can be back-propagated | L02 |
| `x.detach()` | stop-gradient: $\mathrm{sg}(x)$ | L02 |
| `optimizer.step()` | $\theta \leftarrow \theta - \eta\,\mathrm{update}(\nabla\theta)$ | L03 |
| `model.eval()` | switches BN to running stats, disables dropout. **Does not** disable autograd | L02 |
