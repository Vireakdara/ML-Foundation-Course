# Exercise 01 — Softmax-CE Gradients and a CoOp in Miniature

> Linked lesson: [Lesson 01](../lessons/Lesson_01.md) · Back to [README](../README.md) · Record results in [PROGRESS](../PROGRESS.md)
>
> Environment: any Python with `numpy` and `torch` (your conda envs `yolo_bear`, `mdetr_env` and `yolo_world` already have torch). CPU is enough, and each part runs in seconds.
> A reference solution exists in `solutions/Exercise_01_reference.py`. **Don't open it before submitting.**

## Learning objectives
1. Implement softmax-CE and its gradient from the formula $(p-y)/B$, and **prove** your implementation is right numerically.
2. See with your own code that a frozen module passes gradients but doesn't update.
3. Observe the three temperature regimes from Lesson 01 §5.7 and Example 2 in a real training run.

## Prerequisites
Lesson 01 §5.1–5.9. Basic PyTorch (tensors, `requires_grad`, an optimiser).

---

## Part A: NumPy forward/backward and a finite-difference check 🟥

Implement, **without torch**:

```python
def softmax(z):            # z: (B, K) -> (B, K), numerically stable
    ...
def cross_entropy(z, c):   # c: (B,) int class indices -> (mean loss: float, dL/dz: (B, K))
    ...
def finite_difference(z, c, eps=1e-6):  # central differences, same shape as z
    ...
```

Use `rng = np.random.default_rng(0); Z = rng.normal(size=(4, 5)); C = rng.integers(0, 5, 4)`.

**Verify:** `max |analytic - FD| < 1e-8`. Report the loss value.

**Then test stability:** call your `softmax` on `np.array([[1000., 1001., 1002.]])`. Explain why a naive implementation returns `nan`, and why yours doesn't.

## Part B: Compare with autograd 🟥

Compute the same gradient with `torch.tensor(Z, requires_grad=True)` and `F.cross_entropy`. **Verify:** `max |analytic - autograd| < 1e-12` (float64).
Question: you used `mean` reduction. Where does the $1/B$ appear in your formula, and what would change with `reduction="sum"`?

## Part C: CoOp in miniature 🟥

Use **exactly** this task generator. It plays the role of a frozen CLIP: a fixed random "text encoder", class-name embeddings, a "true" context that the image prototypes were made with, and a wrong "hand-crafted" context.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def make_task(seed=0, d=32, K=10, M=4, shots=4, n_test=200, noise=0.1):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear((M + 1) * d, 128), nn.GELU(), nn.Linear(128, d))
    for q in net.parameters():
        q.requires_grad_(False)

    def text_encoder(ctx, cls_emb):                     # ctx (M,d), cls_emb (K,d) -> (K,d)
        t = torch.cat([ctx.unsqueeze(0).expand(cls_emb.shape[0], -1, -1),
                       cls_emb.unsqueeze(1)], dim=1)     # (K, M+1, d): [V]_1..[V]_M [CLASS]
        return net(t.flatten(1))

    cls_emb = torch.randn(K, d)
    V_true = torch.randn(M, d)
    V_hand = torch.randn(M, d)
    proto = F.normalize(text_encoder(V_true, cls_emb), dim=-1)

    def sample(n):
        y = torch.arange(K).repeat_interleave(n)
        return F.normalize(proto[y] + noise * torch.randn(len(y), d), dim=-1), y

    f_tr, y_tr = sample(shots)      # 4-shot training set (40 "image features")
    f_te, y_te = sample(n_test)     # 2000 test features
    return dict(net=net, enc=text_encoder, cls_emb=cls_emb, V_true=V_true,
                V_hand=V_hand, train=(f_tr, y_tr), test=(f_te, y_te))
```

**Tasks**
1. Write `accuracy(task, V)`: zero-shot accuracy on the test set with context `V`, classifying by cosine similarity.
   Report the accuracy for `V_hand` (the "hand-crafted prompt") and `V_true` (the oracle).
2. Write `train_coop(task, tau, steps=300, lr=2e-2)`: initialise `V` from `V_hand`, make **only `V`** trainable, and optimise CoOp's objective (Lesson 01 §5.7, CoOp Eq. 3) with Adam.
3. Before and after training, check that the encoder's first-layer weights are **bit-identical** (`torch.equal`) and that their `.grad is None`, while `V.grad` is not `None`.
4. Report the number of trainable parameters.

## Part D: Temperature sweep 🟥

Run `train_coop` with `tau ∈ {1.0, 0.07, 0.01}`. For each, report the final **train loss** and **test accuracy**.

Then explain, using Lesson 01:
- (a) Why the $\tau=1$ train loss gets stuck high. Compute the theoretical loss floor for $K=10$ with cosines in $[-1,1]$ and $\tau=1$, and explain why the observed floor is even higher.
- (b) Why $\tau=0.01$ reaches the lowest train loss but **not** the best test accuracy. Name the phenomenon, and say what it tells you about choosing hyper-parameters in few-shot papers (link to [03 §Data leakage](../03_Machine_Learning.md#data-leakage), point 2).

## Part E (optional 🟨): Class-specific context

Modify the task so each class has its own context $V^{(i)}$ (CoOp's CSC variant). Compare the trainable parameters and 4-shot test accuracy with the unified version. Predict the result *before* running it, and explain any difference with bias–variance.

---

## Expected results (reference values, seed 0)

Verified on torch 2.4.1 and 2.7.1 (CPU). Small differences on other versions or GPUs are fine, but your **qualitative pattern** must match.

| Check | Expected |
|---|---|
| A: FD error | about 1e-10 |
| B: autograd error | about 1e-17 |
| C: hand / oracle accuracy | 0.240 / 0.922 |
| C: trainable params | 128 |
| D: τ = 1.0 | train loss ≈ 2.02, test acc ≈ 0.69 |
| D: τ = 0.07 | train loss ≈ 0.36, test acc ≈ 0.81 (best) |
| D: τ = 0.01 | train loss ≈ 0.03, test acc ≈ 0.73 |

## How to verify correctness
- Parts A and B are self-verifying (FD and autograd).
- Part C: if the frozen weights changed, you passed the encoder parameters to the optimiser or forgot `requires_grad_(False)`. If `V.grad` is `None`, you broke the graph (e.g. with `.detach()`, `torch.no_grad()` or `.data`).
- Part D: if all temperatures give the same accuracy, you probably forgot to divide by `tau` or to L2-normalise.

## Common failure modes
- Passing softmax probabilities to `F.cross_entropy` (the double-softmax bug, Lesson 01 Example 3).
- Normalising along the wrong dimension (`dim=0` instead of `-1`).
- Leaving `V` as a non-leaf tensor (e.g. `V = V_hand.clone() * 1.0` after `requires_grad_`), so the optimiser holds a tensor that never receives `.grad`.
- Evaluating with gradients on (wasteful). Wrap evaluation in `torch.no_grad()`.

## What this teaches
CoOp's whole mechanism (learn a few input vectors, keep the encoder frozen, train with temperature-scaled cosine CE) fits in about 30 lines. Its success depends on things the paper treats as details: the temperature, the number of shots and the risk of overfitting. Keep this in mind when you read R10–R13.

## Submission
Put your code and printed outputs in `exercises/Exercise_01_Answers.md` (or paste them in chat), together with the written answers to D(a), D(b) and B.
