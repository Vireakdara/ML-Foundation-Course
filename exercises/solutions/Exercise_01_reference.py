"""Reference solution for Exercise 01. Do not open until you have submitted your own attempt.

Run:  python Exercise_01_reference.py   (needs numpy + torch; CPU is fine, ~10 s)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Part A: NumPy softmax-CE forward/backward ----------------
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))   # shift for stability
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(z, c):
    """Mean CE over the batch and its gradient w.r.t. z: (p - y) / B."""
    B = len(c)
    p = softmax(z)
    loss = -np.log(p[np.arange(B), c]).mean()
    grad = p.copy()
    grad[np.arange(B), c] -= 1.0
    return loss, grad / B


def finite_difference(z, c, eps=1e-6):
    g = np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        g[idx] = (cross_entropy(zp, c)[0] - cross_entropy(zm, c)[0]) / (2 * eps)
    return g


# ---------------- Part C: task scaffold (identical to the exercise) ----------------
def make_task(seed=0, d=32, K=10, M=4, shots=4, n_test=200, noise=0.1):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear((M + 1) * d, 128), nn.GELU(), nn.Linear(128, d))
    for q in net.parameters():
        q.requires_grad_(False)

    def text_encoder(ctx, cls_emb):                     # ctx (M,d), cls_emb (K,d) -> (K,d)
        t = torch.cat([ctx.unsqueeze(0).expand(cls_emb.shape[0], -1, -1),
                       cls_emb.unsqueeze(1)], dim=1)     # (K, M+1, d)
        return net(t.flatten(1))

    cls_emb = torch.randn(K, d)
    V_true = torch.randn(M, d)
    V_hand = torch.randn(M, d)
    proto = F.normalize(text_encoder(V_true, cls_emb), dim=-1)

    def sample(n):
        y = torch.arange(K).repeat_interleave(n)
        return F.normalize(proto[y] + noise * torch.randn(len(y), d), dim=-1), y

    f_tr, y_tr = sample(shots)
    f_te, y_te = sample(n_test)
    return dict(net=net, enc=text_encoder, cls_emb=cls_emb, V_true=V_true,
                V_hand=V_hand, train=(f_tr, y_tr), test=(f_te, y_te))


def accuracy(task, V):
    f, y = task["test"]
    with torch.no_grad():
        w = F.normalize(task["enc"](V, task["cls_emb"]), dim=-1)
        return ((f @ w.T).argmax(1) == y).float().mean().item()


def train_coop(task, tau, steps=300, lr=2e-2):
    V = task["V_hand"].clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    f, y = task["train"]
    for _ in range(steps):
        w = F.normalize(task["enc"](V, task["cls_emb"]), dim=-1)
        loss = F.cross_entropy(f @ w.T / tau, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return V, loss.item()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Z, C = rng.normal(size=(4, 5)), rng.integers(0, 5, 4)
    L, G = cross_entropy(Z, C)
    print(f"A: loss={L:.6f}  max|analytic - FD| = {np.abs(G - finite_difference(Z, C)).max():.2e}")

    zt = torch.tensor(Z, requires_grad=True)
    F.cross_entropy(zt, torch.tensor(C)).backward()
    print(f"B: max|analytic - autograd| = {np.abs(zt.grad.numpy() - G).max():.2e}")

    task = make_task()
    w_before = task["net"][0].weight.clone()
    print(f"C: hand-prompt acc={accuracy(task, task['V_hand']):.3f}  "
          f"oracle acc={accuracy(task, task['V_true']):.3f}")
    for tau in (1.0, 0.07, 0.01):
        V, final_loss = train_coop(task, tau)
        print(f"   tau={tau:<5} final train loss={final_loss:.3f}  test acc={accuracy(task, V):.3f}")
    print("   frozen weights unchanged:", torch.equal(w_before, task["net"][0].weight),
          "| encoder .grad is None:", task["net"][0].weight.grad is None,
          "| trainable params:", V.numel())
