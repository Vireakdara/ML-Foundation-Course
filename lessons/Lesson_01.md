# Lesson 01 — Softmax, Cross-Entropy and Temperature: the Gradient Engine Behind Classifiers, CLIP, CoOp and Attention

> Stage 0 · Prereq for: Fast R-CNN (R2), Transformer (R5), CoOp (R7) and its descendants (R8–R11), YOLO-World (R24), plus every classifier in the folder.
> Back to [README](../README.md) · Roadmap: [01](../01_Learning_Roadmap.md) · Exercise: [Exercise 01](../exercises/Exercise_01.md) · Status: [PROGRESS](../PROGRESS.md)
>
> **Labels:** 🟥 Essential · 🟨 Helpful · 🟦 Advanced (optional on first pass)

## 1. Learning objectives

By the end you should be able to, **without notes**:

1. Compute softmax and cross-entropy for a small logit vector by hand, in a numerically stable way.
2. **Derive** $\dfrac{\partial L}{\partial z}=p-y$ from the softmax Jacobian.
3. Explain cross-entropy as negative log-likelihood and as $H(y)+D_{KL}(y\|p)$.
4. Explain what the temperature $\tau$ does to (a) the distribution and (b) the gradient, and why cosine-similarity logits *need* a temperature.
5. Write CLIP's zero-shot classifier and CoOp's objective, and trace the gradient from the loss to CoOp's context vectors **through a frozen encoder**.
6. Recognise the same function inside attention ($1/\sqrt{d_k}$ is a fixed temperature).

## 2. Why this topic first

Count the papers in your folder that end in "softmax over scores, trained with cross-entropy":

- **Classification heads:** Fast R-CNN ($K{+}1$-way softmax), Transformer (vocabulary softmax), ViT/DeiT-style backbones such as MambaVision.
- **CLIP-style similarity classifiers:** CoOp Eq. 3, ATPrompt Eq. 1, SPG, TP-CLIP, CLIP-RC, UniDetector's region classifier, YOLO-World's text-contrastive head.
- **Contrastive objectives:** CLIP/InfoNCE, BLIP's ITC. These are softmax-CE where the "classes" are the other items in the batch.
- **Attention itself:** every row of $\mathrm{softmax}(QK^\top/\sqrt{d_k})$.

That is more than 20 of the 33 papers in scope. If $p-y$, temperature and gradient-through-frozen-modules are not fully under control, every one of those papers will rest on a guess. This lesson is therefore a **genuine prerequisite**, not a warm-up.

## 3. Prerequisites

- 🟥 Exponentials and logarithms; derivative of $e^x$ and $\ln x$.
- 🟥 Partial derivatives; the scalar chain rule.
- 🟥 Vectors, dot products, norms.
- 🟨 Basic probability (a categorical distribution).

## 4. Intuition

A network produces **scores** (logits) $z_1,\dots,z_K$, one per class. They can be any real numbers. We want a **probability distribution** that:

1. is positive and sums to 1,
2. preserves the ordering of the scores,
3. is smooth, so we can differentiate it.

Softmax does this by exponentiating (which makes everything positive and stretches differences) and normalising. Training then asks: *how much probability did you put on the right answer?* Cross-entropy penalises $-\log$ of that probability, gently near 1 and without bound near 0.

The gradient turns out to be **"what you predicted minus what was true"**, $p-y$. Each logit is pushed down in proportion to the probability it wrongly took and the correct one is pushed up by its deficit. Almost all learning in your papers flows through this one expression.

## 5. Mathematical foundations

### 5.1 Softmax 🟥

For logits $z\in\mathbb R^K$:
$$p_i=\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}},\qquad i=1..K.$$

| Symbol | Meaning | Shape |
|---|---|---|
| $z$ | logits (unnormalised scores) | $K$ (per sample); `(B, K)` in a batch |
| $p$ | predicted class distribution | same as $z$ |
| $K$ | number of classes | scalar |

**Shift invariance:** $\mathrm{softmax}(z+c\mathbf 1)=\mathrm{softmax}(z)$, because $e^{c}$ cancels. Two consequences:
- **Numerical stability:** compute with $z-\max(z)$ so the largest exponent is $e^0=1$ (no overflow).
- Only **differences** between logits matter. Keep this in mind for temperature.

### 5.2 Cross-entropy 🟥

With true class $c$ (one-hot $y$, $y_c=1$):
$$L=-\sum_{i}y_i\log p_i=-\log p_c=-z_c+\log\sum_j e^{z_j}.$$
The last form (**log-sum-exp**) is what `F.cross_entropy` computes internally from **raw logits**. It never forms $p$ explicitly, which is more stable.

**Three equivalent readings:**
1. *Likelihood:* the model says $P(Y=c\mid x)=p_c$. The likelihood of a dataset is $\prod_n p^{(n)}_{c_n}$, and maximising it ⇔ minimising $\sum_n -\log p^{(n)}_{c_n}$, which is the mean CE. **Cross-entropy is negative log-likelihood.**
2. *Information theory:* $H(y,p)=H(y)+D_{KL}(y\|p)$. Because $H(y)$ doesn't depend on the model, minimising CE ⇔ minimising KL from the target to the prediction.
3. *Margin-like:* $L=\log\big(1+\sum_{j\ne c}e^{z_j-z_c}\big)$. The loss only falls when the correct logit **beats the others by a margin**.

### 5.3 The softmax Jacobian 🟥

$$\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)\qquad\text{i.e.}\qquad J=\mathrm{diag}(p)-pp^\top.$$

*Derivation (do this once by hand).* For $i=j$, the quotient rule gives $\frac{e^{z_i}S-e^{z_i}e^{z_i}}{S^2}=p_i-p_i^2$. For $i\neq j$, the numerator doesn't depend on $z_j$, so the result is $-\frac{e^{z_i}e^{z_j}}{S^2}=-p_ip_j$.

Properties: $J$ is symmetric, and every row sums to 0. Raising all logits equally changes nothing, which is shift invariance again.

### 5.4 The key result: $\partial L/\partial z = p-y$ 🟥

By the chain rule, summing over the paths through each $p_i$:
$$\frac{\partial L}{\partial z_j}=\sum_i\frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}=\sum_i\Big(-\frac{y_i}{p_i}\Big)p_i(\delta_{ij}-p_j)=-y_j+p_j\underbrace{\sum_i y_i}_{=1}=p_j-y_j.$$

In vector form, $\nabla_z L = p - y$. In a batch with mean reduction, each row is divided by $B$.

**Why this matters:** the gradient is bounded ($|p_j-y_j|\le1$), it never vanishes while the prediction is wrong, and it is cheap to compute. It is also why frameworks fuse softmax and CE: the fused backward pass is just one subtraction.

### 5.5 Sigmoid + BCE: the multi-label cousin 🟥

For **independent** binary decisions (multi-label: YOLOv3 classes, per-class region scores in dense detectors):
$$q=\sigma(z)=\frac1{1+e^{-z}},\qquad L=-[y\log q+(1-y)\log(1-q)],\qquad \frac{\partial L}{\partial z}=q-y.$$
It has the same "prediction minus target" form, but each class is separate and **the classes don't compete**. Choosing softmax vs. sigmoid is a modelling decision: *are the labels mutually exclusive?*

### 5.6 Temperature 🟥

$$p_i=\frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}.$$
- $\tau\to0$: distribution → one-hot on the arg-max ("sharp").
- $\tau\to\infty$: distribution → uniform.
- Gradient w.r.t. the **raw scores** $s$: $\dfrac{\partial L}{\partial s}=\dfrac{1}{\tau}(p-y)$. A smaller $\tau$ gives larger gradients on the scores, until $p$ saturates.

### 5.7 Cosine-similarity classifiers (CLIP, CoOp, YOLO-World) 🟥

CLIP replaces a learned weight matrix with **text embeddings**. Let $f$ be the image feature and $w_i=g(t_i)$ the text feature of the prompt for class $i$. Then (CoOp Eq. 1, ATPrompt Eq. 1; verified in the PDFs):
$$p(y=i\mid x)=\frac{\exp(\cos(w_i,f)/\tau)}{\sum_{j=1}^K\exp(\cos(w_j,f)/\tau)}.$$
This is **exactly** §5.1 with logits $z_i=\cos(w_i,f)/\tau$. A linear classifier $z_i=w_i^\top f$ whose weights are *produced by a text encoder* and normalised.

**Why the temperature is not optional.** Cosines lie in $[-1,1]$. With $\tau=1$, the best case is a correct cosine of $+1$ against all others at $-1$, giving $p_c=\dfrac{e^{1}}{e^{1}+(K-1)e^{-1}}$:
- $K=3$: $p_c\le0.787$, so the loss can never go below $0.240$.
- $K=1000$: $p_c\le0.0073$, so the loss never goes below $4.91$. The model *cannot* express confidence.

Dividing by a small $\tau$ (CLIP learns it, ending near $0.01$, i.e. logit scale 100) widens the logit range so the softmax can become confident. YOLO-World's head $s_{k,j}=\alpha\,\hat e_k^\top\hat w_j+\beta$ does the same job with a learnable scale $\alpha$ (and shift $\beta$).

### 5.8 Gradient of cosine similarity 🟨

With $\hat u=u/\|u\|$ and $s=\hat u^\top\hat w$:
$$\frac{\partial s}{\partial u}=\frac{1}{\|u\|}\big(I-\hat u\hat u^\top\big)\hat w.$$
The gradient is **orthogonal to $u$**: cosine training can only rotate features, not lengthen them. The factor $1/\|u\|$ means large-norm features learn more slowly.

### 5.9 Gradients through a frozen module (the CoOp mechanism) 🟥

CoOp builds prompt $t_i=[V]_1\dots[V]_M[\text{CLASS}_i]$ (Eq. 2), feeds it to the **frozen** text encoder $g$, and trains only $V=\{[V]_m\}$ (shared across classes in the unified version). The computation graph for one image is:

$$V\ \xrightarrow{\ \text{concat}\ }\ t_i\ \xrightarrow{\ g\ (\text{frozen})\ }\ w_i\ \xrightarrow{\ \cos(\cdot,f)/\tau\ }\ z_i\ \xrightarrow{\ \text{softmax-CE}\ }\ L$$

$$\frac{\partial L}{\partial V}=\sum_{i=1}^{K}\underbrace{(p_i-y_i)}_{\S5.4}\cdot\frac{1}{\tau}\cdot\underbrace{\frac{\partial \cos(w_i,f)}{\partial w_i}}_{\S5.8}\cdot\underbrace{\frac{\partial g(t_i)}{\partial t_i}}_{\text{Jacobian of frozen }g}\cdot\frac{\partial t_i}{\partial V}.$$

Three facts follow:
1. **"Frozen" means the weights $\theta_g$ are not updated** (`requires_grad=False`, so no `.grad` is accumulated for them). It does **not** block gradient flow: the Jacobian $\partial g/\partial t$ is still computed, using the frozen weights' values, during `backward()`.
2. Because $V$ is shared, **every class's prompt contributes** to $\partial L/\partial V$, not just the true class. Wrong classes push $V$ in the direction that lowers their similarity.
3. The image encoder is frozen too, and $f$ needs no gradient at all. So CoOp never back-propagates into the image tower, which saves memory.

### 5.10 Connection to attention 🟨

For a query $q$ and keys $k_1..k_N$: $a=\mathrm{softmax}(q^\top k_j/\sqrt{d_k})_j$. This is §5.6 with **fixed** temperature $\tau=\sqrt{d_k}$. The reason for that choice (Lesson 07): if $q,k$ have unit-variance components, $q^\top k$ has variance $d_k$, so without scaling the softmax saturates and its Jacobian $\mathrm{diag}(a)-aa^\top\to0$, which starves learning. That is the §5.3 Jacobian reappearing.

## 6. Worked examples

### Example 1: Softmax-CE by hand
$z=(1,0,-1)$, true class $c=0$.
- $e^{z}=(2.718,\ 1.000,\ 0.368)$, sum $=4.086$.
- $p=(0.665,\ 0.245,\ 0.090)$.
- $L=-\ln 0.665=0.408$.
- $\nabla_z L=p-y=(-0.335,\ 0.245,\ 0.090)$. Check: the components sum to 0 ✓.

A gradient-descent step with learning rate 1 gives $z\leftarrow z-\nabla_zL=(1.335,\,-0.245,\,-1.090)$. The correct logit rises and the others fall in proportion to the probability they "stole".

### Example 2: Temperature on CLIP-like cosines
Cosines $s=(0.30, 0.25, 0.20)$, correct class 0. (Real CLIP cosines are often this close together.)

| $\tau$ | $p$ | $L$ | $\partial L/\partial s=(p-y)/\tau$ |
|---|---|---|---|
| 1 | (0.350, 0.333, 0.317) | 1.049 | (−0.650, 0.333, 0.317) |
| 0.07 (CLIP init) | (0.578, 0.283, 0.139) | 0.548 | (−6.02, 4.04, 1.98) |
| 0.01 (≈ trained CLIP) | (0.9933, 0.0067, 0.00005) | 0.0068 | (−0.674, 0.669, 0.0045) |

Read the table carefully:
- At $\tau=1$ the model is almost uniform even though class 0 has the highest cosine.
- At $\tau=0.01$ the *same* cosines give 99.3% confidence.
- The gradient at $\tau=0.01$ is concentrated on the **hardest negative** (class 1). The easy negative (class 2) gets almost nothing. **Low temperature = hard-negative focus.** This is why contrastive learning's temperature is treated as a key hyper-parameter.

### Example 3: Why the double-softmax bug is silent
If you apply `softmax` and then call `F.cross_entropy` (which applies log-softmax again), the model still trains, but on $\mathrm{softmax}(p)$ where $p\in[0,1]$. The effective logits are squashed into $[0,1]$, the same ceiling problem as §5.7 with $\tau=1$. Loss plateaus high and accuracy can look "okay", so it often goes unnoticed. *Always pass raw logits to `F.cross_entropy`.*

## 7. Connection to machine learning and deep learning

- **MLE:** every CE-trained network is a conditional likelihood model $p_\theta(y\mid x)$. MSE is the Gaussian analogue (Lesson 04).
- **Calibration:** softmax outputs are *not* automatically calibrated probabilities. Temperature scaling after training is the classic fix. UniDetector's prior calibration (R23) and your CEC presentation's entropy calibration both work on this layer.
- **Label smoothing** (Transformer §5.4, YOLOv4) replaces $y$ with $(1-\epsilon)y+\epsilon/K$, so the gradient becomes $p-y_{\text{smooth}}$ and logits stop growing without bound.
- **Entropy** $H(p)=-\sum p_i\log p_i$ (uncertainty) is the quantity active-learning methods such as CEC rank samples by.

## 8. Connection to your papers

| Paper | Where this lesson appears | What to check when you read it |
|---|---|---|
| R2 Fast R-CNN | $L_{cls}=-\log p_u$ over $K{+}1$ classes (incl. background) | Why a background class? How is it sampled (25% fg)? |
| R5 Transformer | attention softmax; output softmax; label smoothing $\epsilon=0.1$ | Where is $\sqrt{d_k}$, and what would happen without it? |
| R7 CoOp | Eq. 3; gradient through the frozen $g$ into $[V]_m$ | Which parameters are trainable? Count them (16×512). |
| R9 ATPrompt | Eq. 1 identical; attribute tokens inside $t_i$ | Which tokens are learnable vs. fixed? |
| R24 YOLO-World | $s_{k,j}=\alpha\,\hat e_k^\top\hat w_j+\beta$ | Why do they say L2-norm + affine are "important for stabilizing"? (Answer: §5.7.) |
| R4 YOLOv3 | switches to independent logistic (BCE) classes | §5.5: labels like "Woman" and "Person" overlap. |

## 9. Implementation (short demo; full assignment in [Exercise 01](../exercises/Exercise_01.md))

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)

z = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)   # logits, shape (B=1, K=3)
c = torch.tensor([0])                                       # true class index

loss = F.cross_entropy(z, c)          # log-softmax + NLL, stable
loss.backward()

p = torch.softmax(z.detach(), dim=-1)
y = F.one_hot(c, 3).float()
print(loss.item())                    # 0.4076
print(z.grad)                         # tensor([[-0.3348, 0.2447, 0.0900]])
print(torch.allclose(z.grad, p - y))  # True  <- the p - y result
```

Gradient flow through a frozen module, in miniature:

```python
enc = torch.nn.Linear(4, 4)                 # stand-in for CLIP's text encoder
for prm in enc.parameters():
    prm.requires_grad_(False)               # "frozen"

V = torch.zeros(1, 4, requires_grad=True)   # learnable context (CoOp-style)
out = enc(V).sum()
out.backward()
print(V.grad is not None)        # True  : gradient flowed THROUGH the frozen encoder
print(enc.weight.grad)           # None  : frozen weights accumulate no gradient
```

## 10. Common misconceptions

| Misconception | Correction |
|---|---|
| "Frozen layers block backprop." | They only stop **their own** updates. Gradients pass through them to anything upstream that requires grad (§5.9). |
| "Softmax output = model confidence." | It's a normalised score, often over-confident. Calibration is a separate problem. |
| "Temperature only matters at inference." | It scales the training gradient by $1/\tau$ and changes *which* negatives get the gradient (Example 2). |
| "Softmax is fine for multi-label." | Softmax forces classes to compete (sum to 1). Use sigmoid + BCE when labels co-occur. |
| "Cosine logits are fine as is." | Bounded logits cap confidence and leave a high loss floor (§5.7). You need a scale or temperature. |
| "I should apply softmax before `F.cross_entropy`." | No: it expects raw logits (Example 3). |
| "$\partial L/\partial z$ only affects the true class." | Every logit gets a gradient $p_j$ (or $p_c-1$). Wrong classes are actively pushed down. |

## 11. Exercises (pen and paper)

1. 🟥 Derive $\partial p_i/\partial z_j$ from scratch, then $\nabla_zL=p-y$. No peeking at §5.3–5.4.
2. 🟥 For $z=(3,1,0.2)$ and $c=2$, compute $p$, $L$, $\nabla_zL$. Which logit receives the largest-magnitude gradient, and why?
3. 🟥 Show that $\mathrm{softmax}(z/\tau)$ approaches one-hot as $\tau\to0^+$ when the arg-max is unique. What happens when there is a tie?
4. 🟥 Derive $\partial L/\partial z$ for sigmoid + BCE.
5. 🟨 Derive $\partial s/\partial u$ for $s=\cos(u,w)$ (§5.8) and verify that $u^\top\frac{\partial s}{\partial u}=0$.
6. 🟥 Write the CoOp gradient $\partial L/\partial V$ for the **class-specific context** version (each class $i$ has its own $V^{(i)}$). Which classes contribute to $\partial L/\partial V^{(i)}$ for one training image of class $c$?
7. 🟦 With label smoothing $\epsilon$, what is the optimal logit gap $z_c-z_j$ (for $j\ne c$) at the minimum of the loss? Why does this stop logits from diverging?

## 12. Comprehension questions (answer in words)

1. Explain to a first-year student why the CE gradient is "prediction minus truth".
2. Why does CLIP *learn* $\tau$ instead of fixing $\tau=1$?
3. In CoOp, the image encoder is also frozen. Does the backward pass need to go into the image tower? What does that save?
4. Why does low temperature focus the gradient on hard negatives? Use Example 2.
5. How is $1/\sqrt{d_k}$ in attention related to the temperature in CLIP?

## 13. Summary

- Softmax maps logits to a distribution, and only logit differences matter. Compute it stably.
- CE = negative log-likelihood = KL to the target (up to a constant). Its gradient w.r.t. logits is $p-y$.
- Sigmoid + BCE is the independent (multi-label) version, with gradient $q-y$.
- Temperature controls sharpness and scales gradients. Cosine logits **require** a temperature or scale.
- CLIP zero-shot = softmax-CE with text-generated weights. CoOp trains only the input context and gets its gradient through a frozen encoder.
- Attention reuses the same softmax with temperature $\sqrt{d_k}$.

## 14. Criteria for moving forward

You can move to Lesson 02 when **all** of these hold (I check them, not you):
- [ ] Exercise 1 (derivation) is correct without help.
- [ ] Exercise 2 is numerically correct.
- [ ] Exercise 6 correctly identifies which classes contribute.
- [ ] [Exercise 01](../exercises/Exercise_01.md) Parts A–C pass their numerical checks.
- [ ] You can answer comprehension questions 1, 3 and 4 in your own words.

If your diagnostic already shows these, we do only Exercise 01 Part C and move on.

## 15. Suggested next lesson

**Lesson 02: Backpropagation by hand.** From a single neuron to a two-layer MLP in vectorised form, computational graphs, and what PyTorch's `backward()` does (leaves, `.grad` accumulation, `no_grad`, `detach`). Then Fast R-CNN's backprop through RoI max-pooling as the first real-paper check.
