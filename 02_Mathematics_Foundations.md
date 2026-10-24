# 02 — Mathematics Foundations

> Back to [README](README.md) · Roadmap: [01](01_Learning_Roadmap.md) · Status: [PROGRESS](PROGRESS.md)
>
> This file is the **map** of the mathematics the course needs. It is prioritised by what *your* papers use, not by a textbook order. Full treatments live in `lessons/`. Each topic lists its priority, the papers that need it, the lesson that teaches it, and a one-line self-check. If you can't answer the self-check confidently *with a derivation*, the topic is not done.
>
> **Priority:** **E** = essential for many papers · **H** = helpful · **A** = advanced / later.

## Notation used throughout the course

| Symbol | Meaning |
|---|---|
| $x\in\mathbb R^d$ | column vector (lower-case bold in print, plain here) |
| $W\in\mathbb R^{m\times n}$ | matrix, $m$ rows (outputs) × $n$ columns (inputs); $y=Wx$ |
| $X\in\mathbb R^{B\times N\times d}$ | tensor: batch $B$, tokens $N$, channels $d$ (PyTorch row-major convention: **rows are samples/tokens**, so a linear layer is $Y = XW^\top + b$) |
| $z$ | logits (pre-softmax scores) |
| $p=\mathrm{softmax}(z)$ | predicted distribution |
| $y$ | one-hot target (or index $c$) |
| $\nabla_\theta L$ | gradient of scalar loss $L$ w.r.t. parameters $\theta$, same shape as $\theta$ |
| $\frac{\partial f}{\partial x}$ | Jacobian, shape $(\text{dim } f)\times(\text{dim } x)$ |
| $\tau$ | temperature |
| $\langle a,b\rangle=a^\top b$ | inner product, $\cos(a,b)=\frac{a^\top b}{\|a\|\|b\|}$ |

## 1. Linear algebra

| Topic | Pri. | Needed by (R#) | Lesson | Self-check |
|---|---|---|---|---|
| Vectors, matrices, tensors, **shapes and broadcasting** | E | all | L01, L07 | Shape of `Q @ K.transpose(-2,-1)` for `Q,K: (B,h,N,d)`? |
| Matrix multiply as (i) a linear map, (ii) a batch of dot products | E | 5, 7, 10, 27 | L07 | Why is attention's score matrix "all pairwise dot products"? |
| Dot product, cosine similarity, L2 normalisation | E | 6, 10–14, 27 | **L01**, L08 | After L2-normalising, how are cosine and Euclidean distance related? ($\|a-b\|^2=2-2\cos$) |
| Norms ($\ell_1,\ell_2$, Frobenius), distances | E | 4 (k-means), 17 (IoU losses), regularisation | L03, L06 | Why does $\ell_2$ weight decay shrink weights proportionally? |
| Linear transformations, rank, column space | E | LoRA, 18 (re-param), 29 | L05, L11 | What is the rank of $BA$ for $B\in\mathbb R^{d\times r},A\in\mathbb R^{r\times k}$? |
| Low-rank factorisation, **SVD** | H | 2 (truncated SVD), LoRA | L11 | How many parameters does a rank-$r$ approximation of a $d\times k$ matrix save? |
| Eigen-decomposition, diagonalisation | H→A | 29 (S4/Mamba use diagonal $A$) | L19 | Why does a diagonal state matrix make $\exp(\Delta A)$ cheap? |
| Convolution as structured (Toeplitz) matrix multiply | E | 2, 4, 17–22 | L05 | Write a 1-D conv with kernel size 3 as a matrix. |
| Linear-map merging (conv ∘ BN, parallel branches) | E | 18, 27 (RepVL-PAN), 21 | L14 | Fold $\mathrm{BN}(\mathrm{conv}(x))$ into a single conv: what are the new $W', b'$? |

## 2. Calculus

| Topic | Pri. | Needed by | Lesson | Self-check |
|---|---|---|---|---|
| Derivatives, partial derivatives, gradients | E | all | **L01** | Gradient of $f(w)=\|Xw-y\|^2$? |
| **Chain rule** (scalar → vector → Jacobian form) | E | all | **L01**, L02 | Express $\partial L/\partial x$ when $L=\ell(g(f(x)))$ with vector intermediates. |
| Computational graphs, reverse-mode autodiff | E | all | L02 | Why is reverse mode cheaper than forward mode for $\mathbb R^n\to\mathbb R$? |
| Gradients of softmax, CE, sigmoid, BCE | E | 25+ papers | **L01** | Show $\partial L_{CE}/\partial z = p-y$. |
| Gradients through max / ReLU (sub-gradients) | E | 2 (RoI max-pool), all CNNs | L02 | Gradient of max-pool w.r.t. a non-maximal input? |
| Gradients of smooth-L1, IoU-family losses | E | 2, 4, 17, 21 | L06 | Where is smooth-L1 non-differentiable, and why doesn't that matter? |
| Jacobian-vector products; why `backward()` needs a scalar | H | L02 | L02 | What does `y.backward(v)` compute for non-scalar `y`? |
| Hessian, curvature, conditioning | H | optimisation intuition | L03 | Why does a badly conditioned Hessian slow down GD? |
| Linear ODEs and **discretisation** (Euler, zero-order hold) | E (Stage 6) | 29–34 | L19 | Discretise $h'=ah+bx$ with ZOH, step $\Delta$. |
| Taylor expansion | H | 3 (intuition), 37 | L03 | Second-order Taylor of $L(\theta+\delta)$. |

## 3. Probability and statistics

| Topic | Pri. | Needed by | Lesson | Self-check |
|---|---|---|---|---|
| Random variables, distributions (Bernoulli, categorical, Gaussian) | E | all | L01 | Which distribution does softmax output parameterise? |
| Conditional probability, **Bayes' theorem** | E | 1, 26 (calibration), 39 | L01, L18 | Correct a posterior $p(c\mid x)$ trained under prior $\pi$ for a new prior $\pi'$. |
| Expectation, variance, variance of sums/products | E | 5 ($1/\sqrt{d_k}$), init schemes | L07 | If $q_i,k_i\sim\mathcal N(0,1)$ i.i.d., what is $\mathrm{Var}(q^\top k)$? |
| **Maximum likelihood** ⇔ minimising CE / MSE | E | all losses | **L01**, L04 | Derive MSE from Gaussian likelihood and CE from categorical likelihood. |
| Sampling, Monte-Carlo estimates, mini-batch gradients as estimators | E | L03 | L03 | Why is the mini-batch gradient unbiased? What is its variance in terms of $B$? |
| Confidence intervals, seeds, significance of small gains | E (research) | 12, 17, 21, all SOTA tables | L04 | Is +0.3 AP from one seed evidence? What would you need? |
| Calibration (confidence vs. accuracy), temperature scaling | H | 26, CEC (your AL presentation) | L18 | Define expected calibration error (ECE). |

## 4. Optimisation

| Topic | Pri. | Needed by | Lesson | Self-check |
|---|---|---|---|---|
| Objectives, empirical risk, regularised objectives | E | all | L03, L04 | Write ERM with weight decay. |
| GD, SGD, mini-batch; learning rate | E | all | L03 | What happens with LR > $2/\lambda_{\max}$ on a quadratic? |
| Momentum, Nesterov | E | YOLO training (SGD+momentum) | L03 | Momentum as an exponential moving average of gradients. |
| Adam, **AdamW** (decoupled weight decay) | E | 5, 10, 20, 27 | L03 | Why is $\ell_2$-in-loss ≠ decoupled weight decay under Adam? |
| LR schedules: warm-up, cosine, step | E | 5 (warm-up), all | L03 | Why does the Transformer need warm-up? |
| Saddle points, local minima, sharp vs. flat minima | H | intuition | L03 | — |
| Constrained / min-max optimisation (GANs) | H | 11 (CGAN) | L12 | Write the CGAN objective and say who minimises what. |
| Combinatorial assignment (**Hungarian algorithm**) | E (Stage 4) | DETR, 20, 21 | L16 | Solve a 3×3 cost matrix by hand. |
| Differentiable relaxation of discrete choices (softmax over options) | H | 12 (attribute search) | L12 | How does DARTS make architecture choice differentiable? |
| MuSGD / Muon-style optimisers | A | 25 | L17 | — |

## 5. Information theory

| Topic | Pri. | Needed by | Lesson | Self-check |
|---|---|---|---|---|
| Entropy $H(p)$ | E | 26, CEC/active learning, 19 | L01 | Entropy of a uniform distribution over $K$ classes? |
| **Cross-entropy** $H(p,q)$ and its link to log-likelihood | E | all classifiers | **L01** | $H(p,q)=H(p)+D_{KL}(p\|q)$: why does minimising CE w.r.t. $q$ equal minimising KL? |
| KL divergence (asymmetry) | E | 36 (KD), 11 | L01, Stage 7 | Why is KD's KL computed with temperature-softened distributions? |
| Mutual information, data-processing inequality | E (Stage 4) | 19 (information bottleneck) | L15 | State the DPI for $X\to Z_1\to Z_2$. |
| InfoNCE as a lower bound on MI | H | CLIP, 15, 27 | L08 | Why do more negatives tighten the bound? |

## 6. Dependency order inside mathematics

```
shapes/matmul ─┬─> dot/cosine ──> softmax+CE (L01) ──> attention (L07)
               │                         │
chain rule ────┴─> Jacobians ──> backprop (L02) ──> optimisers (L03)
probability ─> MLE ─> CE/MSE (L01) ─> calibration/Bayes (L18)
linear maps ─> rank/SVD ─> LoRA, re-param (L11, L14)
linear ODEs ─> discretisation ─> SSMs (L19)
entropy ─> KL ─> MI ─> information bottleneck (L15)
```
