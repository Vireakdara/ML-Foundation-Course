# Exercise 00 — Diagnostic Assessment

> Back to [README](../README.md) · Results will be recorded in [PROGRESS](../PROGRESS.md)
>
> **Purpose:** decide where the course actually starts. It is **not** graded against you. A wrong answer here saves weeks later.
>
> **Rules**
> - Answer from memory. No looking things up, no running code (except in Part F, which says so).
> - For each question, add a **confidence tag**: `[C3]` sure · `[C2]` fairly sure · `[C1]` guessing. "I don't know" is a valid and useful answer.
> - Show working for derivations. A correct final answer without working counts as *familiarity*, not *ability*.
> - Write your answers in a new file `exercises/Exercise_00_Answers.md` (or directly in chat), keeping the question IDs.
> - Time: about 2–3 hours. You may split it across sessions; Parts A–C matter most.

Each question is labelled **[K]** (conceptual familiarity: can you explain it?) or **[A]** (application/derivation: can you do it?). The course tracks both separately.

---

## Part 0: Context (no right answers)

- **D-0a.** Rank your next 12-month priorities: (a) open-vocabulary detection, (b) VLM prompt learning / adaptation, (c) real-time detector design, (d) efficient backbones (SSM/hybrid, sparse attention), (e) PhD applications (and when), (f) other: ___.
- **D-0b.** Hours per week you can realistically commit? GPU access (model, VRAM)?
- **D-0c.** Your master's thesis topic in one sentence, and the paper(s) it was based on.

## Part A: Mathematics

- **A1 [A].** Let $W\in\mathbb R^{3\times 4}$, $x\in\mathbb R^{4}$. What is the shape of $Wx$? Of $x^\top W^\top$? In PyTorch, `nn.Linear(4,3)` applied to `X` of shape `(32, 4)` gives what shape, and what is the shape of its `.weight`?
- **A2 [A].** Given `Q` with shape `(B, h, N, d)` and `K` with shape `(B, h, M, d)`, what is the shape of `Q @ K.transpose(-2, -1)`? What does entry `[b, i, n, m]` mean?
- **A3 [A].** Compute $\cos(a,b)$ for $a=(1,2,2)$, $b=(2,0,1)$. If both vectors are first L2-normalised, what is $\|\hat a-\hat b\|_2^2$ in terms of the cosine?
- **A4 [A].** $f(w)=\sigma(w^\top x)$ with $\sigma$ the sigmoid. Derive $\nabla_w f$.
- **A5 [A].** $L=\frac12\|Xw-y\|_2^2$ with $X\in\mathbb R^{n\times d}$. Derive $\nabla_w L$ and give its shape.
- **A6 [K].** In one or two sentences: what is a Jacobian, and how does the chain rule look when every intermediate is a vector?
- **A7 [A].** A classifier outputs logits $z=(2,1,0)$ and the true class is index 0. Compute softmax $p$ (3 decimals), the cross-entropy loss, and $\partial L/\partial z$. *(You may use $e\approx2.718$, $e^2\approx7.389$.)*
- **A8 [A].** Show that maximising the likelihood of a categorical model is equivalent to minimising cross-entropy. One line of algebra is enough.
- **A9 [K].** What is KL divergence? Is it symmetric? Name one paper in your folder where it appears (any role).
- **A10 [A].** If $q,k\in\mathbb R^{d}$ have i.i.d. components with mean 0 and variance 1, what is $\mathrm{Var}(q^\top k)$? Why does this matter for attention?
- **A11 [A].** A disease has prior 1%. A test has sensitivity 90% and false-positive rate 5%. $P(\text{disease}\mid +)$? Then connect this to UniDetector's "probability calibration" in one sentence (a guess is fine).
- **A12 [K].** What is the rank of $BA$ for $B\in\mathbb R^{768\times 8}$, $A\in\mathbb R^{8\times 768}$ (at most)? How many parameters does $BA$ have, compared with a full $768\times768$ matrix?

## Part B: Machine learning

- **B1 [K].** Why do we need a validation set *and* a test set? What exactly goes wrong if you pick your checkpoint on the test set?
- **B2 [A] (critique).** Read **only §3.2 and §3.3** of [HIR (R31)](<../../Human_Interaction_Recognition_in_Surveillance_Vide.pdf>) (about 1 page). The authors split 550 videos, pooled from four cameras of 13 participants, 80/20 at random. (i) Describe a concrete way information could leak from train to test. (ii) Propose a split that tests generalisation to new people. (iii) Would you expect the 95.45% accuracy to go up or down under your split, and why?
- **B3 [K].** Explain bias vs. variance using the base-to-novel result of prompt learning (CoOp improves base classes but can hurt novel ones).
- **B4 [A].** A detector produces, for one class, detections sorted by score with TP/FP labels: `TP, TP, FP, TP, FP` and there are 4 ground-truth objects. Compute precision and recall after each detection, and the (non-interpolated) AP as the area under the step PR curve.
- **B5 [K].** Multi-class vs. multi-label classification: which output activation and loss for each? Why did YOLOv3 switch to independent logistic classifiers?
- **B6 [K].** A paper reports +0.4 mAP over a baseline from one training run. What would you need to believe the improvement is real?

## Part C: Neural networks and backpropagation

- **C1 [A].** One neuron: $z=w_1x_1+w_2x_2+b$, $a=\mathrm{ReLU}(z)$, loss $L=(a-t)^2$. With $x=(1,2)$, $w=(0.5,-0.25)$, $b=0.1$, $t=1$: compute $z, a, L$ and $\partial L/\partial w_1, \partial L/\partial w_2, \partial L/\partial b$.
- **C2 [A].** Same as C1, but with $w=(-0.5,0.1)$. What are the gradients now, and what general phenomenon does this illustrate?
- **C3 [K].** In PyTorch, what exactly happens during `loss.backward()`? Where are gradients stored, why must you call `optimizer.zero_grad()`, and what does `optimizer.step()` do with them?
- **C4 [K].** CoOp freezes CLIP's text encoder but trains the context vectors at its *input*. How can gradients reach the input if the encoder's weights are frozen? What does `requires_grad=False` actually turn off?
- **C5 [K].** Why do very deep plain networks train poorly, and how do residual connections help the *gradient*?
- **C6 [A].** Conv layer: input `(1, 64, 80, 80)`, `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)`. Output shape? Number of parameters (with bias)?
- **C7 [K].** What changes for BatchNorm between `train()` and `eval()`? What can go wrong when fine-tuning with batch size 2?
- **C8 [K].** Adam vs. SGD with momentum: what does Adam adapt? What is the difference between Adam with L2 regularisation and AdamW?

## Part D: Architectures (paper-specific prerequisites)

- **D1 [A].** Write the scaled dot-product attention formula. Name each symbol and its shape for self-attention over $N$ tokens of width $d$.
- **D2 [K].** Self-attention vs. cross-attention: what plays Q, K and V in each? In DETR / RT-DETR, where do the queries come from?
- **D3 [K].** How does CLIP perform zero-shot classification? Write the probability $p(y=i\mid x)$ and explain the role of temperature.
- **D4 [K].** Bi-encoder vs. cross-encoder: which is CLIP, which is BLIP's ITM head, and why does it matter for YOLO-World's "offline vocabulary"?
- **D5 [A].** Compute IoU for boxes A $=(0,0,4,4)$ and B $=(2,2,6,6)$ in $(x_1,y_1,x_2,y_2)$ format.
- **D6 [K].** Why do YOLO-style detectors need NMS while DETR does not? (Hint: think about how training targets are assigned.)
- **D7 [K].** In YOLOv3, $b_w = p_w e^{t_w}$. Why an exponential, and why are the centre offsets passed through a sigmoid?
- **D8 [K].** What is a state-space model in one or two sentences? (Skip if you've never looked at Mamba; that's fine.)
- **D9 [K].** LoRA: what is frozen, what is trained, and why is $B$ initialised to zero?

## Part E: Research reasoning

- **E1.** Pick one paper you presented (UniDetector, RT-DETR, ATPrompt, MP5, OW-OVD or CEC). In ≤ 150 words: what is its central insight, what is its weakest experimental point, and what would you test next?
- **E2.** MambaVision (R26) claims that putting self-attention blocks in the final stages improves long-range spatial modelling. What experiment would *falsify* that claim, and what would you hold fixed?

## Part F: Implementation (you may run code)

- **F1.** Without `torch.nn` layers or autograd, implement in NumPy a forward and backward pass for `Linear(4→3) → ReLU → Linear(3→2) → softmax-CE` on a batch of 5 random inputs. Then verify your gradients against PyTorch autograd (max abs difference < 1e-6 in float64). Paste the code and the printed differences.

---

### How I will use your answers

| Outcome on the items for a topic | Decision |
|---|---|
| [A] items correct with working, C2–C3 | Mark *Demonstrated*; skip or skim the corresponding lesson |
| [K] correct but [A] wrong or missing | Teach the lesson with emphasis on derivation and practice |
| Both wrong / C1 | Full lesson, then a re-check |

The default first lesson is [Lesson 01](../lessons/Lesson_01.md). If A4–A8 and C1–C4 come back solid, we will compress it to the exercises only and move to L02/L05/L06.
