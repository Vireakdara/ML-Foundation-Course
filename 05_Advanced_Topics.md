# 05 — Advanced Topics

> Back to [README](README.md) · Roadmap: [01](01_Learning_Roadmap.md) · Foundations: [02](02_Mathematics_Foundations.md), [03](03_Machine_Learning.md), [04](04_Deep_Learning.md)
>
> Only topics that your **actual** papers require are included, in proportion to how much they use them. Each section lists what will be taught, the key equations to master, and where it appears. Detailed lessons are written when you reach the stage.

## 1. Transformers and attention (Stage 2 · L07)

**Papers:** R5 directly. Then R9–R13, R17–R19, R25, R26, R28 build on it.

- Motivation: RNNs are sequential and have long paths between distant tokens. Attention is a *differentiable, content-addressed lookup*.
- Scaled dot-product attention: $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$, with $Q\in\mathbb R^{N_q\times d_k},K\in\mathbb R^{N_k\times d_k},V\in\mathbb R^{N_k\times d_v}$ and mask $M$.
- Why $\sqrt{d_k}$: if the components are i.i.d. with unit variance, $\mathrm{Var}(q^\top k)=d_k$, so unscaled logits saturate the softmax and its gradient vanishes.
- Multi-head: $h$ parallel subspaces, $d_k=d_{model}/h$, then concatenate and project.
- Positional information: sinusoidal (R5), learned (ViT), relative/none (YOLOv12 removes it).
- Block: pre-/post-LN, FFN ($d\to4d\to d$), residuals. Encoder, decoder, causal mask, cross-attention.
- Cost: $O(N^2 d)$ time, $O(N^2)$ memory for scores. This is the root motivation for R19 (area attention), R26–R27 (SSM-based), R28 (sparse attention) and R17 (scale-restricted attention).
- Vision Transformers: patchify ($P\times P$ patches → linear embedding), [CLS] token, DeiT distillation token. **Read ViT (external).**

## 2. Representation learning, contrastive learning and CLIP (Stage 3 · L08, L09)

**Papers:** CLIP (external) is the foundation for R7–R11, R23–R25, R27. R6 (SBERT) and R12 (BLIP ITC) cover the same ideas from other angles.

- Embedding spaces, cosine similarity, L2 normalisation, and the temperature $\tau$ (CLIP learns $\log(1/\tau)$).
- **InfoNCE / CLIP loss** for a batch of $B$ pairs with normalised $u_i$ (image) and $w_j$ (text):
  $$\mathcal L=\frac1{2B}\sum_i\Big[-\log\frac{e^{u_i^\top w_i/\tau}}{\sum_j e^{u_i^\top w_j/\tau}}-\log\frac{e^{u_i^\top w_i/\tau}}{\sum_j e^{u_j^\top w_i/\tau}}\Big]$$
  This is cross-entropy where the "classes" are the other items in the batch, which is exactly why [Lesson 01](lessons/Lesson_01.md) comes first.
- Zero-shot classification: text embeddings of prompts *are* the classifier weights. CoOp Eq. 1/3 (verified in the PDF).
- Bi-encoder (CLIP, SBERT: cacheable) vs. cross-encoder / fusion encoder (BLIP ITM, BEiT-3: more expressive, not cacheable).
- Region-level alignment: CLIP is image-level, while detection and segmentation need region-level alignment. That gap is what R11, R23, R24 and R25 each solve differently.
- **Original CLIP vs. later work:** CLIP = contrastive pre-training plus hand-crafted prompts. Everything in your folder is *adaptation* or *extension* of CLIP, not CLIP itself.

## 3. Computer vision and object detection (Stages 1, 4, 5 · L06, L13–L17)

**Papers:** R1–R4, R14–R25.

- Box coordinate systems ($xyxy$, $cxcywh$, normalised), IoU, GIoU, DIoU, CIoU. Box parameterisation (R-CNN deltas, YOLO sigmoid/exp, anchor-free distance-to-sides, **DFL** distributions).
- Two-stage (proposals + RoI head: R2, R23) vs. one-stage dense (R4, R14–R19) vs. set prediction (DETR, R17, R25).
- **Label assignment**: IoU-threshold, SimOTA, TAL (task-aligned), one-to-one/Hungarian. R18's consistent dual assignment.
- NMS (greedy, soft-NMS) and why NMS-free inference matters (R17, R18, R22).
- AP/mAP computation (COCO protocol), LVIS frequency splits.
- Real-time design: CSP, ELAN/E-ELAN/GELAN/R-ELAN, SPP(F), PAN, re-parameterisation, decoupled heads, EMA, Mosaic.
- Open-vocabulary detection: region–text contrastive head (R24), grounding formulation (GLIP, Grounding DINO: external), heterogeneous label spaces and calibration (R23), caption supervision (R25), open-world unknown detection (OW-OVD presentation).
- **Multi-object tracking:** no paper in the folder is about MOT, so it is *not* scheduled. If it becomes a goal, add it as Stage 7 (detection → association → ByteTrack/SORT, Kalman filter).

## 4. Adaptation: pretraining, fine-tuning, prompts, LoRA (Stage 3 · L10)

**Papers:** prompt learning dominates (R7–R11). LoRA is used only in R33 (MP5's Percipient MLLM). KD appears in R29.

The **adaptation spectrum**, taught in one lesson with one shared notation (frozen $\theta_0$, trainable $\phi$):

| Method | What is trained ($\phi$) | Params trained | Where it appears |
|---|---|---|---|
| Zero-shot | nothing | 0 | CLIP baselines in R7–R11 |
| Linear probe | a linear head on frozen features | $d\cdot K$ | CoOp comparisons |
| Full fine-tuning | everything | all | YOLO-World fine-tuning, LLMDet |
| Partial / layer freezing | the last blocks | some | many |
| **Textual prompt tuning** | input context vectors $[V]_{1..M}$ | $M\cdot d$ (e.g. 16×512) | **R7, R8, R9** |
| Deep prompts | per-layer prompt tokens | $L\cdot M\cdot d$ | R9 deep version (MaPLe/PromptSRC style) |
| Visual prompts | tokens on the image side | small | R10 |
| Adapters | small bottleneck MLPs | small | (CLIP-Adapter, external) |
| **LoRA** | low-rank $\Delta W = BA$ | $r(d+k)$ per matrix | R33 |
| Knowledge distillation | a student trained against a teacher | student size | R29, DeiT, BLIP (momentum distillation) |

**LoRA in brief** (full derivation in L10). The frozen weight is $W_0\in\mathbb R^{d\times k}$, and the update is $W=W_0+\frac{\alpha}{r}BA$ with $B\in\mathbb R^{d\times r}$ initialised to 0 and $A\in\mathbb R^{r\times k}$ initialised randomly, $r\ll\min(d,k)$. Only $A,B$ get gradients: $\nabla_B L=\frac{\alpha}{r}\,(\nabla_W L)\,A^\top$, $\nabla_A L=\frac{\alpha}{r}\,B^\top(\nabla_W L)$. Because $B=0$ at the start, the model initially equals the pretrained one. After training, $BA$ can be merged into $W_0$, so there is no inference cost. Trade-offs: the rank limits the expressivity of the update, and the choice of $r$, $\alpha$ and which matrices to adapt matters.

**Quantization** (FP16/INT8 export): needed only for R22 (YOLO26 edge deployment). Covered briefly in L16.

## 5. State-space models (Stage 6 · L18–L19)

**Papers:** R26, R27 (with Mamba, external, as the reference). Continuous SSM $h'(t)=Ah(t)+Bx(t),\ y=Ch(t)$. ZOH discretisation $\bar A=\exp(\Delta A)$, $\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\Delta B$. Recurrence $h_t=\bar Ah_{t-1}+\bar Bx_t$ ⇔ convolution with kernel $\bar K=(C\bar B, C\bar A\bar B,\dots)$ when the parameters are time-invariant. **Selectivity** ($\Delta,B,C$ depend on $x_t$) breaks time-invariance, so Mamba replaces the convolution with a hardware-aware parallel scan. For vision, the open questions are scan order (images have no natural 1-D order) and causality (a causal scan sees only earlier patches). MambaVision answers with a non-causal mixer plus attention blocks, and Mamba-YOLO-World with text-guided scans.

## 6. Efficient attention at scale (Stage 6, optional · L20)

**Papers:** R28 (NSA), with R19 (YOLOv12 area attention + FlashAttention) as the vision counterpart. Topics: KV-cache, MQA/GQA, memory-bound vs. compute-bound kernels (arithmetic intensity), blockwise sparsity, and why "theoretical FLOPs saved" ≠ wall-clock speedup.

## 7. Topics deliberately *not* scheduled

| Topic | Why not (yet) |
|---|---|
| Multi-object tracking | No MOT paper in the folder. Add on request. |
| Diffusion models | Not used by any paper. |
| Full RL theory | Needed only for R32 (peripheral). A short primer is planned only if you choose that track. |
| Graph neural networks in depth | Only R30 (ADGConv) uses them, and it is peripheral. |
