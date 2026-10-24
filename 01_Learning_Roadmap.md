# 01 — Learning Roadmap

> Back to [README](README.md) · Paper details: [00_Paper_Inventory](00_Paper_Inventory.md) · Status: [PROGRESS](PROGRESS.md)
>
> **Version 1 (2026-10-24).** Built from the inventory *before* your diagnostic. Expect revisions once [the diagnostic](exercises/Exercise_00_Diagnostic.md) is answered. Changes are logged at the bottom of this file and in [PROGRESS](PROGRESS.md).

## 1. What the collection is actually about

The 40 papers are **not** a generic ML reading list. They form roughly five connected sub-trees around one centre of gravity: **open-vocabulary / real-time object detection with vision-language models**.

```
                Transformer (R5)
               /     |      \         \
   multi-label     CLIP*     DETR*      SSM/Mamba (R29)
   heads (R7–9)    |    \       \           |
                prompt   VL pre-   RT-DETR → YOLOv10 → YOLOv12
                learning training  (R20)    (R21)     (R22)
                (R10–14) (R15–16)     \       /
                     \        \      YOLO lineage (R4,17–19,23–25)
                      \        \       /
                       open-vocabulary detection
                       UniDetector → YOLO-World → LLMDet
                            (R26)      (R27)      (R28)
                                         \
                                     Mamba-YOLO-World (R34) ← SSM track
   * = essential paper that is NOT in the folder (see inventory §2)
```

Three consequences shape the course:

1. **Two concepts carry most of the weight:** (a) *softmax-over-similarities trained with cross-entropy* (CLIP zero-shot, CoOp, ATPrompt, YOLO-World's contrastive head, attention itself) and (b) *attention / learnable queries* (Transformer, Query2Label, DETR, RT-DETR, BLIP, YOLOv12). Both are taught early and rigorously.
2. **Detection-specific machinery** (IoU, anchors, label assignment, NMS, AP) is needed by about 17 papers. You have used it, but the course checks the math.
3. **LoRA appears in only one paper** (MP5, peripheral). The parameter-efficient method that dominates your collection is **prompt learning** (CoOp family), so the course gives prompt learning the depth and covers LoRA compactly as part of the adaptation spectrum ([05_Advanced_Topics](05_Advanced_Topics.md), §4).

## 2. Stages at a glance

| Stage | Theme | Lessons (planned) | Papers (R#) | Exit criterion |
|---|---|---|---|---|
| **0** | Foundations rebuilt (diagnostic-driven) | L01–L05 | (R38 as a critique exercise) | Derive softmax-CE and MLP gradients by hand. Explain AdamW and a sound train/val/test protocol. Compute conv output shapes. |
| **1** | Detection fundamentals | L06 | R1, R2, R3 (selected), R4 | Compute IoU, decode YOLO boxes, run NMS and compute AP by hand. Derive the Fast R-CNN multi-task loss. |
| **2** | Attention, embeddings, query heads | L07–L09 | R5, R6, R7, R8, R9 | Implement multi-head attention from scratch with correct shapes. Explain bi- vs. cross-encoders. Explain query-based pooling. |
| **3** | Vision-language models and prompt learning | L10–L13 | *CLIP, ViT* (external), R10–R16 | Derive CoOp's gradient path. Explain base-to-novel evaluation. Compare ITC/ITM/LM objectives. |
| **4** | Real-time detection: YOLO and DETR | L14–L17 | R17, R18, R19, *DETR* (external), R20, R21, R22, R23–R25 (reference) | Explain label assignment (one-to-many vs. one-to-one), Hungarian matching, re-parameterisation, and why NMS-free matters. |
| **5** | Open-vocabulary detection | L18 | R26, R27, R28 | Explain region–text alignment end to end, from text prompt to box, and critique the OVD evaluation protocols. |
| **6** | State-space models and efficient token mixing | L19–L21 | R29–R34, R35 (optional) | Discretise an SSM, explain selectivity, and reproduce MambaOut's argument. |
| **7** | Specialised topics (on demand) | as needed | R36 (KD), R37 (3-D), R39–R40 (agents/RL) | Driven by your research direction. |
| **8** | Research skills (runs throughout, intensifies from Stage 4) | — | critiques, a reproduction project | A reproduction report plus one well-scoped research proposal. |

**Parallelism.** Stages 3 and 4 share only Stage 2 as a prerequisite, so they can be **interleaved** (e.g. alternate a VLM lesson with a YOLO lesson). Stage 5 needs both. Stage 6 needs only Stage 2 plus L03, and can move earlier if GSARP (diagnostic D-0) is your active project.

## 3. Prerequisite (dependency) map

### 3.1 Concept → paper graph

```mermaid
graph TD
  subgraph Foundations
    LA[Linear algebra: shapes, matmul, norms] --> SM[Softmax + CE + temperature]
    CR[Chain rule / Jacobians] --> SM
    CR --> BP[Backprop + autograd]
    SM --> BP
    BP --> OPT[SGD / Adam / schedules]
    PROB[Probability, MLE, Bayes] --> SM
    PROB --> CAL[Prior correction / calibration]
    ML[Splits, leakage, metrics] --> EVAL[Detection & retrieval metrics]
    BP --> CNN[CNN, BN, residuals]
  end
  CNN --> DET[Boxes, IoU, anchors, NMS, AP]
  DET --> FRCNN[R2 Fast R-CNN]
  DET --> Y3[R4 YOLOv3]
  SM --> ATT[Attention]
  LA --> ATT
  ATT --> TF[R5 Transformer]
  TF --> EMB[Embeddings & contrastive losses]
  EMB --> SBERT[R6 SBERT]
  TF --> Q2L[R7 Query2Label / R8 C-Tran / R9 ML-Decoder]
  TF --> VIT[ViT*]
  EMB --> CLIP[CLIP*]
  VIT --> CLIP
  CLIP --> COOP[R10 CoOp]
  COOP --> SPG[R11 SPG]
  COOP --> ATP[R12 ATPrompt]
  COOP --> TPC[R13 TP-CLIP]
  COOP --> CRC[R14 CLIP-RC]
  CLIP --> BLIP[R15 BLIP] --> BEIT[R16 BEiT-3]
  Y3 --> Y4[R17 YOLOv4] --> Y7[R18 YOLOv7] --> Y9[R19 YOLOv9]
  TF --> DETR[DETR*]
  Q2L --> DETR
  DETR --> RTD[R20 RT-DETR] --> Y10[R21 YOLOv10]
  Y7 --> Y10
  Y10 --> Y12[R22 YOLOv12]
  FRCNN --> UNI[R26 UniDetector]
  CLIP --> UNI
  CAL --> UNI
  Y7 -->|YOLOv8 anatomy via R23| YW[R27 YOLO-World]
  CLIP --> YW
  SBERT --> YW
  RTD --> LLMD[R28 LLMDet]
  YW --> LLMD
  BLIP --> LLMD
  TF --> MAMBA[R29 Mamba]
  MAMBA --> VIM[R30 Vim] --> VMB[R31 VMamba] --> MV[R32 MambaVision]
  VMB --> MO[R33 MambaOut]
  MAMBA --> MYW[R34 Mamba-YOLO-World]
  YW --> MYW
  TF --> NSA[R35 NSA]
```

`*` marks a paper **not in the folder** (CLIP, ViT, DETR). If Mermaid doesn't render in your viewer, use the per-paper tables in §4, which carry the same information.

### 3.2 Shared prerequisites (most-reused concepts)

| Concept | Approx. # papers that need it | Taught in |
|---|---|---|
| Softmax / cross-entropy / temperature / cosine-similarity logits | 25+ | **L01** |
| Backprop through frozen modules (gradient flows *through* but doesn't *update*) | 10 (all prompt-learning, YOLO-World text encoder, CLIP-RC) | **L01** (intro), L02, L11 |
| Scaled dot-product and multi-head attention | about 30 | L07 |
| IoU, box parameterisation, NMS, AP | about 17 | L06 |
| Label assignment (one-to-many, one-to-one/Hungarian) | 8 | L14, L16 |
| Contrastive learning (InfoNCE, ITC, region–text) | 9 | L08, L10 |
| Few-shot / base-to-novel / domain-generalization protocols | 6 | L04, L12 |
| Linear recurrences and SSM discretisation | 6 | L19 |

## 4. Recommended reading order

Legend for "Read": **Full** = read everything. **Core** = abstract, intro, method, main tables, limitations; skim related work and appendix. **Selected** = only the named sections. **Reference** = look things up as needed.

### Stage 0: Foundations (no papers required)

The lessons come first because every paper below assumes them. Optional critique exercise: **R38 HIR**, which is short and readable, and lets you practise spotting a train/test protocol problem (see [03 §Data leakage](03_Machine_Learning.md#data-leakage)).

### Stage 1: Detection fundamentals

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| 1 | Object Detection (Amit et al.) | classifier basics | L04 | Full (1 h) | state what a detector outputs | contrast generative vs. discriminative detection |
| 2 | **Fast R-CNN** | softmax-CE, chain rule, max-pool gradient, CNN fine-tuning | L01, L02, L05, L06 | Full | derive $\partial L/\partial z$ for softmax-CE and the max-pool sub-gradient | re-derive Eq. 4 (backprop through RoI pooling), explain smooth-L1 vs. L2, implement RoIPool on a toy tensor |
| 3 | Zou et al. survey | IoU, AP | L06 | Selected (§ milestones, metrics, NMS) | compute AP from a PR curve | place every later detector on the survey's timeline |
| 4 | YOLOv3 | sigmoid/exp box decoding, BCE, anchors | L06 | Full | decode $(t_x,t_y,t_w,t_h)$ to a box by hand | explain why logistic class scores (multi-label) replaced softmax |

**Why Fast R-CNN is the first real paper.** It is short, foundational for two of your sub-trees (detection lineage and UniDetector's two-stage design), and it contains an **explicit backprop derivation** (through RoI pooling) and an **explicit multi-task loss**. Reading it immediately tests whether L01–L02 actually stuck. The Transformer is more central to your collection, but its key equation is opaque without fluency in softmax, matmul shapes and gradients, so it comes second.

### Stage 2: Attention, embeddings and query heads

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| 5 | **Attention Is All You Need** | matmul shapes, softmax, residual, LayerNorm | L01, L07 | Full | write the shapes of $Q,K,V$ for batch $B$, length $N$, heads $h$ | derive why $1/\sqrt{d_k}$, implement MHA with a causal mask, state the $O(N^2d)$ cost |
| 6 | Sentence-BERT | cosine similarity, triplet/contrastive loss | L08 | Core | explain why a cross-encoder can't be cached | explain the bi-encoder design behind CLIP and YOLO-World's offline vocabulary |
| 7 | Query2Label | cross-attention, BCE, multi-label mAP | L09 | Full | explain cross-attention as weighted pooling | sketch the model and loss from memory; compare it with GAP |
| 8 | C-Tran | R5, masked training | L09 | Core | explain BERT-style masking | explain label-mask training and its partial-label use |
| 9 | ML-Decoder | R7, attention complexity | L09 | Core | count FLOPs of a decoder head for $K$ classes | explain group decoding and why it scales |

### Stage 3: Vision-language models and prompt learning

**Add two external papers first:** CLIP (Radford et al. 2021) and ViT (Dosovitskiy et al. 2021). They are prerequisites for nearly everything in this stage.

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| — | *ViT* (external) | R5 | L10 | Core | — | turn a $224^2$ image into a token sequence with exact shapes |
| — | *CLIP* (external) | InfoNCE, dual encoders | L08, L10 | Core (§2, §3.1) | write the symmetric contrastive loss | build a zero-shot classifier from text embeddings |
| 10 | **CoOp** | CLIP zero-shot, backprop through frozen modules | L01, L11 | Full | explain which tensors have `requires_grad=True` | implement CoOp's prompt learner; explain the unified vs. class-specific trade-off |
| 11 | SPG | R10, GAN/CGAN, DG protocol | L12 | Core | write the CGAN min-max objective | explain stage I vs. stage II and what the generator outputs |
| 12 | ATPrompt | R10, differentiable search | L12 | Full | explain the base-to-novel harmonic mean | reproduce Eq. 4, explain the attribute search, critique its gain sizes |
| 13 | TP-CLIP | R10, visual prompts, video basics | L12 | Core | explain VPT | explain how temporal information enters without changing CLIP |
| 14 | CLIP-RC | R10, ViT patch tokens, segmentation, mIoU | L13 | Core | explain why CLIP is weak at localisation | explain the region bridge and the recovery loss as anti-forgetting |
| 15 | BLIP | contrastive loss, cross-attention, causal LM | L13 | Core | write ITC, ITM and LM losses | explain CapFilt as data bootstrapping |
| 16 | BEiT-3 | R5, masked modelling, R15 | L13 | Selected (§1–2, Table 1) | explain masked image modelling | explain Multiway Transformers |

### Stage 4: Real-time detection, YOLO and DETR

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| 17 | YOLOv4 | IoU-family losses, BN, augmentation | L14 | Core + ablation tables | derive CIoU's terms | judge which "freebies" the ablations actually support |
| 18 | YOLOv7 | re-parameterisation, label assignment | L14 | Core | fold a BN into a conv by hand | explain planned re-param and lead/aux assignment |
| 19 | YOLOv9 | mutual information, auxiliary supervision | L15 | Core | state the data-processing inequality | explain PGI and assess the rigour of its information-theory argument |
| — | *DETR* (external) | R5, Hungarian matching | L16 | Core | solve a 3×3 assignment by hand | explain set prediction and why DETR needs no NMS |
| 20 | **RT-DETR** | DETR, multi-scale features, deformable attention | L16 | Full | explain why the encoder dominates DETR latency | explain AIFI+CCFF and uncertainty-minimal query selection |
| 21 | **YOLOv10** | R20 one-to-one idea, TAL, NMS | L17 | Full | explain why one-to-many needs NMS | explain consistent dual assignments and the matching metric |
| 22 | YOLOv12 | attention cost, FlashAttention idea | L17 | Core | compute attention FLOPs/memory for a $80^2$ map | explain area attention's cost reduction |
| 23 | YOLOv8→11 review | R4, R17 | L14 | Reference | — | map each Ultralytics block to its function |
| 24 | YOLOv11 overview | R23 | — | Reference (skim) | — | — |
| 25 | YOLO26 analysis | DFL, NMS-free, quantization | L17 | Selected | — | list what YOLO26 removed and why, and flag unverified claims |

### Stage 5: Open-vocabulary detection

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| 26 | **UniDetector** | two-stage detection, CLIP, Bayes prior correction, LVIS metrics | L06, L18 | Full | correct a classifier's posterior for a new prior | explain heterogeneous label-space training, decoupling, and calibration |
| 27 | **YOLO-World** | YOLOv8 anatomy, CLIP text encoder, contrastive head | L14, L18 | Full | explain region–text contrastive loss | trace a text prompt through RepVL-PAN to a box score, and explain re-parameterisation of the vocabulary |
| 28 | LLMDet | DETR/Grounding DINO, LM loss | L16, L18 | Core | write an autoregressive LM loss | explain why caption supervision helps detection, and critique the data confound |

### Stage 6: State-space models and efficient token mixing

| R# | Paper | Must know first (E) | Prepared by | Read | Ready when you can… | After reading, you should be able to… |
|---|---|---|---|---|---|---|
| 29 | **Mamba** | linear ODE, discretisation, RNN gating, convolution ↔ recurrence | L19 | Core (§1–3, §4.1) | discretise $h'=Ah+Bx$ with ZOH for a scalar | explain selectivity and why it breaks the convolution mode |
| 30 | Vim | R29, ViT | L20 | Core | — | explain bidirectional scanning and position embeddings |
| 31 | VMamba | R29, hierarchical backbones | L20 | Core | — | explain SS2D's four scan routes |
| 32 | MambaVision | R29–31, R5 | L20 | Core | — | explain the hybrid design and its ablation |
| 33 | MambaOut | R29–31 | L20 | Full | define causal vs. non-causal token mixing | reproduce the argument and judge whether the experiments test it |
| 34 | Mamba-YOLO-World | R27 + R29 | L20 | Core | — | explain guided selective scan as multimodal fusion |
| 35 | NSA (optional) | R5, KV-cache, GPU memory | L21 | Selected | — | explain compression / selection / window branches |

### Stage 7: Specialised topics (on demand)

| R# | Paper | Must know first (E) | Notes |
|---|---|---|---|
| 36 | Pixel Distillation | KD (soft targets, KL, temperature), detection heads | Pair with Hinton et al. 2015 (external). |
| 37 | GPSFormer | PointNet/PointNet++ (external), kNN graphs | Only if 3-D vision becomes relevant. |
| 38 | HIR | splits/leakage, classical classifiers | Already used as a Stage 0 critique exercise. |
| 39 | Auto MC-Reward | RL basics (MDP, PPO, reward shaping) | Needs a short RL primer lesson first. |
| 40 | MP5 | MLLMs, LoRA | Pairs with the LoRA section of 05 and the LoRA paper (external). |

## 5. How each paper is studied

Every paper follows the nine-step workflow (requirements → diagnose → teach gaps → readiness check → strategic read → reconstruct → implement → critique → record) defined in [06_Paper_Reading_Guide](06_Paper_Reading_Guide.md). The "Ready when…" column above is the readiness check. The "After reading…" column is the minimum comprehension deliverable.

## 6. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-24 | v1 created | Initial inventory of 40 papers. Diagnostic pending. |
