# 00 — Research Paper Inventory

> **Scope.** Every PDF found under `E:\Papers` (40 files), inventoried on 2026-10-24.
> Back to [README](README.md) · Reading order and dependencies: [01_Learning_Roadmap](01_Learning_Roadmap.md)

## How to read this file

- **Inspection level** (how much of the paper I actually read):
  - **P1**: title, authors, abstract, introduction, and full section outline.
  - **P2**: P1 plus selected method passages, equations, or experimental sections (named in the card).
  - **Full**: the entire paper. *No paper is at Full yet.* Each paper moves to Full when we study it together, and the card gets corrected then.
- **Venue.** Where the venue is printed in the PDF, it is given as is. A **†** means the venue comes from my general knowledge of the literature, not from the PDF text. Check those before citing.
- **Difficulty** (for *you*, assuming the foundations are rebuilt): 1 = primer, 5 = research-frontier / heavy math or systems.
- **R#** = position in the recommended reading order (see [01_Learning_Roadmap](01_Learning_Roadmap.md#4-recommended-reading-order)).
- **Stated vs. interpreted.** *Problem*, *Contribution* and *Method* lines restate what the paper says. *Prerequisites*, *Difficulty*, *Position* and anything marked *(my reading)* are my interpretation.

## 1. File-level findings

| Item | Finding |
|---|---|
| Location | `E:\Papers` (this is the "Papers" folder itself), with subfolders `YOLO/`, `GSARP/MAMBA & VISION MAMBA PAPERS/`, `GSARP/MULTI-LABEL CLASSIFICATION PAPERS/`, `cvpr2024Minecraft/`, `presentation/` |
| Formats | 40 × PDF (all text-extractable with `pdftotext`, no scans), 6 × PPTX (your own presentation slides), 1 × `~$ATPrompt…pptx` (a 165-byte PowerPoint lock file, not a real document; safe to delete when PowerPoint is closed) |
| Duplicates | **None.** Two pairs overlap in topic but are distinct documents: (a) *Object detection Mark.pdf* (Amit, Felzenszwalb & Girshick, encyclopedia entry) vs. *Object_detection_in_20_years_A_survey Mark.pdf* (Zou et al., survey); (b) *YOLOv11.pdf* (Khanam & Hussain overview) vs. *YOLOv8 to YOLO11…pdf* (Hidayatullah et al. review). |
| Supplementary material | Your presentations in `presentation/` for ATPrompt, UniDetector, RT-DETR and MP5 (papers present), plus **OW-OVD** and **Active Learning for VLMs** (the **papers for these two are not in the folder**). No code or notes files. |
| Unreadable files | None. |

**What the presentations tell me about you** (authored "Ly Vireak Dara"): you presented UniDetector (Oct 2023), MP5 (Apr 2024), RT-DETR (Jul 2024), ATPrompt (Mar 2025), Active Learning for VLMs / CEC (May 2025) and OW-OVD (Oct 2025). The `GSARP` folder (files dated Jan 2026) groups Mamba and multi-label classification papers, which looks like a recent or current project. **Diagnostic question D-0 asks you what GSARP is**, because that changes how the Mamba and multi-label tracks are prioritised.

## 2. Master table

| R# | Paper (short) | Year | Venue | Area | Diff. | Insp. |
|---|---|---|---|---|---|---|
| 1 | [Object Detection (encyclopedia entry)](<../Object detection Mark.pdf>) | 2020 | Springer *Computer Vision: A Reference Guide* | Detection: concepts | 1 | P1 |
| 2 | [Fast R-CNN](<../Fast R-CNN.pdf>) | 2015 | ICCV† | Detection: two-stage | 2 | P1 |
| 3 | [Object Detection in 20 Years: A Survey](<../Object_detection_in_20_years_A_survey Mark.pdf>) | 2023 | Proceedings of the IEEE | Detection: survey | 2 | P1 |
| 4 | [YOLOv3](<../YOLO/YOLOv3 An Incremental Improvement.pdf>) | 2018 | arXiv tech report | Detection: one-stage | 2 | P1 (+§2.1 box eqs) |
| 5 | [Attention Is All You Need](<../Attention Is All You Need.pdf>) | 2017 | NIPS 2017 | Architecture: Transformer | 3 | P1 |
| 6 | [Sentence-BERT](<../Sentence-BERT.pdf>) | 2019 | EMNLP-IJCNLP† | NLP: embeddings | 2 | P1 |
| 7 | [Query2Label](<../GSARP/MULTI-LABEL CLASSIFICATION PAPERS/Query2Label - A Simple Transformer Way to Multi-Label Classification.pdf>) | 2021 | arXiv | Multi-label classification | 2 | P1 |
| 8 | [C-Tran](<../GSARP/MULTI-LABEL CLASSIFICATION PAPERS/General Multi-label Image Classification with Transformers.pdf>) | 2021 | CVPR† | Multi-label classification | 3 | P1 |
| 9 | [ML-Decoder](<../GSARP/MULTI-LABEL CLASSIFICATION PAPERS/ML-Decoder - Scalable and Versatile Classification Head.pdf>) | 2023 | WACV† | Multi-label / zero-shot cls. | 3 | P1 |
| 10 | [CoOp: Learning to Prompt for VLMs](<../Learning to Prompt for Vision-Language Models.pdf>) | 2022 | IJCV† | VLM prompt learning | 2 | P2 (§3 method) |
| 11 | [Soft Prompt Generation (SPG)](<../Soft Prompt Generation for Domain Generalization.pdf>) | 2024 | ECCV† | VLM prompts + domain generalization | 3 | P1 (+CGAN passages) |
| 12 | [ATPrompt](<../ATPrompt-Textual Prompt Learning with Embedded Attributes.pdf>) | 2024/25 | arXiv 2412.09442; ICCV 2025† | VLM prompt learning | 3 | P2 (§3.1–3.2) |
| 13 | [TP-CLIP: Temporal Prompting](<../Is Temporal Prompting All We Need For Limited Labeled Action Recognition.pdf>) | 2025 | arXiv 2504.01890 | Video VLM adaptation | 3 | P1 |
| 14 | [CLIP-RC: Regional Clues in CLIP](<../Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation.pdf>) | 2024 | CVPR (2024†) | Zero-shot segmentation | 4 | P1 |
| 15 | [BLIP](<../Bootstrapped Language-Image Pre-training (BLIP).pdf>) | 2022 | ICML† | VL pre-training | 3 | P1 |
| 16 | [BEiT-3](<../BEITv3.pdf>) | 2022 | CVPR 2023† | Multimodal foundation model | 4 | P1 |
| 17 | [YOLOv4](<../YOLO/YOLOv4 Optimal Speed and Accuracy of Object Detection.pdf>) | 2020 | arXiv | Real-time detection | 3 | P1 |
| 18 | [YOLOv7](<../YOLO/YOLOv7 Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.pdf>) | 2022 | CVPR 2023† | Real-time detection | 3 | P1 |
| 19 | [YOLOv9 (PGI, GELAN)](<../YOLO/YOLOv9.pdf>) | 2024 | ECCV† | Real-time detection | 4 | P1 |
| 20 | [RT-DETR: DETRs Beat YOLOs](<../DETRs Beat YOLOs on Real-time Object Detection.pdf>) | 2023/24 | CVPR 2024† | End-to-end detection | 4 | P1 |
| 21 | [YOLOv10](<../YOLO/yolov10.pdf>) | 2024 | NeurIPS 2024† | NMS-free detection | 3 | P2 (§3.1 dual assignment) |
| 22 | [YOLOv12](<../YOLO/YOLOv12-Attention-Centric Real-Time Object Detectors.pdf>) | 2025 | arXiv tech report; NeurIPS 2025† | Attention-based real-time detection | 4 | P1 |
| 23 | [YOLOv8 → YOLO11 review](<../YOLO/YOLOv8 to YOLO11-A Comprehensive Architecture In-depth Comparative Review.pdf>) | 2025 | arXiv | Architecture review (secondary) | 2 | P1 |
| 24 | [YOLOv11 overview](<../YOLO/YOLOv11.pdf>) | 2024 | arXiv | Architecture review (secondary) | 1 | P1 |
| 25 | [YOLO26 analysis](<../YOLO/YOLO26-KEY ARCHITECTURAL ENHANCEMENTS AND.pdf>) | 2025/26 | arXiv 2509.25164 | Edge detection review (secondary) | 2 | P1 |
| 26 | [UniDetector](<../Detecting Everything in the Open World-Towards Universal Object Detection.pdf>) | 2023 | CVPR† | Open-world / open-vocab detection | 4 | P1 (+decoupling/calibration passages) |
| 27 | [YOLO-World](<../YOLO/YOLO-World real time open vocabulary object detection.pdf>) | 2024 | CVPR† | Open-vocabulary detection | 3 | P2 (§3.1–3.2) |
| 28 | [LLMDet](<../LLMDet Learning Strong Open-Vocabulary Object Detectors under the.pdf>) | 2025 | CVPR 2025† | OVD + LLM supervision | 4 | P1 |
| 29 | [Mamba](<../GSARP/MAMBA & VISION MAMBA PAPERS/Mamba-Linear-Time Sequence Modeling with Selective State Spaces.pdf>) | 2023 | arXiv; COLM 2024† | Sequence modelling: SSM | 5 | P1 |
| 30 | [Vision Mamba (Vim)](<../GSARP/MAMBA & VISION MAMBA PAPERS/Vision Mamba-Efficient Visual Representation Learning with Bidirectional.pdf>) | 2024 | ICML 2024 | Vision backbone: SSM | 4 | P1 |
| 31 | [VMamba](<../GSARP/MAMBA & VISION MAMBA PAPERS/VMamba-Visual State Space Model.pdf>) | 2024 | NeurIPS 2024† | Vision backbone: SSM | 4 | P1 |
| 32 | [MambaVision](<../MambaVision A Hybrid Mamba-Transformer Vision.pdf>) | 2024 | arXiv; CVPR 2025† | Hybrid backbone | 4 | P1 |
| 33 | [MambaOut](<../GSARP/MAMBA & VISION MAMBA PAPERS/MambaOut - Do We Really Need Mamba for Vision.pdf>) | 2024 | arXiv; CVPR 2025† | Critical analysis of SSMs | 3 | P1 |
| 34 | [Mamba-YOLO-World](<../Mamba-YOLO-World.pdf>) | 2024 | arXiv; ICASSP 2025† | OVD + SSM neck | 4 | P1 |
| 35 | [Native Sparse Attention (NSA)](<../Native Sparse Attention Hardware-Aligned and NativelyTrainable Sparse Attention.pdf>) | 2025 | arXiv; ACL 2025† | Efficient attention (LLMs) | 5 | P1 |
| 36 | [Pixel Distillation](<../Pixel Distillation Cost-flexible Distillation across Image Sizes and Heterogeneous Networks.pdf>) | 2021/24 | arXiv v2 (journal venue unconfirmed) | Knowledge distillation | 4 | P1 |
| 37 | [GPSFormer](<../GPSFormer A Global Perception and Local Structure Fitting-based Transformer for Point Cloud Understanding.pdf>) | 2024 | ECCV 2024† | 3D point clouds | 4 | P1 |
| 38 | [Human Interaction Recognition (HIR)](<../Human_Interaction_Recognition_in_Surveillance_Vide.pdf>) | 2024 | CMC vol. 81 no. 1 | Applied action recognition | 1 | P2 (§3.2–3.3) |
| 39 | [Auto MC-Reward](<../cvpr2024Minecraft/Auto MC-Reward Automated Dense Reward Design with Large Language Models for Minecraft.pdf>) | 2023/24 | CVPR 2024† | LLM agents + RL | 4 | P1 |
| 40 | [MP5](<../cvpr2024Minecraft/MP5 A Multi-modal Open-ended Embodied System in Minecraft via Active Perception.pdf>) | 2023/24 | CVPR 2024† | MLLM embodied agents | 3 | P1 |

### Importance classification (my judgement)

| Class | Papers (R#) | Why |
|---|---|---|
| **Foundational** (many others depend on it) | 2 Fast R-CNN, 5 Transformer, 10 CoOp, 29 Mamba | Each is the root of one sub-tree of your collection. |
| **Core to your research line** (open-vocab detection, VLM adaptation, real-time detection) | 12 ATPrompt, 20 RT-DETR, 21 YOLOv10, 26 UniDetector, 27 YOLO-World, 28 LLMDet, 7 Query2Label | These match your presentations and thesis area. |
| **Intermediate / lineage** | 4, 17, 18, 19, 22 (YOLO lineage), 11, 13, 14 (prompt variants), 15, 16 (VL pre-training), 8, 9, 30–34 | Build on a foundational paper with one or two new ideas. |
| **Reference / secondary** (skim, don't study) | 1, 3, 23, 24, 25 | Surveys and third-party descriptions; use them as lookup tables. Note: 24 and 25 are *not* written by the model creators (Ultralytics), so treat their claims as secondary. |
| **Peripheral to current goals** | 35 NSA, 36 Pixel Distillation, 37 GPSFormer, 38 HIR, 39 Auto MC-Reward, 40 MP5 | Different sub-fields (LLM efficiency, KD, 3-D, RL agents) or lower methodological depth. 38 is useful as a **critique exercise**. |

### Important papers that are *missing* from the folder

Several papers in your collection build directly on works that are **not in the folder**. These are the external papers I recommend adding (they are *not* inventoried, I only name them):

| Missing paper | Needed by (R#) | Priority |
|---|---|---|
| Radford et al., 2021, **CLIP**: *Learning Transferable Visual Models From Natural Language Supervision* | 10–14, 26–28, 34 (about 10 papers) | **Essential**, add before Stage 4 |
| Dosovitskiy et al., 2021, **ViT**: *An Image is Worth 16x16 Words* | 10–16, 22, 30–33 | **Essential** |
| He et al., 2016, **ResNet** | nearly all vision papers | **Essential** (short) |
| Carion et al., 2020, **DETR**: *End-to-End Object Detection with Transformers* | 20, 21, 28 | **Essential** before RT-DETR |
| Ren et al., 2015, **Faster R-CNN**; Lin et al., 2017, **FPN** | 3, 4, 17–22, 26 | Helpful |
| Devlin et al., 2019, **BERT** | 6, 15, 16 | Helpful |
| Zhou et al., 2022, **CoCoOp** | 11, 12 | Helpful |
| Li et al., 2022, **GLIP**; Liu et al., 2024, **Grounding DINO** | 27, 28, 34 | Helpful before Stage 5 |
| Gu et al., 2022, **S4** | 29 | Advanced (Mamba §2 summarises enough) |
| Hinton et al., 2015, **Distilling the Knowledge in a Neural Network** | 36 | Helpful |
| Hu et al., 2022, **LoRA** | 40 (only paper that uses LoRA), your goals | Helpful |
| **OW-OVD** and **Active Learning for VLMs (CEC)** | papers you presented | Add if still relevant |

---

## 3. Paper cards

Grouped by learning track. Prerequisite labels: **[E]** essential, **[H]** helpful, **[A]** advanced.

### Track A: Object detection foundations

#### R1. Object Detection (encyclopedia entry)
- **Authors:** Yali Amit, Pedro Felzenszwalb, Ross Girshick. **Year/venue:** 2020, Springer *Computer Vision: A Reference Guide*.
- **Problem:** Defines object detection and surveys the classic approaches: generative vs. discriminative models, scanning-window search, deformable part models, CNN detectors.
- **Contribution:** A conceptual primer, not a research contribution.
- **Prerequisites.** Math: [H] posterior ratios, Bayes' rule. ML: [E] classifier, training/test, overfitting. DL: [H] CNN at a high level. Impl: none.
- **Prereq papers:** none. **Read:** fully, about 1 hour. **Diff.** 1.

#### R2. Fast R-CNN
- **Authors:** Ross Girshick. **Year/venue:** 2015, ICCV†.
- **Problem:** R-CNN and SPPnet are slow and use multi-stage training (CNN fine-tuning, then SVMs, then box regressors) with features cached on disk.
- **Contribution:** Single-stage training of a network that jointly classifies proposals and regresses their boxes. **RoI pooling** shares the convolutional computation across proposals. 9× faster training and 213× faster testing than R-CNN (VGG16).
- **Method:** Shared conv feature map → RoI max-pooling to fixed size → FC layers → two sibling heads (softmax over K+1 classes; per-class box offsets). Multi-task loss = log loss + smooth-L1. Truncated SVD to speed up FC layers.
- **Prerequisites.** Math: [E] chain rule, [E] max-pool sub-gradient (the paper derives backprop through RoI pooling), [H] SVD. ML: [E] softmax classification, multi-task loss, mini-batch sampling. DL: [E] CNN, pooling, fine-tuning a pretrained ImageNet net. Impl: [H] RoIPool/RoIAlign in torchvision.
- **Prereq papers:** R1. **Read:** fully (9 pages). **Diff.** 2.
- **Why it matters (my reading):** it is a compact, honest paper with an explicit backprop derivation, which makes it a good first test of whether your gradient foundations are solid.

#### R3. Object Detection in 20 Years: A Survey
- **Authors:** Zhengxia Zou, Keyan Chen, Zhenwei Shi, Yuhong Guo, Jieping Ye. **Year/venue:** 2023, *Proceedings of the IEEE*.
- **Problem / contribution:** Reviews detection from the 1990s to 2022: milestone detectors, datasets, metrics, building blocks (multi-scale, context, NMS, hard-negative mining), speed-up techniques, and recent methods.
- **Prerequisites:** [E] IoU, precision/recall, AP. [H] CNN basics.
- **Read:** **selected sections** (milestones, metrics, NMS, multi-scale), then use the rest as a reference. **Diff.** 2.

#### R4. YOLOv3: An Incremental Improvement
- **Authors:** Joseph Redmon, Ali Farhadi. **Year/venue:** 2018, arXiv tech report.
- **Problem / contribution:** Incremental improvements to YOLO: anchor priors from k-means dimension clusters, sigmoid offsets $b_x=\sigma(t_x)+c_x$, $b_w=p_we^{t_w}$; independent logistic (multi-label) class prediction instead of softmax; predictions at 3 scales (FPN-like); Darknet-53 backbone. Also reports "things that didn't work".
- **Prerequisites.** Math: [E] sigmoid, exp/log parameterisation, k-means with an IoU distance. ML: [E] BCE vs. softmax CE, i.e. multi-label vs. multi-class. DL: [E] residual blocks, multi-scale feature maps, anchors.
- **Prereq papers:** R2 (helpful), R3. **Read:** fully (6 pages). **Diff.** 2.

### Track B: Attention, embeddings and set-prediction heads

#### R5. Attention Is All You Need
- **Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. **Year/venue:** 2017, NIPS.
- **Problem:** RNN sequence-to-sequence models are inherently sequential (they can't be parallelised within a sequence) and struggle with long-range dependencies.
- **Contribution:** The Transformer, built entirely on attention: scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder–decoder with residual + LayerNorm. State-of-the-art BLEU on WMT14 at a fraction of the training cost.
- **Prerequisites.** Math: [E] matrix multiplication and shapes, dot product as similarity, softmax, [H] variance of a dot product (the reason for $1/\sqrt{d_k}$), sinusoids. ML: [E] cross-entropy, [H] label smoothing. DL: [E] embeddings, residual connections, LayerNorm, masking, Adam with warm-up.
- **Prereq papers:** none in folder. **Read:** fully. Derive Eq. 1 and the shape of every tensor. **Diff.** 3.
- **Dependants:** about 30 papers in the folder.

#### R6. Sentence-BERT
- **Authors:** Nils Reimers, Iryna Gurevych. **Year/venue:** 2019, EMNLP-IJCNLP†.
- **Problem:** BERT is a *cross-encoder*. Finding the most similar pair among 10,000 sentences needs about 50M forward passes (around 65 h).
- **Contribution:** Siamese/triplet fine-tuning of BERT to produce fixed-size sentence embeddings comparable with cosine similarity (about 5 s for the same search).
- **Prerequisites.** Math: [E] cosine similarity, norms. ML: [E] classification/regression objectives, triplet loss. DL: [H] BERT, pooling strategies.
- **Why here (my reading):** the cleanest explanation of **bi-encoder vs. cross-encoder**, which is exactly CLIP's dual-encoder design decision and the reason YOLO-World can cache "offline vocabulary" embeddings. **Diff.** 2.

#### R7. Query2Label
- **Authors:** Shilong Liu, Lei Zhang, Xiao Yang, Hang Su, Jun Zhu. **Year/venue:** 2021, arXiv.
- **Problem:** Multi-label classification suffers from positive/negative imbalance and from global pooling that dilutes small objects.
- **Contribution:** Learned label embeddings act as **queries** in a Transformer decoder that cross-attends to backbone features, followed by per-class binary classifiers. 91.3% mAP on MS-COCO.
- **Prerequisites.** Math: [E] attention. ML: [E] BCE, one-vs-all, [E] mAP for multi-label, [H] focal/asymmetric loss. DL: [E] cross-attention, ResNet/ViT backbones.
- **Prereq papers:** R5. **Diff.** 2. **Why here:** the simplest "learnable query + cross-attention" model in the collection, and a gentle on-ramp to DETR/RT-DETR object queries.

#### R8. C-Tran: General Multi-label Image Classification with Transformers
- **Authors:** Jack Lanchantin, Tianlu Wang, Vicente Ordonez, Yanjun Qi. **Year/venue:** 2021, CVPR†.
- **Problem / contribution:** A Transformer *encoder* jointly over CNN features and label embeddings, trained with **label-mask training** (ternary label states: positive/negative/unknown), so it can use partial labels at inference.
- **Prerequisites:** R5, [E] BCE, [H] masked-modelling idea (BERT). **Diff.** 3.

#### R9. ML-Decoder
- **Authors:** Tal Ridnik, Gilad Sharir, Avi Ben-Cohen, Emanuel Ben-Baruch, Asaf Noy. **Year/venue:** 2021, WACV 2023†.
- **Problem:** Transformer-decoder heads scale quadratically with the number of classes and global average pooling (GAP) wastes spatial information.
- **Contribution:** Attention-based head with **group decoding** (scales to thousands of classes), word-vector queries for zero-shot, and query augmentation. 91.4% COCO mAP, 80.7% ImageNet with ResNet-50.
- **Prerequisites:** R7, [E] complexity analysis of attention, [H] zero-shot learning with word embeddings. **Diff.** 3.

### Track C: Vision-language models and prompt learning

#### R10. CoOp: Learning to Prompt for Vision-Language Models
- **Authors:** Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu. **Year/venue:** 2022, IJCV†.
- **Problem:** Hand-crafted prompts ("a photo of a [CLASS].") for CLIP are sub-optimal and expensive to tune.
- **Contribution (verified in §3):** Replaces the context words with $M$ learnable vectors $[V]_1\dots[V]_M$ (512-d for CLIP) while **all CLIP weights are frozen**. Two variants: unified context and class-specific context. Prediction is $p(y=i\mid x)=\frac{\exp(\cos(g(t_i),f)/\tau)}{\sum_j\exp(\cos(g(t_j),f)/\tau)}$, trained with cross-entropy, with gradients back-propagated **through the frozen text encoder** into the context vectors. With 16 shots it gains about 15% on average over hand-crafted prompts.
- **Prerequisites.** Math: [E] softmax with temperature, cosine similarity, chain rule through a frozen function. ML: [E] few-shot evaluation, train/val/test discipline, [E] domain shift. DL: [E] CLIP dual encoder, [E] token embeddings, [H] BPE tokenisation, `requires_grad=False`.
- **Prereq papers:** **CLIP (external, essential)**, R5. **Read:** fully. **Diff.** 2. **Lesson link:** [Lesson 01](lessons/Lesson_01.md) derives exactly the gradient CoOp relies on.

#### R11. Soft Prompt Generation (SPG) for Domain Generalization
- **Authors:** Shuanghao Bai, Yuedi Zhang, Wanqi Zhou, Zhirong Luan, Badong Chen. **Year/venue:** 2024, ECCV†.
- **Problem:** Learned prompts (CoOp/CoCoOp) lack diversity and ignore unseen domains.
- **Contribution:** Two-stage training. (I) Learn a "domain prompt label" per source domain. (II) Train a **conditional GAN** that generates instance-specific soft prompts from image features. At inference, the generator produces prompts for the unseen target domain.
- **Prerequisites.** Math: [E] min-max objectives, [H] KL divergence. ML: [E] domain generalization protocol (leave-one-domain-out). DL: [E] GAN/CGAN training, R10. **Diff.** 3.

#### R12. ATPrompt: Textual Prompt Learning with Embedded Attributes
- **Authors:** Zheng Li, Yibing Song, Penghai Zhao, Ming-Ming Cheng, Xiang Li, Jian Yang. **Year/venue:** arXiv Dec 2024; ICCV 2025†.
- **Problem:** Soft prompts align images only to *known* categories, so alignment to unknown categories is poor (base-to-novel gap).
- **Contribution (verified in §3.1–3.2):** Inserts fixed **attribute tokens** (e.g. [A], [B]) between soft tokens: $P_T=[T_{a_1}..][A][T_{b_1}..][B][T_1..T_M][CLS]$. Shallow (input-level) and deep (per-layer, à la MaPLe/PromptSRC) versions. Attributes are chosen by a **differentiable attribute search** over an LLM-generated candidate pool. Plug-in for existing textual prompt methods.
- **Prerequisites.** Math: [E] softmax relaxation of a discrete choice (DARTS-style search). ML: [E] base-to-novel and cross-dataset protocols. DL: [E] R10, [H] deep prompt tuning (MaPLe). **Diff.** 3. **You presented this paper (Mar 2025).**

#### R13. TP-CLIP: Is Temporal Prompting All We Need for Limited Labeled Action Recognition?
- **Authors:** Shreyank N Gowda, Boyan Gao, Xiao Gu, Xiaobo Jin. **Year/venue:** 2025, arXiv.
- **Problem / contribution:** Adapts image CLIP to video with **temporal visual prompts** without changing the CLIP architecture. Reports 1/3 the GFLOPs and 1/28 the tunable parameters of a recent SOTA, with gains up to 15.8% in zero-/few-shot settings.
- **Prerequisites:** R10, [E] visual prompt tuning (VPT idea), [H] video representations (frames, temporal pooling). **Diff.** 3.

#### R14. CLIP-RC: Exploring Regional Clues in CLIP for Zero-Shot Semantic Segmentation
- **Authors:** Yi Zhang, Meng-Hao Guo, Miao Wang, Shi-Min Hu. **Year/venue:** CVPR (2024†).
- **Problem:** CLIP's image-level knowledge doesn't transfer to pixel-level prediction, and fine-tuning overfits to seen classes.
- **Contribution:** A **region-level bridge** (region alignment module) and a **recovery decoder with recovery loss** that limits forgetting of CLIP's generalisation. SOTA on VOC, Context and COCO-Stuff ZS3.
- **Prerequisites:** R10, [E] ViT patch tokens, [E] segmentation basics (mIoU, decoders), [H] catastrophic forgetting / regularising towards a frozen model. **Diff.** 4.

#### R15. BLIP
- **Authors:** Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi. **Year/venue:** 2022, ICML†.
- **Problem:** VLP models excel at *either* understanding or generation, and noisy web captions are poor supervision.
- **Contribution:** **MED**, a multimodal mixture of encoder-decoder (unimodal encoder / image-grounded encoder / image-grounded decoder) trained with three losses: **ITC** (contrastive), **ITM** (binary matching with hard negatives), **LM** (captioning). **CapFilt** bootstraps the data with a captioner plus a filter.
- **Prerequisites.** Math: [E] InfoNCE-type contrastive loss. DL: [E] cross-attention, causal masking, [H] momentum distillation (ALBEF). **Diff.** 3.

#### R16. BEiT-3: Image as a Foreign Language
- **Authors:** Wenhui Wang, Hangbo Bao, Li Dong, et al. (Microsoft). **Year/venue:** 2022, CVPR 2023†.
- **Contribution:** **Multiway Transformer** (shared self-attention, modality-specific FFN experts) pretrained with a single masked "language" modelling objective on images, texts and image–text pairs. SOTA across detection, segmentation, classification, VQA, retrieval.
- **Prerequisites:** R5, R15, [E] masked image modelling (BEiT, visual tokenizer), [H] mixture-of-experts idea. **Read:** §2 method + Table 1. **Diff.** 4.

### Track D: Real-time detection lineage (YOLO and DETR)

#### R17. YOLOv4
- **Authors:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao. **Year/venue:** 2020, arXiv.
- **Contribution:** A systematic "bag of freebies / bag of specials" study: CSPDarknet53, SPP, PAN, Mosaic augmentation, CmBN, DropBlock, Mish, **CIoU loss**, label smoothing, self-adversarial training. 43.5% AP at 65 FPS (V100).
- **Prerequisites:** [E] IoU-family losses (IoU→GIoU→DIoU→CIoU), [E] BatchNorm statistics, [E] data augmentation as regularisation, [H] receptive field. **Read:** §3 + ablation tables. **Diff.** 3.

#### R18. YOLOv7
- **Authors:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao. **Year/venue:** 2022, CVPR 2023†.
- **Contribution:** E-ELAN architecture, model scaling for concatenation-based nets, **planned re-parameterised convolution**, coarse-to-fine lead/auxiliary-head label assignment. Trained from scratch on COCO.
- **Prerequisites:** [E] structural re-parameterisation (RepVGG: merging conv+BN and parallel branches *is linear algebra*), [E] label assignment, [H] deep supervision. **Diff.** 3.

#### R19. YOLOv9: Programmable Gradient Information
- **Authors:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao. **Year/venue:** 2024, ECCV†.
- **Contribution:** Frames deep-network information loss with the **information bottleneck** and **reversible functions**. Proposes **PGI** (an auxiliary reversible branch providing reliable gradients, removed at inference) and the **GELAN** architecture.
- **Prerequisites.** Math: [E] mutual information, data-processing inequality, [H] invertible functions. DL: [E] auxiliary supervision, gradient flow. **Diff.** 4. *(my reading: the information-theory argument is informal, which makes it a good critique target.)*

#### R20. RT-DETR: DETRs Beat YOLOs on Real-time Object Detection
- **Authors:** Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen. **Year/venue:** 2023 (v3 2024), CVPR 2024†.
- **Problem:** NMS slows down YOLOs and destabilises them, while DETRs are NMS-free but too slow.
- **Contribution:** Analyses NMS latency. Introduces an **efficient hybrid encoder** (intra-scale attention AIFI on the top level only + CNN cross-scale fusion CCFF), **uncertainty-minimal query selection**, and speed tuning by dropping decoder layers. R50: 53.1 AP at 108 FPS (T4).
- **Prerequisites:** **DETR (external, essential)**: [E] bipartite/Hungarian matching, set-prediction loss, object queries; [E] deformable attention (helpful-to-essential); R5, R7. **Diff.** 4. **You presented this paper (Jul 2024).**

#### R21. YOLOv10: Real-Time End-to-End Object Detection
- **Authors:** Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. **Year/venue:** 2024, NeurIPS 2024†.
- **Contribution (verified §3.1):** **Consistent dual assignments**: a one-to-many head (TAL, rich supervision) and a one-to-one head (top-1 matching, NMS-free) trained jointly with the same metric $m(\alpha,\beta)=s\cdot p^{\alpha}\cdot \mathrm{IoU}(\hat b,b)^{\beta}$, with the one-to-many head discarded at inference. Plus an efficiency–accuracy-driven redesign of the components.
- **Prerequisites:** R20 (for the one-to-one idea), [E] label assignment (TAL), [E] NMS. **Diff.** 3.

#### R22. YOLOv12: Attention-Centric Real-Time Object Detectors
- **Authors:** Yunjie Tian, Qixiang Ye, David Doermann. **Year/venue:** 2025, arXiv tech report (NeurIPS 2025†).
- **Contribution:** **Area attention** (splits the feature map into areas to cut attention cost), **R-ELAN**, and architectural tweaks (FlashAttention, removing positional encoding, MLP ratio). Matches CNN-YOLO speed with higher mAP.
- **Prerequisites:** R5, R18, [E] attention FLOPs/memory analysis, [H] FlashAttention and memory-bound vs. compute-bound kernels. **Diff.** 4.

#### R23–R25. Secondary YOLO reviews
- **R23 YOLOv8→YOLO11 review:** Priyanto Hidayatullah et al., 2025, arXiv. Block-by-block diagrams reconstructed from source code. **Use as an architecture lookup** (C2f, C3k2, SPPF, C2PSA, decoupled head, DFL). Diff. 2.
- **R24 YOLOv11 overview:** Rahima Khanam, Muhammad Hussain, 2024, arXiv. Short descriptive overview of C3k2, SPPF, C2PSA. Diff. 1. *(my reading: largely descriptive; little evidence beyond vendor numbers.)*
- **R25 YOLO26:** Ranjan Sapkota, Rahul Harsha Cheppally, Ajay Sharda, Manoj Karkee, arXiv 2509.25164 (v3 Jan 2026). Describes Ultralytics YOLO26: DFL removal, NMS-free inference, ProgLoss, STAL, MuSGD optimizer, edge benchmarks, INT8/FP16 export. Diff. 2. *Third-party analysis: verify claims against the official docs/code.*
- **Prerequisites (all three):** R4, R17, R21, [E] DFL (distribution focal loss), [H] quantization.

### Track E: Open-vocabulary detection

#### R26. UniDetector: Detecting Everything in the Open World
- **Authors:** Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, Shengjin Wang. **Year/venue:** 2023, CVPR†.
- **Problem:** Detectors depend on human annotations of a fixed label space and fail on novel categories and scenes.
- **Contribution:** (1) Training on multiple sources with **heterogeneous label spaces** via image–text alignment (partitioned structure). (2) **Decoupled** proposal generation and RoI classification, with a class-agnostic localization network (CLN). (3) **Probability calibration**: estimate class priors and de-bias predictions. Detects 7k+ categories after training on about 500.
- **Prerequisites.** Math: [E] Bayes' rule / prior correction. ML: [E] long-tail evaluation (LVIS APr/APc/APf). DL: [E] two-stage detectors (R2 + Faster R-CNN), CLIP (external). **Diff.** 4. **You presented this paper (Oct 2023).**

#### R27. YOLO-World
- **Authors:** Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan. **Year/venue:** 2024, CVPR†.
- **Contribution (verified §3.1–3.2):** Reformulates boxes and labels as **region–text pairs**. YOLOv8 detector + frozen CLIP text encoder + **RepVL-PAN** (text-guided CSP layers and image-pooling attention). Text-contrastive head $s_{k,j}=\alpha\,\hat e_k^{\top}\hat w_j+\beta$ with L2-normalised embeddings. Online vocabulary in training (80 nouns per mosaic). **Prompt-then-detect** with an offline vocabulary at inference (re-parameterised into the weights). 35.4 AP on LVIS at 52 FPS.
- **Prerequisites:** R4/R23 (YOLOv8), R6 (bi-encoder caching), CLIP (external), [E] region–text contrastive loss, [H] GLIP. **Diff.** 3.

#### R28. LLMDet
- **Authors:** Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, Wei-Shi Zheng. **Year/venue:** 2025, CVPR 2025†.
- **Contribution:** **GroundingCap-1M** (grounding labels + image-level detailed captions). Fine-tunes MM-Grounding-DINO with the grounding loss plus **caption generation losses** (image-level long captions and region-level short captions) produced through an LLM. The improved detector in turn builds a stronger large multimodal model.
- **Prerequisites:** R20/DETR, Grounding DINO (external), R15 (LM loss), [E] autoregressive language-modelling loss. **Diff.** 4.

### Track F: State-space models and efficient token mixing

#### R29. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors:** Albert Gu, Tri Dao. **Year/venue:** arXiv Dec 2023; COLM 2024†.
- **Problem:** Sub-quadratic models (linear attention, S4, etc.) can't perform content-based reasoning.
- **Contribution:** **Selective SSM**: the parameters $\Delta, B, C$ become functions of the input, which allows selective remembering and forgetting. A **hardware-aware parallel scan** replaces the convolution mode. The simplified Mamba block has no attention and no MLP. 5× inference throughput, linear in sequence length, and Mamba-3B matches Transformers of twice its size.
- **Prerequisites.** Math: [E] linear ODEs $h'(t)=Ah(t)+Bx(t)$, [E] discretisation (zero-order hold), [E] linear recurrences ↔ convolutions, [A] parallel prefix scan, [H] GPU memory hierarchy (SRAM vs. HBM). DL: [E] RNNs and gating, R5. **Read:** §1–3 first, synthetic tasks in §4.1, skip the kernels initially. **Diff.** 5.

#### R30. Vision Mamba (Vim)
- **Authors:** Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang. **Year/venue:** 2024, ICML 2024.
- **Contribution:** ViT-style patch sequence + position embeddings + **bidirectional** Mamba blocks. 2.8× faster than DeiT and 86.8% less GPU memory at 1248² resolution.
- **Prerequisites:** R29, ViT/DeiT (external). **Diff.** 4.

#### R31. VMamba: Visual State Space Model
- **Authors:** Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, Yunfan Liu. **Year/venue:** 2024, NeurIPS 2024†.
- **Contribution:** **SS2D**, a four-direction 2-D selective scan, inside VSS blocks of a hierarchical backbone, plus a series of speed optimisations. Analyses input-scaling efficiency.
- **Prerequisites:** R29, Swin-style hierarchical backbones (external, helpful). **Diff.** 4.

#### R32. MambaVision
- **Authors:** Ali Hatamizadeh, Jan Kautz (NVIDIA). **Year/venue:** 2024, CVPR 2025†.
- **Contribution:** Redesigned Mamba mixer for vision (non-causal conv, symmetric branch) and **hybrid** stages with self-attention blocks in the final layers. New ImageNet accuracy–throughput Pareto front.
- **Prerequisites:** R29–R31, R5. **Diff.** 4.

#### R33. MambaOut: Do We Really Need Mamba for Vision?
- **Authors:** Weihao Yu, Xinchao Wang. **Year/venue:** 2024, CVPR 2025†.
- **Contribution:** A conceptual argument that SSMs suit **long-sequence** and **autoregressive (causal)** tasks. ImageNet classification is neither, so the SSM should be unnecessary there. MambaOut (Gated CNN blocks with the SSM removed) beats visual Mamba models on ImageNet but not on detection/segmentation.
- **Prerequisites:** R29–R31, [E] causal vs. non-causal token mixing, [E] how to design a hypothesis-driven ablation. **Diff.** 3. *(Excellent model of a research critique paper.)*

#### R34. Mamba-YOLO-World
- **Authors:** Haoxuan Wang, Qingdong He, Jinlong Peng, Hao Yang, Mingmin Chi, Yabiao Wang. **Year/venue:** 2024, ICASSP 2025†.
- **Contribution:** Replaces YOLO-World's RepVL-PAN neck with **MambaFusion-PAN**: Parallel-Guided and Serial-Guided Selective Scan for linear-complexity multimodal fusion. Beats YOLO-World on COCO/LVIS zero-shot and fine-tuning at similar params/FLOPs.
- **Prerequisites:** R27 **and** R29. **Diff.** 4. *(This is where Tracks E and F merge.)*

#### R35. Native Sparse Attention (NSA)
- **Authors:** Jingyang Yuan, Huazuo Gao, Damai Dai, et al. (DeepSeek-AI, PKU, UW). **Year/venue:** 2025, ACL 2025†.
- **Contribution:** Hierarchical sparse attention (**compressed** coarse tokens + **selected** fine-grained blocks + **sliding window**) with learned gates. Kernels balanced for arithmetic intensity and natively trainable end to end. Matches or beats full attention, with large speedups at 64k context.
- **Prerequisites:** R5, [E] KV-cache, GQA/MQA, [A] arithmetic intensity / roofline model, Triton kernels. **Diff.** 5. *Peripheral for vision, but it connects to YOLOv12's efficiency arguments.*

### Track G: Peripheral and special topics

#### R36. Pixel Distillation
- **Authors:** Guangyu Guo, Dingwen Zhang, Longfei Han, Nian Liu, Ming-Ming Cheng, Junwei Han. **Year/venue:** arXiv 2112.09532 v2 (Jul 2024), IEEE-journal format (venue not confirmed).
- **Contribution:** Extends knowledge distillation to the **input resolution** (a low-res student). **ISRD** (input spatial representation distillation) works across CNN↔ViT. A **Teacher–Assistant–Student** framework splits model compression from input compression. **AFP** adapts it to detection.
- **Prerequisites:** [E] KD loss (soft targets, temperature, KL), [E] R2/R4 detection heads. **Diff.** 4.

#### R37. GPSFormer
- **Authors:** Changshuo Wang, Meiqing Wu, Siew-Kei Lam, Xin Ning, Shangshu Yu, Ruiping Wang, Weijun Li, Thambipillai Srikanthan. **Year/venue:** 2024, ECCV†.
- **Contribution:** For point clouds: a **Global Perception Module** (adaptive deformable graph conv + MHA) and **Local Structure Fitting Conv** (Taylor-series-inspired low-/high-order local geometry).
- **Prerequisites:** [E] PointNet / PointNet++ (external), [E] kNN graphs, [H] Taylor expansion. **Diff.** 4. *Only relevant if you move to 3-D vision.*

#### R38. Human Interaction Recognition in Surveillance Videos (HIR)
- **Authors:** Vesal Khean, Chomyong Kim, Sunjoo Ryu, Awais Khan, Min Kyung Hong, Eun Young Kim, Joungmin Kim, Yunyoung Nam. **Year/venue:** 2024, *CMC* 81(1).
- **Method (verified §3.2–3.3):** OpenPose keypoints (25 per person) from 51 frames per video → fine-tuned 2-D ResNet features → six classical classifiers (SVM, KNN, RF, DT, NB, XGBoost). Reports 95.45% accuracy on 5 interaction classes, 13 participants, 4 cameras.
- **Critique hook (my reading, still to verify in full):** §3.2 describes a single 80/20 split of 550 videos "comprised of combining four channel cameras". It does not state a **subject-wise or recording-wise split**, nor a validation set. If the same interaction recorded from different cameras lands in both train and test, the accuracy is inflated by **leakage**. This makes it a good data-leakage case study (see [03_Machine_Learning](03_Machine_Learning.md#data-leakage)).
- **Prerequisites:** [E] train/val/test methodology, [E] classical classifiers, [H] pose estimation. **Diff.** 1.

#### R39. Auto MC-Reward
- **Authors:** Hao Li, Xue Yang, Zhaokai Wang, Xizhou Zhu, Jie Zhou, Yu Qiao, Xiaogang Wang, Hongsheng Li, Lewei Lu, Jifeng Dai. **Year/venue:** 2023/24, CVPR 2024†.
- **Contribution:** LLM-based Reward Designer (writes Python reward functions), Reward Critic (checks the code), and Trajectory Analyzer (diagnoses failures) iteratively build **dense rewards** for sparse-reward Minecraft tasks.
- **Prerequisites:** [E] RL basics (MDP, return, policy gradient/PPO, reward shaping). **Diff.** 4. *Peripheral.*

#### R40. MP5
- **Authors:** Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, Jing Shao. **Year/venue:** 2023/24, CVPR 2024†.
- **Contribution:** Modular MLLM agent (Parser, Percipient, Planner, Performer, Patroller) with **goal-conditioned active perception**. The Percipient is a **LoRA-enabled MLLM** (the only paper in the folder that uses LoRA). 22% success on hard process-dependent tasks, 91% on context-dependent ones.
- **Prerequisites:** [H] MLLMs (LLaVA-style), [H] LoRA, [H] agent planning. **Diff.** 3. **You presented this paper (Apr 2024).**
