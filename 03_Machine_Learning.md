# 03 — Machine Learning Fundamentals

> Back to [README](README.md) · Roadmap: [01](01_Learning_Roadmap.md) · Math: [02](02_Mathematics_Foundations.md)
>
> This file covers the principles that decide whether an **experiment** (yours or a paper's) supports its claim. Each topic is tied to the papers where it matters. Priority: **E** essential · **H** helpful · **A** advanced.

## 1. Topic map

| Topic | Pri. | Where it bites in your papers | Lesson |
|---|---|---|---|
| Supervised / self-supervised / weakly-supervised learning | E | CLIP & BLIP (web image–text pairs as weak supervision), BEiT-3 (masked modelling), LLMDet (LLM pseudo-captions) | L04, L10 |
| Classification vs. regression; multi-class vs. **multi-label** | E | YOLOv3 (logistic classes), Query2Label/C-Tran/ML-Decoder | L01, L09 |
| Loss = negative log-likelihood; ERM | E | every paper | L01, L04 |
| Train / val / test, and **data leakage** | E | HIR (R38), all few-shot VLM papers (selecting on test?) | **L04** |
| Over/under-fitting, bias–variance | E | CLIP-RC (overfitting to seen classes), prompt learning (base→novel drop) | L04 |
| Regularisation: weight decay, dropout, augmentation, label smoothing, early stopping | E | YOLOv4 "bag of freebies", Transformer (label smoothing) | L03, L14 |
| Generalisation under **distribution shift** (domain generalisation, base-to-novel, open-vocabulary) | E | CoOp, SPG, ATPrompt, TP-CLIP, UniDetector, YOLO-World | L12, L18 |
| Class imbalance and long tail | E | Query2Label (pos/neg imbalance), UniDetector & LVIS (rare classes), focal loss | L09, L18 |
| Evaluation metrics | E | see §3 | L06, L09 |
| Model selection, hyper-parameter tuning | E | every SOTA table | L04 |
| Reproducibility: seeds, variance, compute budgets | E | small-margin claims (YOLO versions, prompt methods) | L04 |
| Classical models (SVM, kNN, RF, DT, NB, boosting) | H | HIR (R38) | L04 |
| Calibration, uncertainty, active learning | H | UniDetector calibration, your CEC presentation | L18 |
| Few-shot / zero-shot protocols | E | CoOp (1–16 shots), TP-CLIP, ML-Decoder ZSL | L12 |

## 2. Key ideas (short form; full treatment in lessons)

### Empirical risk minimisation
$$\hat\theta=\arg\min_\theta \frac1n\sum_{i=1}^n \ell(f_\theta(x_i),y_i)+\lambda\,\Omega(\theta)$$
We minimise the average loss on the *sample*, hoping it approximates the expected loss on the *distribution*. Every generalisation problem in your papers (base→novel, source→target domain, seen→unseen class) is a case where **the test distribution differs from the training distribution**, so ERM's guarantee no longer applies directly.

### Data leakage
Leakage = information from the evaluation set influences training or model selection. Common forms in vision:

1. **Near-duplicate leakage:** the same scene, subject or recording appears in train and test (e.g. multiple cameras of the same event, consecutive video frames).
2. **Selection leakage:** hyper-parameters or checkpoints chosen on the test set. Few-shot papers are especially exposed, because a validation set often doesn't exist.
3. **Pre-training leakage:** the pre-training corpus (e.g. CLIP's 400M pairs, Objects365, GoldG) overlaps the "zero-shot" test classes or images. This matters for every OVD claim.
4. **Pre-processing leakage:** normalisation statistics or feature selection computed on all data.

*Case study (Stage 0 exercise):* HIR (R38) §3.2 describes an 80/20 split of 550 videos pooled across four cameras, without stating a subject-wise or recording-wise split or a validation set. Your task in [Exercise 00](exercises/Exercise_00_Diagnostic.md) is to decide whether this can inflate the 95.45% accuracy and what protocol would fix it.

### Bias–variance and the base-to-novel trade-off
CoOp-style prompts fit base classes well (low bias on base) but can overfit them, hurting novel classes. ATPrompt, CoCoOp, CLIP-RC's recovery loss and SPG are all variance-control or anti-forgetting devices. Learn to name **which generalisation gap** each method attacks.

## 3. Metrics you must be able to compute by hand

| Metric | Used in | Definition sketch | Lesson |
|---|---|---|---|
| Accuracy, top-k | ImageNet, prompt learning | fraction correct | L04 |
| Precision, recall, F1 | everywhere | $P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$ | L04 |
| **AP / mAP (detection)**, AP50, AP@[.5:.95], AP_S/M/L | all detection | area under the interpolated PR curve per class, averaged (COCO: also over IoU thresholds) | **L06** |
| LVIS AP_r / AP_c / AP_f | UniDetector, YOLO-World, LLMDet | AP split by class frequency (rare/common/frequent) | L18 |
| mAP (multi-label classification) | Query2Label, C-Tran, ML-Decoder | per-class AP over images, averaged | L09 |
| Harmonic mean (base-to-novel) | CoOp-family | $H=\frac{2\,B\,N}{B+N}$ | L12 |
| mIoU | CLIP-RC, BEiT-3, Vim/VMamba (ADE20K) | per-class IoU of pixel sets, averaged | L13 |
| Recall@K, retrieval | BLIP, BEiT-3, SBERT | fraction of queries whose match is in the top-K | L08 |
| BLEU, CIDEr | Transformer, BLIP | n-gram overlap metrics | L13 |
| U-Recall, WI, A-OSE (open-world) | OW-OVD (your presentation) | recall of unknowns, wilderness impact | L18 |
| Latency, FPS, FLOPs, params | all YOLO/DETR/Mamba | measure with a fixed batch, precision and hardware; FLOPs ≠ latency | L17 |

## 4. Experimental design checklist (applied to every paper in Stage 8 critiques)

1. Is there a held-out **validation** set separate from test? Who chose the hyper-parameters, and on what?
2. Are baselines given the **same** data, schedule, backbone, resolution and pre-training?
3. How many **seeds**? Are gains larger than the seed-to-seed variance?
4. Do **ablations** isolate one change at a time?
5. Could the gain come from **extra data or compute** rather than the idea (e.g. LLMDet's new dataset, YOLOv4's tricks)?
6. Is the "zero-shot" setting really zero-shot w.r.t. the pre-training data?
7. Are speed numbers measured **end to end** (including NMS)? RT-DETR and YOLOv10 make this point themselves.
