# PROGRESS — Persistent Learning Record

> Back to [README](README.md) · Roadmap: [01](01_Learning_Roadmap.md)
>
> **Rule:** a status above "In progress" requires *evidence*: an answer, derivation, exercise result or implementation. Reading an explanation is never evidence.
>
> **Status scale:** `Not started` → `In progress` → `Needs review` → `Demonstrated understanding` → `Applied successfully`

## 1. Current position

| Field | Value |
|---|---|
| Course start | 2026-10-24 |
| Current stage | **Stage 0: Foundations** (pre-diagnostic) |
| Last session | 2026-10-24: course set up, papers inventoried, roadmap v1, Lesson 01 and Exercise 01 written |
| **Next recommended activity** | **Answer the [Diagnostic (Exercise 00)](exercises/Exercise_00_Diagnostic.md)**, then study [Lesson 01](lessons/Lesson_01.md) and do [Exercise 01](exercises/Exercise_01.md) (compressed if the diagnostic shows mastery) |
| Pending decisions | D-0a (priorities), D-0b (time/GPU) |

## 2. Known information (from files, not from assessment)

- **Presentations you gave** (from `presentation/`): UniDetector (Oct 2023, mentor Wang Rongfang), MP5 (Apr 2024), RT-DETR (Jul 2024), ATPrompt (Mar 2025), Active Learning for VLMs/CEC (May 2025), OW-OVD (Oct 2025).
- **Local environments observed:** conda envs `mdetr_env`, `yolo_world`, `yolov9`, `yolo_bear`, `multimodal-rag`, `rag`, `labelme`, `tf-env`. This suggests hands-on work with MDETR, YOLO-World, YOLOv9 and a custom YOLO project. PyTorch 2.4–2.8 is available, and `yolo_bear` has CUDA 11.8. *Familiarity with code is not taken as evidence of conceptual mastery.*

## 3. Diagnostic and assessment results

| Assessment | Date | Status | Summary |
|---|---|---|---|
| [Exercise 00: Diagnostic](exercises/Exercise_00_Diagnostic.md) | — | **Not started (awaiting your answers)** | — |

## 4. Lessons

| Lesson | Topic | Status | Evidence / notes |
|---|---|---|---|
| [L01](lessons/Lesson_01.md) | Softmax, CE, temperature, gradients through frozen modules | Not started | — |
| L02 | Backprop by hand; autograd | Not written | Written after the diagnostic |
| L03 | Optimisation (SGD, AdamW, schedules) | Not written | |
| L04 | Generalisation and experimental methodology | Not written | |
| L05 | CNN mechanics | Not written | |
| L06 | Detection geometry, NMS, AP | Not written | |

## 5. Concept tracker

| Concept | Status | Evidence |
|---|---|---|
| Softmax-CE gradient $p-y$ | Not started | |
| Temperature and cosine logits | Not started | |
| Gradient flow through frozen modules | Not started | |
| Chain rule / Jacobians (vector form) | Not started | |
| Backprop through an MLP | Not started | |
| Optimisers (AdamW, schedules) | Not started | |
| Train/val/test and leakage | Not started | |
| Conv shapes, BN semantics | Not started | |
| IoU, NMS, AP computation | Not started | |
| Attention (shapes, scaling) | Not started | |

## 6. Derivations completed

| Derivation | Date | Result |
|---|---|---|
| — | | |

## 7. Implementations completed

| Exercise | Date | Result |
|---|---|---|
| [Exercise 01](exercises/Exercise_01.md) | — | Not started |

## 8. Papers

| R# | Paper | Status | Comprehension | Notes |
|---|---|---|---|---|
| all 33 | see [00_Paper_Inventory](00_Paper_Inventory.md) | Not started (course reading) | — | Inventory is at P1/P2 inspection, *not* a study of the papers |

## 9. Current weaknesses

*Unknown until the diagnostic is answered.* I won't guess.

## 10. Questions to revisit

- Do you still want OW-OVD and CEC (active learning for VLMs) in the course? Their PDFs are not in the folder.
- Add the missing essential external papers (CLIP, ViT, DETR, ResNet) to the folder before Stages 3–4.

## 11. Roadmap changes

| Date | Change | Reason |
|---|---|---|
| 2026-10-24 | Roadmap v1 created | Initial setup |
| 2026-10-24 | Roadmap v2: scope narrowed to 33 papers; papers renumbered R1–R33, lessons L01–L20 | Your decision to narrow the course scope |
