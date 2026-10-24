# ML Foundations Rebuilding Course

A personal, paper-driven course to rebuild the mathematical and conceptual foundations of machine learning and deep learning, organised around the 40 research papers in `E:\Papers`. The aim is to read, reproduce, critique and extend computer-vision / vision-language research at PhD level.

## Purpose

You (MSc in AI, Xidian University) have implemented and presented detection and VLM papers, but sometimes followed implementations without fully owning the underlying mathematics. This course closes those gaps **in the order your papers need them**, and it skips what you can already demonstrate.

## Your learning objectives

1. Derive and explain the core math (gradients, softmax/CE, attention, contrastive losses, label assignment, SSM discretisation) rather than just recognising it.
2. Understand *why* each architecture component exists, and when it fails.
3. Read the papers in the roadmap independently, then reconstruct, implement and critique them.
4. Develop well-scoped, testable research ideas in open-vocabulary detection / VLM adaptation (or wherever the diagnostic and your priorities point).

## Course map

| File | Contents |
|---|---|
| [00_Paper_Inventory.md](00_Paper_Inventory.md) | All 40 papers: metadata, problem, contribution, prerequisites, difficulty, inspection level, missing external papers |
| [01_Learning_Roadmap.md](01_Learning_Roadmap.md) | Stages, dependency graph, reading order with readiness tests, change log |
| [02_Mathematics_Foundations.md](02_Mathematics_Foundations.md) | Prioritised math map (linear algebra, calculus, probability, optimisation, information theory) |
| [03_Machine_Learning.md](03_Machine_Learning.md) | ERM, leakage, generalisation and shift, metrics, experimental-design checklist |
| [04_Deep_Learning.md](04_Deep_Learning.md) | Networks, backprop, losses, normalisation, CNNs, PyTorch ↔ math bridge |
| [05_Advanced_Topics.md](05_Advanced_Topics.md) | Transformers, CLIP/contrastive, detection, adaptation (prompts, LoRA, KD), SSMs, efficient attention |
| [06_Paper_Reading_Guide.md](06_Paper_Reading_Guide.md) | Nine-step workflow, reading passes, 18-point analysis template, contribution classification |
| [lessons/](lessons/) | One file per lesson: [Lesson_01](lessons/Lesson_01.md) |
| [exercises/](exercises/) | [Exercise_00_Diagnostic](exercises/Exercise_00_Diagnostic.md), [Exercise_01](exercises/Exercise_01.md), reference solutions in `exercises/solutions/` |
| [PROGRESS.md](PROGRESS.md) | Evidence-based status of lessons, concepts, exercises and papers |

Later, `papers/R##_Name.md` holds per-paper analysis notes (created when the first paper is studied).

## Stages (summary; details in [01](01_Learning_Roadmap.md))

0. **Foundations:** softmax/CE and gradients, backprop, optimisation, methodology, CNNs
1. **Detection fundamentals:** Fast R-CNN, YOLOv3
2. **Attention and query heads:** Transformer, SBERT, Query2Label, C-Tran, ML-Decoder
3. **Vision-language and prompt learning:** CLIP/ViT (external), CoOp, SPG, ATPrompt, TP-CLIP, CLIP-RC, BLIP, BEiT-3
4. **Real-time detection:** YOLOv4/7/9, DETR (external), RT-DETR, YOLOv10/12
5. **Open-vocabulary detection:** UniDetector, YOLO-World, LLMDet
6. **State-space models:** Mamba, Vim, VMamba, MambaVision, MambaOut, Mamba-YOLO-World (+ NSA)
7. **Specialised topics on demand:** distillation, point clouds, agents/RL
8. **Research skills** (throughout): critiques, a reproduction project, a research proposal

## Study workflow

```
diagnostic → lesson → exercise → readiness check → paper (06 workflow) → critique → PROGRESS update
                ↑___________________ gaps found ____________________________|
```

- **Lessons** teach a prerequisite with intuition → math → worked numbers → code → misconceptions.
- **Exercises** are where mastery is *demonstrated*. Status changes in PROGRESS only come from them (or from your answers and derivations).
- **Papers** are read only after their Essential prerequisites pass a readiness check.
- You advance when the "Criteria for moving forward" at the end of each lesson are met, not when the lesson is read.

## Resuming in a future CLI session

Start Claude Code in `E:\Papers` and say something like *"resume my ML course"*. The assistant should:

1. Read this `README.md` and [PROGRESS.md](PROGRESS.md).
2. Review [01_Learning_Roadmap.md](01_Learning_Roadmap.md) and the current lesson/exercise files.
3. Check `E:\Papers` (including subfolders) for new PDFs and add them to [00_Paper_Inventory.md](00_Paper_Inventory.md).
4. Continue from **"Next recommended activity"** in PROGRESS.
5. Not restart from the beginning unless you ask, or PROGRESS shows a prerequisite must be revisited.

The Markdown files are the course record. Conversation history is not relied on.

## Conventions

- 🟥 Essential · 🟨 Helpful · 🟦 Advanced.
- Paper IDs `R1`–`R40` follow the reading order in [00](00_Paper_Inventory.md) and [01](01_Learning_Roadmap.md).
- Venues marked † come from general knowledge, not the PDF. Verify them before citing.
- Math uses LaTeX (`$…$`). Use a Markdown viewer with math support (VS Code, Obsidian, GitHub).
