# 06 — Paper Reading Guide

> Back to [README](README.md) · Inventory: [00](00_Paper_Inventory.md) · Roadmap: [01](01_Learning_Roadmap.md)
>
> A repeatable method for reading, reconstructing, implementing and critiquing a paper. Use it for every paper in the roadmap. Save each paper's notes as `papers/R##_ShortName.md` (that folder is created when the first paper is studied).

## Part A: The nine-step workflow (how we study each paper together)

| Step | What happens | Output |
|---|---|---|
| 1. Requirements | I list the Essential / Helpful / Advanced prerequisites (from [00](00_Paper_Inventory.md)) | prerequisite list |
| 2. Diagnose | 3–6 targeted questions on the Essential items only | your answers |
| 3. Teach gaps | short lesson(s) only for what you couldn't demonstrate | lesson file(s) |
| 4. Readiness check | the "Ready when…" test from [01 §4](01_Learning_Roadmap.md#4-recommended-reading-order) | pass / retry |
| 5. Strategic read | the passes in Part B | annotated notes |
| 6. Reconstruct | explain the method without the paper open; re-derive key equations; draw the architecture with tensor shapes | template §6–9 filled in |
| 7. Implement | the smallest experiment that exercises the core idea | `exercises/` entry |
| 8. Critique | the template §13–17 plus the checklist in [03 §4](03_Machine_Learning.md#4-experimental-design-checklist-applied-to-every-paper-in-stage-8-critiques) | critique notes |
| 9. Record | update [PROGRESS](PROGRESS.md) and the roadmap change log if needed | updated files |

## Part B: Reading passes

**Pass 1 (15 min): map.** Title → abstract → figures and captions → section headings → conclusion → the main results table. Write *one sentence* for the problem and *one sentence* for the claimed insight. Decide whether to read Full, Core or Selected.

**Pass 2 (1–2 h): method.** Read the method section with a pen.
- For every equation: name each symbol, give its **shape**, and say whether it is a parameter, an input or an intermediate value.
- For every figure: trace one input through it, writing tensor shapes at each arrow.
- Mark every sentence that says "we found", "empirically" or "we observe". These are claims that need evidence.

**Pass 3 (1 h): evidence.** Experiments, ablations, implementation details, appendix.
- Which table supports which claim? Is there a claim with no table?
- Are the baselines fair (same backbone, data, schedule, resolution, pre-training)?
- How big are the gains compared with plausible seed variance?

**Pass 4 (after implementing): critique.** Re-read the intro and the limitations with your implementation experience in mind.

### Section-by-section: what deserves attention

| Section | Why it matters | Read for |
|---|---|---|
| Abstract | the claims the authors want cited | the list of claims to verify |
| Introduction | the authors' framing of the gap | is the gap real, or constructed to fit the method? |
| Related work | the lineage | which 2–3 papers this one is a *delta* on (read those deltas, not everything) |
| Method | the actual contribution | equations, shapes, what is trained vs. frozen |
| Experiments setup | where results become (in)comparable | data, pre-training, schedules, metrics |
| Main tables | headline evidence | compare like with like (same backbone and data columns) |
| Ablations | which components matter | does each ablation change one thing? |
| Limitations | often the most honest paragraph | research opportunities |
| Appendix | where the real hyper-parameters live | reproduction details |

## Part C: Standard paper analysis template

Copy this into `papers/R##_ShortName.md`.

```markdown
# R## — <Title> (<Year>, <Venue>)
Inspection: Full / Core / Selected (<sections>) · Date: YYYY-MM-DD

1. Research context: what problem?
2. Motivation: why does it matter?
3. Prior approaches: what existed (the 2–3 direct predecessors)?
4. Research gap: what remained unsolved, *according to the authors*? Do I agree?
5. Main contribution: what is new? (Classify: new idea / new combination / engineering / new data / new analysis.)
6. Prerequisites: what I needed first (E/H/A), and which ones I had to learn.
7. Core idea: the central insight in ≤ 3 sentences, without jargon.
8. Architecture: components and data flow, with tensor shapes.
9. Mathematics: each key equation restated in my own notation, with symbol shapes.
10. Training: losses, what's frozen, optimiser, schedule, data.
11. Data: datasets, splits, assumptions, possible leakage/overlap with pre-training.
12. Evaluation: metrics and why they fit (or don't).
13. Results: what the evidence shows, and what it doesn't.
14. Ablations: which components matter, and by how much compared with noise?
15. Limitations: stated by the authors, plus the ones I found.
16. Reproducibility: code? compute? missing details?
17. Research opportunities: open questions. For each: is it meaningful, testable, differentiated?
18. Comprehension check: explain the method in 5 minutes without the paper. (Record: pass / partial / fail.)
```

## Part D: Classifying a contribution

Ask in order:
1. **Does removing it change the result substantially** (per the ablation)? If there is no ablation, the claim is unsupported.
2. **Is it a new principle or a new instance of a known principle?** E.g. YOLOv10's one-to-one head applies DETR-style one-to-one matching to YOLO, which makes it an *adaptation*, valuable mainly through its consistency analysis.
3. **Could the gain come from extra data, compute or tuning?** (LLMDet's new dataset, YOLOv4's large trick search.)
4. **Is the evaluation the one the claim needs?** (A "generalisation" claim needs novel classes or domains, not just base-class accuracy.)

## Part E: From critique to research idea

A limitation becomes a candidate research direction only if:
- it is **real** (you can reproduce or measure it),
- it is **important** (it affects a metric or use case people care about),
- it is **testable** with your compute,
- it is **not already solved** (search for follow-ups: who cites this paper and mentions the limitation?),
- you can state a **falsifiable hypothesis** plus a controlled experiment that would refute it.

Most limitations fail at least one test. That is expected, and recording *why* they fail is itself research training.
