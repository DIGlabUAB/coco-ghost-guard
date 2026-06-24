# COCO-Ghost + GHOST-Guard

COCO-Ghost is a local, training-free benchmark for measuring context-induced object hallucination in vision-language models. It creates object-removed counterfactual images from COCO annotations, asks local Ollama VLMs whether the removed object is still visible, then applies GHOST-Guard: a black-box evidence gate that accepts an object claim only when it is supported by the original image and crop, and disappears after object removal.

The core question is deliberately simple:

> If the visual evidence for an object is removed, does the model still claim the object is there?

For scaled experiments, the default intervention is a conservative padded bounding-box occlusion with solid local-mean fill. This is less naturalistic than inpainting, but it is stricter for evaluation because it avoids object-shaped blur artifacts.

## Current Scaled Result

200 COCO validation instances, 20 object categories, 3 image views per instance, 3 local Ollama VLMs. All 200 counterfactual removals were labeled `clean_removal` by the artifact-labeling pass.

| Model | Samples | Original YES | Crop YES | Masked YES | Clean Ghost Rate | Guard Accept | Guard Abstain | Accepted Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5vl:7b` | 200 | 96.0% | 94.5% | 49.5% | 51.6% | 44.5% | 51.5% | 0.0% |
| `qwen2.5vl:3b` | 200 | 88.5% | 80.0% | 43.0% | 47.5% | 30.0% | 58.5% | 0.0% |
| `llava:7b` | 200 | 43.0% | 60.5% | 14.5% | 33.7% | 24.5% | 18.5% | 0.0% |

The headline is not that the guard improves the model. It abstains on a measurable failure mode: object-presence claims that survive removal of the object evidence.

See [RESULTS.md](RESULTS.md), [ARTIFACT_AUDIT.md](ARTIFACT_AUDIT.md), and generated outputs under `outputs/comparison/`.

## What It Builds

- COCO annotation download and reproducible subset selection.
- Counterfactual object removal using COCO segmentation or padded bounding-box occlusion.
- Local Ollama VLM inference over original, crop, and masked views.
- Artifact/removal-quality labeling.
- Clean-removal reporting separate from artifact-prone cases.
- Baseline comparisons against original-only and crop-gated acceptance.
- Publication-style figures, reports, and archived per-model runs.

## Setup

Install Ollama:

```bash
brew install ollama
```

Start Ollama:

```bash
ollama serve
```

Pull the local vision models used in the scaled runs:

```bash
ollama pull qwen2.5vl:7b
ollama pull qwen2.5vl:3b
ollama pull llava:7b
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the setup:

```bash
python -m src.setup_check --config config/large_200_qwen2_5vl_7b.yaml
```

## Reproduce the Scaled Suite

The large suite uses the shared 200-sample bbox-solid counterfactual set:

```bash
bash scripts/run_large_200_model_suite.sh
```

Manual pipeline for one model:

```bash
python -m src.download_coco_subset --config config/large_200_qwen2_5vl_7b.yaml
python -m src.make_counterfactuals --config config/large_200_qwen2_5vl_7b.yaml
python -m src.label_artifacts --config config/large_200_qwen2_5vl_7b.yaml --out outputs/artifact_labels_large200_bbox_solid.csv
python -m src.run_experiment --config config/large_200_qwen2_5vl_7b.yaml
python -m src.ghost_guard --config config/large_200_qwen2_5vl_7b.yaml
python -m src.evaluate --config config/large_200_qwen2_5vl_7b.yaml
python -m src.plot_results --config config/large_200_qwen2_5vl_7b.yaml
python -m src.make_report --config config/large_200_qwen2_5vl_7b.yaml
python -m src.archive_run --config config/large_200_qwen2_5vl_7b.yaml --run-name large200_bboxsolid_qwen2_5vl_7b
```

Then compare archived runs:

```bash
python -m src.compare_runs --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b
python -m src.baseline_analysis --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b
python -m src.clean_removal_report --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b --labels outputs/artifact_labels_large200_bbox_solid.csv
```

## Expected Outputs

- `outputs/results/view_answers.csv`
- `outputs/results/raw_vlm_outputs.jsonl`
- `outputs/results/guard_decisions.csv`
- `outputs/results/summary_metrics.csv`
- `outputs/results/category_metrics.csv`
- `outputs/results/summary_metrics_with_ci.csv`
- `outputs/runs/<run_name>/`
- `outputs/comparison/model_summary_metrics.csv`
- `outputs/comparison/model_metric_comparison.png`
- `outputs/comparison/category_ghost_rate_comparison.png`
- `outputs/comparison/clean_removal_metrics.md`
- `outputs/comparison/baseline_safety_comparison.md`
- `outputs/comparison/multi_model_summary.md`

## Metrics

- Original YES Rate: how often the model reports the object before removal.
- Crop YES Rate: how often the model recognizes the object when context is minimized.
- Masked YES Rate: how often the model reports the object after removal.
- Ghost Object Rate: among original YES cases, how often the model still says YES after removal.
- Context Persistence Index: masked YES rate divided by original YES rate.
- GHOST-Guard Acceptance Rate: how often the guard accepts the object claim.
- GHOST-Guard Abstention Rate: how often the guard refuses unsupported or suspicious claims.
- Prevented Ghost Claim Rate: among detected ghost claims, how many were downgraded to `UNSURE`.

## Research Positioning

CHAIR measures hallucinated objects in captions. POPE probes object presence in real images. AMBER evaluates broader hallucination types. Decoding and correction methods such as VCD, OPERA, and Woodpecker rely on model internals, auxiliary tools, or correction pipelines.

COCO-Ghost is narrower and more diagnostic: it asks whether an object-presence claim survives controlled removal of the target object while surrounding scene context remains. GHOST-Guard is a decision-level abstention wrapper for black-box VLMs, not a model-internal fix.

## Important Caveat

Early segmentation blur counterfactuals produced high ghost rates, but some blur masks visibly preserved target-object shape. Those runs are useful as an ablation and as motivation for the artifact audit, not as the headline claim.

The scaled results above use padded bbox occlusion with solid local-mean fill and report all metrics on `clean_removal` cases.
