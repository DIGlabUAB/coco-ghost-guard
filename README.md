# COCO-Ghost + GHOST-Guard

COCO-Ghost is a local, training-free research prototype for measuring context-induced object hallucination in vision-language models. It uses COCO instance segmentation masks to create object-removed counterfactual images, asks a local Ollama vision model whether the removed object is still visible, then applies GHOST-Guard: a black-box evidence gate that accepts an object claim only when it is supported by the object crop and disappears after object removal.

The core question is deliberately simple:

> If the visual evidence for an object is removed, does the model still claim the object is there?

Best framing:

> We present COCO-Ghost, a counterfactual object-removal benchmark for measuring context-induced object hallucination in VLMs, and GHOST-Guard, a training-free evidence gate that reduces unsupported object claims by requiring agreement between original-image, crop-level, and object-removed views.

This project does not claim to solve VLM hallucination. It targets a specific, interpretable subtype: context-induced persistence of removed objects.

## Current Result

Expanded local experiment: 50 COCO instances, 10 categories, 3 image views per instance, 3 Ollama VLMs.

| Model | Samples | Original YES | Crop YES | Masked YES | Ghost Object Rate | Guard Accept | Guard Abstain | Prevented Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llava:7b` | 50 | 52.0% | 74.0% | 32.0% | 61.5% | 18.0% | 34.0% | 100.0% |
| `qwen2.5vl:3b` | 50 | 92.0% | 86.0% | 74.0% | 78.3% | 14.0% | 78.0% | 100.0% |
| `qwen2.5vl:7b` | 50 | 100.0% | 96.0% | 84.0% | 84.0% | 16.0% | 84.0% | 100.0% |

See [RESULTS.md](RESULTS.md) and `outputs/comparison/multi_model_summary.md` for the generated experiment summary.

## What It Builds

- Automatic COCO annotation download and small-subset image selection.
- Counterfactual object removal using COCO segmentation masks.
- Local Ollama vision-language model inference.
- Baseline hallucination metrics.
- GHOST-Guard decision metrics.
- Publication-style figures and example panels.
- A markdown pilot report with novelty, significance, limitations, and next steps.

## Setup

Install Ollama:

```bash
brew install ollama
```

Or download it from the Ollama website if Homebrew is not available.

Start Ollama:

```bash
ollama serve
```

In another terminal, pull the default vision model:

```bash
ollama pull qwen2.5vl:7b
```

Optional fallbacks:

```bash
ollama pull moondream
ollama pull minicpm-v
ollama pull llama3.2-vision
ollama pull qwen2.5vl:3b
ollama pull llava:7b
```

Check local models:

```bash
ollama list
```

Check the Ollama API:

```bash
curl http://localhost:11434/api/tags
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## One-Command Run

```bash
bash scripts/run_fast_pilot.sh
```

The fast pilot defaults to 6 object categories, 3 images per category, and about 54 VLM calls.

For the expanded pilot:

```bash
bash scripts/run_expanded_pilot.sh
```

For the multi-model expanded suite:

```bash
bash scripts/run_expanded_model_suite.sh
```

## Manual Pipeline

```bash
python -m src.setup_check --config config/default.yaml
python -m src.download_coco_subset --config config/default.yaml
python -m src.make_counterfactuals --config config/default.yaml
python -m src.run_experiment --config config/default.yaml
python -m src.ghost_guard --config config/default.yaml
python -m src.evaluate --config config/default.yaml
python -m src.plot_results --config config/default.yaml
python -m src.make_report --config config/default.yaml
```

## Expected Outputs

- `outputs/results/view_answers.csv`
- `outputs/results/raw_vlm_outputs.jsonl`
- `outputs/results/guard_decisions.csv`
- `outputs/results/summary_metrics.csv`
- `outputs/results/summary_metrics.json`
- `outputs/results/summary_metrics_with_ci.csv`
- `outputs/results/category_metrics.csv`
- `outputs/figures/ghost_rate_by_category.png`
- `outputs/figures/original_vs_masked_yes_rate.png`
- `outputs/figures/guard_decision_breakdown.png`
- `outputs/figures/example_contact_sheet.png`
- `outputs/reports/pilot_report.md`
- `outputs/runs/<run_name>/`
- `outputs/comparison/model_summary_metrics.csv`
- `outputs/comparison/model_metric_comparison.png`
- `outputs/comparison/category_ghost_rate_comparison.png`
- `outputs/comparison/multi_model_summary.md`

## Metrics

- Original YES Rate: how often the model sees the object before removal.
- Crop YES Rate: how often the model recognizes the object when context is minimized.
- Ghost Object Rate: among original YES cases, how often the model still says YES after object removal.
- Context Persistence Index: masked YES rate divided by original YES rate.
- GHOST-Guard Acceptance Rate: how often the guard accepts the object claim.
- GHOST-Guard Abstention Rate: how often the guard refuses unsupported or suspicious claims.
- Prevented Ghost Claim Rate: among ghost claims, how many were downgraded to `UNSURE`.

## Troubleshooting

Ollama not running:

```text
Connection refused
```

Fix:

```bash
ollama serve
```

Model not found:

```bash
ollama pull qwen2.5vl:7b
```

Model too slow:

```bash
ollama pull moondream
```

Then edit `config/default.yaml`:

```yaml
ollama:
  model: "moondream"
```

JSON parse failures are expected sometimes. Raw responses are preserved, and parse failures are treated as `UNSURE` during evaluation.

COCO download slow:

```yaml
dataset:
  max_images_per_category: 1
  categories:
    - "tennis racket"
    - "fork"
```

## Research Positioning

CHAIR measures hallucinated objects in captions. POPE probes object presence in real images. AMBER evaluates broader hallucination types. Visual Contrastive Decoding, OPERA, Woodpecker, and newer object-aligned contrastive methods generally rely on decoding internals, auxiliary tools, or broader correction pipelines.

COCO-Ghost is narrower and more mechanistic: it asks whether an object claim survives controlled visual removal of the target object while scene context remains. GHOST-Guard is a decision-level safety wrapper for black-box local VLMs, not a claim of model-internal improvement.

## Acceptance Criteria

The project is complete when:

- `bash scripts/run_fast_pilot.sh` runs end-to-end.
- At least 10 samples are processed successfully.
- `outputs/results/view_answers.csv` exists.
- `outputs/results/guard_decisions.csv` exists.
- `outputs/results/summary_metrics.csv` exists.
- At least 3 figures are created.
- `outputs/reports/pilot_report.md` exists.
- The report includes Ghost Object Rate, category-level metrics, GHOST-Guard decision breakdown, novelty vs SOTA, limitations, and next steps.

## Good Result Patterns

A good result is not necessarily high accuracy. It may be:

- Strong finding: the model says YES on original images and YES on many masked images.
- Strong mitigation: GHOST-Guard downgrades most masked-image YES cases to `UNSURE`.
- Interesting negative finding: the model rarely says YES after masking.
- Model comparison finding: one Ollama model has lower Ghost Object Rate than another.
