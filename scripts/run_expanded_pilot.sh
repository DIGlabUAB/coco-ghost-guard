#!/usr/bin/env bash
set -e

source .venv/bin/activate

python -m src.setup_check --config config/expanded.yaml
python -m src.download_coco_subset --config config/expanded.yaml
python -m src.make_counterfactuals --config config/expanded.yaml
python -m src.run_experiment --config config/expanded.yaml
python -m src.ghost_guard --config config/expanded.yaml
python -m src.evaluate --config config/expanded.yaml
python -m src.plot_results --config config/expanded.yaml
python -m src.make_report --config config/expanded.yaml

echo "Done. See outputs/reports/pilot_report.md"
