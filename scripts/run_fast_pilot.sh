#!/usr/bin/env bash
set -e

source .venv/bin/activate

python -m src.setup_check --config config/default.yaml
python -m src.download_coco_subset --config config/default.yaml
python -m src.make_counterfactuals --config config/default.yaml
python -m src.run_experiment --config config/default.yaml
python -m src.ghost_guard --config config/default.yaml
python -m src.evaluate --config config/default.yaml
python -m src.plot_results --config config/default.yaml
python -m src.make_report --config config/default.yaml

echo "Done. See outputs/reports/pilot_report.md"
