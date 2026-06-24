#!/usr/bin/env bash
set -e

source .venv/bin/activate

python -m src.download_coco_subset --config config/expanded_qwen2_5vl_7b.yaml
python -m src.make_counterfactuals --config config/expanded_qwen2_5vl_7b.yaml

for item in \
  "qwen2_5vl_7b config/expanded_qwen2_5vl_7b.yaml" \
  "qwen2_5vl_3b config/expanded_qwen2_5vl_3b.yaml" \
  "llava_7b config/expanded_llava_7b.yaml"
do
  set -- $item
  run_name="$1"
  config="$2"
  python -m src.setup_check --config "$config"
  python -m src.run_experiment --config "$config"
  python -m src.ghost_guard --config "$config"
  python -m src.evaluate --config "$config"
  python -m src.plot_results --config "$config"
  python -m src.make_report --config "$config"
  python -m src.archive_run --config "$config" --run-name "$run_name"
done

python -m src.compare_runs --runs qwen2_5vl_7b qwen2_5vl_3b llava_7b

echo "Done. See outputs/comparison/multi_model_summary.md"
