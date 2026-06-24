#!/usr/bin/env bash
set -e

source .venv/bin/activate

python -m src.download_coco_subset --config config/large_200_qwen2_5vl_7b.yaml
python -m src.make_counterfactuals --config config/large_200_qwen2_5vl_7b.yaml
python -m src.label_artifacts --config config/large_200_qwen2_5vl_7b.yaml --out outputs/artifact_labels_large200_bbox_solid.csv

for item in \
  "large200_bboxsolid_qwen2_5vl_7b config/large_200_qwen2_5vl_7b.yaml" \
  "large200_bboxsolid_qwen2_5vl_3b config/large_200_qwen2_5vl_3b.yaml" \
  "large200_bboxsolid_llava_7b config/large_200_llava_7b.yaml"
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

python -m src.compare_runs --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b
python -m src.baseline_analysis --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b
python -m src.clean_removal_report --runs large200_bboxsolid_qwen2_5vl_7b large200_bboxsolid_qwen2_5vl_3b large200_bboxsolid_llava_7b --labels outputs/artifact_labels_large200_bbox_solid.csv

echo "Done. See outputs/comparison/clean_removal_metrics.md"
