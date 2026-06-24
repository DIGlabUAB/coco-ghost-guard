from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from .common import add_config_arg, ensure_dirs, load_config, project_path, write_json


MetricFn = Callable[[pd.DataFrame], float]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metric_functions() -> dict[str, tuple[str, MetricFn]]:
    return {
        "original_yes_rate": ("Original YES Rate", lambda d: safe_div((d.original_answer == "YES").sum(), len(d))),
        "crop_yes_rate": ("Crop YES Rate", lambda d: safe_div((d.crop_answer == "YES").sum(), len(d))),
        "masked_yes_rate": ("Masked YES Rate", lambda d: safe_div((d.masked_answer == "YES").sum(), len(d))),
        "ghost_object_rate": (
            "Ghost Object Rate",
            lambda d: safe_div((d.ghost_flag == True).sum(), (d.original_answer == "YES").sum()),
        ),
        "context_persistence_index": (
            "Context Persistence Index",
            lambda d: safe_div(
                safe_div((d.masked_answer == "YES").sum(), len(d)),
                safe_div((d.original_answer == "YES").sum(), len(d)),
            ),
        ),
        "guard_acceptance_rate": ("GHOST-Guard Acceptance Rate", lambda d: safe_div((d.final_answer == "YES").sum(), len(d))),
        "guard_abstention_rate": ("GHOST-Guard Abstention Rate", lambda d: safe_div((d.final_answer == "UNSURE").sum(), len(d))),
        "prevented_ghost_claim_rate": (
            "Prevented Ghost Claim Rate",
            lambda d: safe_div(((d.ghost_flag == True) & (d.final_answer == "UNSURE")).sum(), (d.ghost_flag == True).sum()),
        ),
    }


def compute_metrics(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for key, (label, fn) in metric_functions().items():
        rows.append({"metric": key, "label": label, "value": fn(df), "n": len(df)})
    return rows


def bootstrap_ci(df: pd.DataFrame, iters: int, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    funcs = metric_functions()
    values = {key: [] for key in funcs}
    for _ in range(iters):
        idx = rng.integers(0, len(df), len(df))
        sample = df.iloc[idx]
        for key, (_, fn) in funcs.items():
            values[key].append(fn(sample))
    rows = []
    base = {row["metric"]: row for row in compute_metrics(df)}
    for key, vals in values.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        row = dict(base[key])
        row.update({"ci_low": float(lo), "ci_high": float(hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("category_name", sort=True):
        for row in compute_metrics(group):
            row = dict(row)
            row["category_name"] = category
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute COCO-Ghost summary metrics.")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_path = project_path("outputs", "results", "guard_decisions.csv")
    out_dir = project_path("outputs", "results")
    ensure_dirs(out_dir)
    if not in_path.exists():
        raise SystemExit(f"[ERROR] Missing guard decisions: {in_path}")
    df = pd.read_csv(in_path)
    summary = pd.DataFrame(compute_metrics(df))
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    write_json(out_dir / "summary_metrics.json", summary.to_dict("records"))
    category_metrics(df).to_csv(out_dir / "category_metrics.csv", index=False)
    bootstrap_ci(df, int(cfg["evaluation"].get("bootstrap_iters", 1000)), int(cfg["dataset"].get("seed", 42))).to_csv(
        out_dir / "summary_metrics_with_ci.csv", index=False
    )
    print(f"[OK] Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
