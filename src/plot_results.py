from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from .common import add_config_arg, ensure_dirs, load_config, project_path


def pct(values: pd.Series) -> pd.Series:
    total = values.sum()
    return values / total if total else values


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#2f6f73") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values, default=0) * 1.15))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ghost_rate_by_category(decisions: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for category, group in decisions.groupby("category_name", sort=True):
        original_yes = (group.original_answer == "YES").sum()
        rate = ((group.ghost_flag == True).sum() / original_yes) if original_yes else 0.0
        rows.append((category, rate, len(group)))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color="#6b8f71")
    ax.set_title("Ghost Object Rate by Category")
    ax.set_ylabel("Ghost Object Rate")
    ax.set_ylim(0, max(1.0, max(values, default=0) * 1.2))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, value, n) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ghost_rate_by_category.png", dpi=180)
    plt.close(fig)


def original_vs_masked(decisions: pd.DataFrame, out_dir: Path) -> None:
    n = len(decisions)
    values = [
        (decisions.original_answer == "YES").sum() / n if n else 0.0,
        (decisions.crop_answer == "YES").sum() / n if n else 0.0,
        (decisions.masked_answer == "YES").sum() / n if n else 0.0,
    ]
    save_bar(
        out_dir / "original_vs_masked_yes_rate.png",
        ["Original", "Crop", "Masked"],
        values,
        "Original, Crop, and Masked YES Rates",
        "YES Rate",
        "#4b7ea8",
    )


def decision_breakdown(decisions: pd.DataFrame, out_dir: Path) -> None:
    labels = ["Accepted YES", "Ghost risk", "Weak crop evidence", "NO", "Other UNSURE"]
    n = len(decisions)
    values = [
        ((decisions.final_answer == "YES").sum() / n) if n else 0.0,
        (((decisions.risk == "high_context_hallucination") & (decisions.final_answer == "UNSURE")).sum() / n) if n else 0.0,
        (((decisions.risk == "weak_visual_evidence") & (decisions.final_answer == "UNSURE")).sum() / n) if n else 0.0,
        ((decisions.final_answer == "NO").sum() / n) if n else 0.0,
        (((decisions.final_answer == "UNSURE") & (~decisions.risk.isin(["high_context_hallucination", "weak_visual_evidence"]))).sum() / n)
        if n
        else 0.0,
    ]
    save_bar(out_dir / "guard_decision_breakdown.png", labels, values, "GHOST-Guard Decision Breakdown", "Proportion", "#8a6f3d")


def load_answers() -> pd.DataFrame:
    path = project_path("outputs", "results", "view_answers.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_contact_sheet(decisions: pd.DataFrame, out_dir: Path) -> None:
    answers = load_answers()
    manifest_path = project_path("outputs", "counterfactuals", "manifest.jsonl")
    if answers.empty or not manifest_path.exists():
        return
    manifest = pd.read_json(manifest_path, lines=True).set_index("sample_id")
    priority = decisions.sort_values(["ghost_flag", "guard_abstain"], ascending=False).head(6)
    if priority.empty:
        return

    thumb_w, thumb_h = 210, 160
    label_h = 70
    cols = 3
    rows = len(priority)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(priority.to_dict("records")):
        sample_id = row["sample_id"]
        if sample_id not in manifest.index:
            continue
        sample = manifest.loc[sample_id]
        by_view = answers[answers.sample_id == sample_id].set_index("view")
        for c, view in enumerate(["original", "crop", "masked"]):
            img = Image.open(sample[f"{view}_path"]).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            x = c * thumb_w + (thumb_w - img.width) // 2
            y = r * (thumb_h + label_h)
            sheet.paste(img, (x, y))
            ans = by_view.loc[view, "answer"] if view in by_view.index else "NA"
            label = f"{view}: {ans}"
            draw.text((c * thumb_w + 8, y + thumb_h + 6), label, fill=(20, 20, 20))
        final_label = f"{row['category_name']} | guard: {row['final_answer']} | risk: {row['risk']}"
        draw.text((8, r * (thumb_h + label_h) + thumb_h + 28), textwrap.shorten(final_label, width=95), fill=(20, 20, 20))
    sheet.save(out_dir / "example_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COCO-Ghost figures.")
    add_config_arg(parser)
    args = parser.parse_args()
    load_config(args.config)
    out_dir = project_path("outputs", "figures")
    ensure_dirs(out_dir)
    decisions_path = project_path("outputs", "results", "guard_decisions.csv")
    if not decisions_path.exists():
        raise SystemExit(f"[ERROR] Missing guard decisions: {decisions_path}")
    decisions = pd.read_csv(decisions_path)
    ghost_rate_by_category(decisions, out_dir)
    original_vs_masked(decisions, out_dir)
    decision_breakdown(decisions, out_dir)
    make_contact_sheet(decisions, out_dir)
    print(f"[OK] Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
