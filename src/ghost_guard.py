from __future__ import annotations

import argparse

import pandas as pd

from .common import add_config_arg, ensure_dirs, load_config, project_path
from .parse_outputs import answer_for_eval


def decide(original_answer: str, crop_answer: str, masked_answer: str) -> dict[str, object]:
    original_answer = answer_for_eval(original_answer)
    crop_answer = answer_for_eval(crop_answer)
    masked_answer = answer_for_eval(masked_answer)

    if original_answer == "NO":
        final_answer = "NO"
        risk = "none"
    elif original_answer == "YES" and crop_answer == "YES" and masked_answer == "NO":
        final_answer = "YES"
        risk = "low"
    elif original_answer == "YES" and masked_answer == "YES":
        final_answer = "UNSURE"
        risk = "high_context_hallucination"
    elif original_answer == "YES" and crop_answer != "YES":
        final_answer = "UNSURE"
        risk = "weak_visual_evidence"
    else:
        final_answer = "UNSURE"
        risk = "ambiguous"

    ghost_flag = original_answer == "YES" and masked_answer == "YES"
    evidence_supported = crop_answer == "YES"
    return {
        "original_answer": original_answer,
        "crop_answer": crop_answer,
        "masked_answer": masked_answer,
        "ghost_flag": ghost_flag,
        "evidence_supported": evidence_supported,
        "final_answer": final_answer,
        "risk": risk,
        "guard_accept": final_answer == "YES",
        "guard_abstain": final_answer == "UNSURE",
    }


def build_guard_decisions(answers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id, group in answers.groupby("sample_id", sort=False):
        by_view = group.set_index("view")
        missing = {"original", "crop", "masked"} - set(by_view.index)
        if missing:
            print(f"[WARN] Skipping {sample_id}; missing views: {sorted(missing)}")
            continue
        first = group.iloc[0]
        decision = decide(
            by_view.loc["original", "answer"],
            by_view.loc["crop", "answer"],
            by_view.loc["masked", "answer"],
        )
        rows.append(
            {
                "sample_id": sample_id,
                "category_name": first["category_name"],
                **decision,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply GHOST-Guard to view-level answers.")
    add_config_arg(parser)
    args = parser.parse_args()
    load_config(args.config)

    answers_path = project_path("outputs", "results", "view_answers.csv")
    out_path = project_path("outputs", "results", "guard_decisions.csv")
    ensure_dirs(out_path.parent)
    if not answers_path.exists():
        raise SystemExit(f"[ERROR] Missing view answers: {answers_path}")
    decisions = build_guard_decisions(pd.read_csv(answers_path))
    decisions.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
