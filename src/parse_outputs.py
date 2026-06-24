from __future__ import annotations

import json
import re
from typing import Any, Optional


ANSWERS = {"YES", "NO", "UNSURE"}


def _strip_fence(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    text = _strip_fence(text)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_presence_response(raw: str) -> dict[str, Any]:
    raw = raw or ""
    obj = _extract_json_object(raw)
    if obj:
        answer = str(obj.get("answer", "")).upper().strip()
        if answer in ANSWERS:
            return {
                "answer": answer,
                "evidence": str(obj.get("evidence", "")).strip(),
                "parse_ok": True,
                "raw": raw,
            }

    match = re.search(r"\b(YES|NO|UNSURE)\b", raw, flags=re.IGNORECASE)
    if match:
        return {
            "answer": match.group(1).upper(),
            "evidence": raw.strip()[:300],
            "parse_ok": True,
            "raw": raw,
        }

    return {"answer": "PARSE_FAIL", "evidence": "", "parse_ok": False, "raw": raw}


def answer_for_eval(answer: str) -> str:
    answer = str(answer).upper()
    return "UNSURE" if answer == "PARSE_FAIL" else answer
