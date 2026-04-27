#!/usr/bin/env python3
"""Execute final validation, bounded repair, and release compilation.

이 스크립트는 counted snapshot을 최종 제출 후보로 바로 승격하지 않는다.
먼저 split-lock disposition을 적용하고, strict final Judge API를 실행한 뒤,
수정 가능한 row만 bounded repair/rejudge로 보정하고, 통과 row만
``final_dataset`` release로 컴파일한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_DIR = REPO_ROOT / "scripts/aihub/problem_generation/shared"
sys.path.insert(0, str(SHARED_DIR))

from production_batch_common import call_gemini_json, call_openai_json  # noqa: E402


PROCESSED_ROOT = REPO_ROOT / "data/processed/aihub/problem_generation"
DATASET_CANDIDATES = PROCESSED_ROOT / "dataset_candidates"
FINAL_DATASET_ROOT = PROCESSED_ROOT / "final_dataset"
SPLITS = ("train", "dev", "test")


JUDGE_MODEL_CANDIDATES = tuple(
    item.strip()
    for item in os.environ.get("FINAL_VALIDATION_JUDGE_MODELS", "gemini-2.5-pro").split(",")
    if item.strip()
)
REPAIR_MODEL_CANDIDATES = tuple(
    item.strip()
    for item in os.environ.get("FINAL_VALIDATION_REPAIR_MODELS", "gpt-5.4").split(",")
    if item.strip()
)
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("FINAL_VALIDATION_JUDGE_TIMEOUT_SECONDS", "240"))
REPAIR_TIMEOUT_SECONDS = int(os.environ.get("FINAL_VALIDATION_REPAIR_TIMEOUT_SECONDS", "240"))
JUDGE_TEMPERATURE = float(os.environ.get("FINAL_VALIDATION_JUDGE_TEMPERATURE", "0"))
REPAIR_TEMPERATURE = float(os.environ.get("FINAL_VALIDATION_REPAIR_TEMPERATURE", "0.2"))
REPAIR_MAX_TOKENS = int(os.environ.get("FINAL_VALIDATION_REPAIR_MAX_TOKENS", "3200"))


OBJECTIVE_ERROR_TAGS = {
    "근거불일치",
    "정답비유일",
    "정답없음",
    "오답이정답가능",
    "오답약함",
    "복수쟁점혼합",
    "발문어색함",
    "정답누설",
    "해설부족",
    "메타데이터충돌",
}
DESCRIPTIVE_ERROR_TAGS = {
    "근거불일치",
    "핵심요건누락",
    "채점기준불명확",
    "복수과제혼합",
    "해설부족",
    "문장어색함",
    "환각",
    "메타데이터충돌",
}
NO_REPAIR_ACTIONS = {
    "",
    "없음",
    "none",
    "no repair",
    "no_repair",
    "no repair needed",
    "해당 없음",
    "불필요",
}
OBJECTIVE_RELEASE_SCORE_KEYS = (
    "source_grounding",
    "answer_blind_keyedness",
    "single_best_answer",
    "distractor_refutation",
    "metadata_consistency",
    "stem_naturalness",
    "explanation_completeness",
)
DESCRIPTIVE_RELEASE_SCORE_KEYS = (
    "source_grounding",
    "answer_completeness",
    "question_clarity",
    "evaluation_feasibility",
    "explanation_quality",
    "korean_naturalness",
    "no_hallucination",
)
UNREPAIRABLE_TAGS = {"근거불일치", "환각", "정답없음"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalized_action(action: Any) -> str:
    return re.sub(r"\s+", " ", str(action or "")).strip().lower()


def parse_judge_payload(judge_row: dict[str, Any]) -> dict[str, Any]:
    raw = judge_row.get("judge_json", "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def coerce_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def judge_scores(judge_row: dict[str, Any]) -> dict[str, float]:
    payload = parse_judge_payload(judge_row)
    scores = payload.get("scores", {})
    if not isinstance(scores, dict):
        return {}
    coerced: dict[str, float] = {}
    for key, value in scores.items():
        score = coerce_score(value)
        if score is not None:
            coerced[str(key)] = score
    naturalness = coerce_score(payload.get("naturalness_score"))
    if naturalness is not None and "stem_naturalness" not in coerced:
        coerced["stem_naturalness"] = naturalness
    return coerced


def low_score_reasons(task: str, judge_row: dict[str, Any]) -> list[str]:
    scores = judge_scores(judge_row)
    required_keys = OBJECTIVE_RELEASE_SCORE_KEYS if task == "objective" else DESCRIPTIVE_RELEASE_SCORE_KEYS
    reasons: list[str] = []
    for key in required_keys:
        score = scores.get(key)
        if score is None:
            reasons.append(f"{key}=missing")
        elif score < 4:
            reasons.append(f"{key}={score:g}")
    return reasons


def repair_action_required(judge_row: dict[str, Any]) -> bool:
    action = normalized_action(judge_row.get("repair_action", ""))
    return action not in NO_REPAIR_ACTIONS


def predicted_choice_mismatch(row: dict[str, Any], judge_row: dict[str, Any]) -> bool:
    if row.get("_task") != "objective":
        return False
    predicted = str(judge_row.get("predicted_correct_choice", "") or "").strip().upper()
    provided = str(judge_row.get("provided_correct_choice", row.get("correct_choice", "")) or "").strip().upper()
    return bool(predicted and provided and predicted != provided)


def source_span_reasons(row: dict[str, Any], judge_row: dict[str, Any]) -> list[str]:
    if row.get("_task") != "objective":
        return []
    payload = parse_judge_payload(judge_row)
    provided = str(judge_row.get("provided_correct_choice", row.get("correct_choice", "")) or "").strip().upper()
    reasons: list[str] = []
    if not str(payload.get("correct_support_span", "") or "").strip():
        reasons.append("correct_support_span=missing")
    # 정답 선택지 자체에는 refutation이 비어 있는 것이 정상이라 오답 선택지만 요구한다.
    for label in ("A", "B", "C", "D"):
        if label == provided:
            continue
        if not str(payload.get(f"distractor_{label}_refutation", "") or "").strip():
            reasons.append(f"distractor_{label}_refutation=missing")
    return reasons


def derived_release_gate(row: dict[str, Any], judge_row: dict[str, Any]) -> dict[str, Any]:
    """Judge pass를 그대로 믿지 않고 final release용 hard gate를 다시 계산한다."""
    if not judge_row:
        return {
            "derived_status": "manual_review",
            "derived_gate_reasons": "judge_missing",
            "repair_allowed": "아니오",
        }

    tags = set(filter(None, str(judge_row.get("hard_fail_tags", "")).split("|")))
    reasons: list[str] = []
    if judge_row.get("pass_or_fail") != "pass":
        reasons.append("pass_or_fail!=pass")
    if judge_row.get("critical_fail") not in {"false", "False", "0"}:
        reasons.append("critical_fail")
    if tags:
        reasons.append("hard_fail_tags=" + "|".join(sorted(tags)))
    if repair_action_required(judge_row):
        reasons.append("repair_action=" + str(judge_row.get("repair_action", "")))
    if predicted_choice_mismatch(row, judge_row):
        reasons.append("predicted_correct_choice_mismatch")
    reasons.extend(low_score_reasons(row.get("_task", ""), judge_row))
    reasons.extend(source_span_reasons(row, judge_row))

    if not reasons:
        return {
            "derived_status": "release_ready",
            "derived_gate_reasons": "",
            "repair_allowed": "아니오",
        }

    scores = judge_scores(judge_row)
    if tags & UNREPAIRABLE_TAGS or scores.get("source_grounding", 5) <= 2:
        status = "quarantine"
        repair_allowed = "아니오"
    elif predicted_choice_mismatch(row, judge_row):
        status = "manual_review"
        repair_allowed = "아니오"
    else:
        status = "repair_planned"
        repair_allowed = "예"
    return {
        "derived_status": status,
        "derived_gate_reasons": "; ".join(reasons),
        "repair_allowed": repair_allowed,
    }


def row_id(row: dict[str, Any], task: str, split: str, index: int) -> str:
    return row.get("problem_id") or row.get("candidate_id") or row.get("seed_sample_id") or f"{task}_{split}_{index:06d}"


def load_snapshot_rows(snapshot_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ("objective", "descriptive"):
        for split in SPLITS:
            for index, row in enumerate(read_jsonl(snapshot_root / task / f"{split}.jsonl"), start=1):
                item = dict(row)
                item["_task"] = task
                item["_split"] = split
                item["_row_id"] = row_id(row, task, split, index)
                rows.append(item)
    return rows


def load_quarantine_ids(snapshot_root: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_csv(snapshot_root / "manifests/split_lock_disposition_manifest.csv"):
        if row.get("split_lock_disposition") == "quarantine_before_final_release":
            ids.add(row.get("problem_id", ""))
    return ids


def clean_stem_text(text: str) -> tuple[str, list[str]]:
    original = text
    replacements = [
        (r"가장 타당한 것은 무엇인가 옳은 설명을 고르시오", "가장 타당한 설명을 고르시오"),
        (r"기준은 무엇인가 옳은 설명을 고르시오", "기준에 관한 옳은 설명을 고르시오"),
        (r"무엇인가 옳은 설명을 고르시오", "무엇인지에 관한 옳은 설명을 고르시오"),
        (r"무엇인가 를", "무엇인지를"),
        (r"기간은 얼마인가요의 범위", "기간의 범위"),
        (r"가장 옳은 것은\s*(.*?)\s*고르시오", r"\1에 관하여 가장 옳은 설명을 고르시오"),
    ]
    applied: list[str] = []
    repaired = text
    for pattern, replacement in replacements:
        new_repaired = re.sub(pattern, replacement, repaired)
        if new_repaired != repaired:
            applied.append(pattern)
            repaired = new_repaired
    repaired = re.sub(r"\s+", " ", repaired).strip()
    if repaired and not repaired.endswith((".", "?", "다", "오")):
        repaired += "."
    return repaired, applied if repaired != original else []


def apply_prefinal_repairs(rows: list[dict[str, Any]], quarantine_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired_rows: list[dict[str, Any]] = []
    revision_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        rid = row["_row_id"]
        if rid in quarantine_ids or row.get("problem_id") in quarantine_ids:
            row["_release_gate_status"] = "quarantine"
            row["_release_gate_reason"] = "split_lock_disposition"
            repaired_rows.append(row)
            continue

        if row["_task"] == "objective":
            repaired_stem, patterns = clean_stem_text(row.get("generated_stem", ""))
            if patterns:
                before = row.get("generated_stem", "")
                row["generated_stem"] = repaired_stem
                row["_prefinal_repair_applied"] = "예"
                row["_prefinal_repair_reason"] = "stem_hygiene"
                revision_rows.append(
                    {
                        "problem_id": rid,
                        "revision_id": f"{rid}::prefinal_stem_r1",
                        "parent_problem_id": rid,
                        "repair_stage": "prefinal_deterministic",
                        "repair_attempt": 1,
                        "repair_reason": "stem_hygiene",
                        "repair_action": "clean_stem_text",
                        "applied_patterns": "|".join(patterns),
                        "before_text": before,
                        "after_text": repaired_stem,
                        "validation_status": "pending_rejudge",
                    }
                )
        row.setdefault("_release_gate_status", "pending_validation")
        row.setdefault("_release_gate_reason", "")
        repaired_rows.append(row)
    return repaired_rows, revision_rows


def compact_row_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    if row["_task"] == "objective":
        return {
            "problem_id": row["_row_id"],
            "task": "objective",
            "doc_type_name": row.get("doc_type_name", ""),
            "split": row.get("_split", ""),
            "stem": row.get("generated_stem", ""),
            "choices": {
                "A": row.get("choice_a", ""),
                "B": row.get("choice_b", ""),
                "C": row.get("choice_c", ""),
                "D": row.get("choice_d", ""),
            },
            "provided_correct_choice": row.get("correct_choice", ""),
            "gold_short_answer": row.get("gold_short_answer", ""),
            "gold_reference_explanation": row.get("gold_reference_explanation", ""),
            "family_id": row.get("family_id", ""),
            "source_subset": row.get("source_subset", ""),
        }
    return {
        "problem_id": row["_row_id"],
        "task": "descriptive",
        "doc_type_name": row.get("doc_type_name", ""),
        "split": row.get("_split", ""),
        "question": row.get("generated_problem", ""),
        "gold_short_answer": row.get("gold_short_answer", ""),
        "gold_reference_explanation": row.get("gold_reference_explanation", ""),
        "focus_issue": row.get("focus_issue", ""),
        "family_id": row.get("family_id", ""),
        "source_subset": row.get("source_subset", ""),
    }


def build_judge_prompt(row: dict[str, Any]) -> str:
    payload = compact_row_for_prompt(row)
    if row["_task"] == "objective":
        criteria = """
Evaluate the item as a strict final certifier for Korean legal multiple-choice QA.
Do not grade leniently. If any critical gate fails, pass_or_fail must be "fail".
Check: Source Grounding, Answer-Blind Keyedness, Single-Best Answer, Distractor Refutation,
Near-Miss Quality, Single-Axis Stem, Korean Legal Exam Naturalness, Answer Leakage,
Explanation Completeness, Metadata Consistency.
Return JSON only with:
pass_or_fail, critical_fail, predicted_correct_choice, provided_correct_choice, axis,
scores(source_grounding, answer_blind_keyedness, single_best_answer, distractor_refutation,
near_miss_quality, stem_naturalness, explanation_completeness, metadata_consistency),
hard_fail_tags, repair_action, one_sentence_reason, correct_support_span,
distractor_A_refutation, distractor_B_refutation, distractor_C_refutation, distractor_D_refutation,
stem_axis, axis_consistency, naturalness_score.
Use Korean strings for reasons and tags.
Allowed hard_fail_tags: 근거불일치, 정답비유일, 정답없음, 오답이정답가능, 오답약함,
복수쟁점혼합, 발문어색함, 정답누설, 해설부족, 메타데이터충돌.
"""
    else:
        criteria = """
Evaluate the item as a strict final certifier for Korean legal descriptive QA.
Do not grade leniently. If any critical gate fails, pass_or_fail must be "fail".
Check: Source Grounding, Answer Completeness, Rubric Atomicity, Question Clarity,
Evaluation Feasibility, Explanation Quality, Korean Naturalness, No Hallucination,
Metadata Consistency.
Return JSON only with:
pass_or_fail, critical_fail, scores(source_grounding, answer_completeness,
rubric_atomicity, question_clarity, evaluation_feasibility, explanation_quality,
korean_naturalness, no_hallucination), hard_fail_tags, repair_action,
one_sentence_reason, required_answer_elements, missing_elements, hallucination_risk.
Use Korean strings for reasons and tags.
Allowed hard_fail_tags: 근거불일치, 핵심요건누락, 채점기준불명확, 복수과제혼합,
해설부족, 문장어색함, 환각, 메타데이터충돌.
"""
    return criteria + "\n\nITEM_JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_judge_json(parsed: dict[str, Any], task: str) -> dict[str, Any]:
    tags = parsed.get("hard_fail_tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]
    allowed = OBJECTIVE_ERROR_TAGS if task == "objective" else DESCRIPTIVE_ERROR_TAGS
    parsed["hard_fail_tags"] = [str(tag) for tag in tags if str(tag) in allowed]
    parsed.setdefault("pass_or_fail", "fail")
    parsed.setdefault("critical_fail", parsed.get("pass_or_fail") != "pass")
    parsed.setdefault("repair_action", "manual_review")
    parsed.setdefault("one_sentence_reason", "")
    return parsed


def judge_one(row: dict[str, Any], judge_stage: str = "strict_final") -> dict[str, Any]:
    started = time.monotonic()
    response = call_gemini_json(
        build_judge_prompt(row),
        response_label=f"{row['_row_id']}::{judge_stage}",
        model_candidates=JUDGE_MODEL_CANDIDATES,
        temperature=JUDGE_TEMPERATURE,
        timeout_seconds=JUDGE_TIMEOUT_SECONDS,
        allowed_error_tags=None,
    )
    parsed = normalize_judge_json(response["json"], row["_task"])
    return {
        "row_id": row["_row_id"],
        "problem_id": row.get("problem_id", row["_row_id"]),
        "task": row["_task"],
        "split": row["_split"],
        "doc_type_name": row.get("doc_type_name", ""),
        "family_id": row.get("family_id", ""),
        "judge_stage": judge_stage,
        "judge_model": response["model"],
        "judge_mode": "gemini_api",
        "judge_elapsed_seconds": round(time.monotonic() - started, 3),
        "pass_or_fail": parsed.get("pass_or_fail", "fail"),
        "critical_fail": str(parsed.get("critical_fail", True)).lower(),
        "hard_fail_tags": "|".join(parsed.get("hard_fail_tags", [])),
        "repair_action": parsed.get("repair_action", "manual_review"),
        "one_sentence_reason": parsed.get("one_sentence_reason", ""),
        "predicted_correct_choice": parsed.get("predicted_correct_choice", ""),
        "provided_correct_choice": parsed.get("provided_correct_choice", row.get("correct_choice", "")),
        "judge_json": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
    }


def load_judge_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["row_id"]] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_judge_batch(
    rows: list[dict[str, Any]],
    output_path: Path,
    workers: int,
    judge_stage: str = "strict_final",
) -> dict[str, dict[str, Any]]:
    existing = load_judge_rows(output_path)
    pending = [row for row in rows if row["_row_id"] not in existing and row.get("_release_gate_status") != "quarantine"]
    if not pending:
        return existing

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge_one, row, judge_stage): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            existing[row["row_id"]] = row
            append_jsonl(output_path, row)
            if index % 25 == 0:
                print(f"[strict judge checkpoint] completed={len(existing)} pending={len(pending)-index}")
    return existing


def needs_openai_repair(row: dict[str, Any], judge_row: dict[str, Any]) -> bool:
    gate = derived_release_gate(row, judge_row)
    if gate["derived_status"] == "release_ready":
        return False
    if gate["repair_allowed"] != "예":
        return False
    action = normalized_action(judge_row.get("repair_action", ""))
    tags = set(filter(None, judge_row.get("hard_fail_tags", "").split("|")))
    if UNREPAIRABLE_TAGS & tags:
        return False
    return action not in {"quarantine", "manual_quarantine", "no_repair", "no repair"}


def build_repair_messages(row: dict[str, Any], judge_row: dict[str, Any]) -> list[dict[str, str]]:
    payload = compact_row_for_prompt(row)
    system = (
        "You repair Korean legal QA dataset rows. Return strict JSON only. "
        "Do not add facts outside the provided gold answer/explanation. "
        "Preserve legal meaning. Preserve split, identifiers, and task type. "
        "If the row cannot be safely repaired, return {\"repair_possible\": false}."
    )
    if row["_task"] == "objective":
        user = {
            "task": "repair_objective_single_best",
            "item": payload,
            "judge_failure": judge_row,
            "required_output": {
                "repair_possible": True,
                "generated_stem": "natural Korean legal exam stem",
                "choice_a": "A",
                "choice_b": "B",
                "choice_c": "C",
                "choice_d": "D",
                "correct_choice": "A/B/C/D",
                "gold_reference_explanation": "complete explanation grounded only in source",
                "repair_reason": "short Korean reason",
            },
        }
    else:
        user = {
            "task": "repair_descriptive_qa",
            "item": payload,
            "judge_failure": judge_row,
            "required_output": {
                "repair_possible": True,
                "generated_problem": "clear single-task Korean descriptive question",
                "gold_short_answer": "grounded model answer",
                "gold_reference_explanation": "complete rubric/explanation grounded only in source",
                "repair_reason": "short Korean reason",
            },
        }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def apply_repair_json(row: dict[str, Any], repair_json: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not repair_json.get("repair_possible", False):
        return None, {"repair_possible": "아니오", "repair_reason": repair_json.get("repair_reason", "repair_not_possible")}
    repaired = dict(row)
    if row["_task"] == "objective":
        for key in ("generated_stem", "choice_a", "choice_b", "choice_c", "choice_d", "correct_choice", "gold_reference_explanation"):
            if repair_json.get(key):
                repaired[key] = repair_json[key]
    else:
        for key in ("generated_problem", "gold_short_answer", "gold_reference_explanation"):
            if repair_json.get(key):
                repaired[key] = repair_json[key]
    repaired["_openai_repair_applied"] = "예"
    repaired["_openai_repair_reason"] = repair_json.get("repair_reason", "")
    return repaired, {"repair_possible": "예", "repair_reason": repair_json.get("repair_reason", "")}


def run_repairs(
    rows_by_id: dict[str, dict[str, Any]],
    judge_rows: dict[str, dict[str, Any]],
    repair_log_path: Path,
    workers: int,
    attempt: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    existing_repairs = load_judge_rows(repair_log_path)
    repair_targets = [
        rows_by_id[row_id]
        for row_id, judge_row in judge_rows.items()
        if row_id in rows_by_id and row_id not in existing_repairs and needs_openai_repair(rows_by_id[row_id], judge_row)
    ]
    repaired_rows: dict[str, dict[str, Any]] = {}
    revision_rows: list[dict[str, Any]] = []

    def repair_one(row: dict[str, Any]) -> dict[str, Any]:
        judge_row = judge_rows[row["_row_id"]]
        started = time.monotonic()
        response = call_openai_json(
            build_repair_messages(row, judge_row),
            response_label=f"{row['_row_id']}::bounded_repair_attempt{attempt}",
            model_candidates=REPAIR_MODEL_CANDIDATES,
            temperature=REPAIR_TEMPERATURE,
            max_tokens=REPAIR_MAX_TOKENS,
            timeout_seconds=REPAIR_TIMEOUT_SECONDS,
        )
        repaired, repair_meta = apply_repair_json(row, response["json"])
        return {
            "row_id": row["_row_id"],
            "problem_id": row.get("problem_id", row["_row_id"]),
            "task": row["_task"],
            "repair_model": response["model"],
            "repair_elapsed_seconds": round(time.monotonic() - started, 3),
            "repair_possible": repair_meta["repair_possible"],
            "repair_reason": repair_meta["repair_reason"],
            "repaired_row_json": json.dumps(repaired or {}, ensure_ascii=False, sort_keys=True),
        }

    if repair_targets:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(repair_one, row): row for row in repair_targets}
            for index, future in enumerate(as_completed(futures), start=1):
                repair_result = future.result()
                existing_repairs[repair_result["row_id"]] = repair_result
                append_jsonl(repair_log_path, repair_result)
                if index % 10 == 0:
                    print(f"[repair checkpoint] completed={index} total={len(repair_targets)}")

    for row_id, repair_result in existing_repairs.items():
        if repair_result.get("repair_possible") == "예":
            repaired = json.loads(repair_result.get("repaired_row_json", "{}"))
            if repaired:
                repaired_rows[row_id] = repaired
                revision_rows.append(
                    {
                        "problem_id": row_id,
                        "revision_id": f"{row_id}::bounded_repair_r{attempt}",
                        "parent_problem_id": row_id,
                        "repair_stage": "bounded_openai_repair",
                        "repair_attempt": attempt,
                        "repair_reason": repair_result.get("repair_reason", ""),
                        "repair_action": judge_rows.get(row_id, {}).get("repair_action", ""),
                        "validation_status": "pending_rejudge",
                    }
                )
    return repaired_rows, revision_rows


def build_release_rows(
    rows: list[dict[str, Any]],
    judge_rows: dict[str, dict[str, Any]],
    max_repair_exhausted: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for row in rows:
        rid = row["_row_id"]
        task = row["_task"]
        split = row["_split"]
        if row.get("_release_gate_status") == "quarantine":
            release_status = "quarantine"
            reason = row.get("_release_gate_reason", "split_lock_disposition")
            selected_judge = {}
            gate = {"derived_gate_reasons": reason, "repair_allowed": "아니오"}
        else:
            selected_judge = judge_rows.get(rid, {})
            gate = derived_release_gate(row, selected_judge)
            release_status = gate["derived_status"]
            reason = gate["derived_gate_reasons"] or selected_judge.get("one_sentence_reason", "")
            if release_status == "repair_planned" and max_repair_exhausted:
                release_status = "manual_review"
                reason = "max_repair_attempts_exhausted; " + reason

        validation_rows.append(
            {
                "row_id": rid,
                "problem_id": row.get("problem_id", rid),
                "task": task,
                "split": split,
                "doc_type_name": row.get("doc_type_name", ""),
                "family_id": row.get("family_id", ""),
                "release_status": release_status,
                "release_reason": reason,
                "derived_gate_reasons": gate.get("derived_gate_reasons", ""),
                "repair_allowed": gate.get("repair_allowed", ""),
                "judge_stage_used": selected_judge.get("judge_stage", ""),
                "pass_or_fail": selected_judge.get("pass_or_fail", ""),
                "critical_fail": selected_judge.get("critical_fail", ""),
                "hard_fail_tags": selected_judge.get("hard_fail_tags", ""),
                "repair_action": selected_judge.get("repair_action", ""),
                "predicted_correct_choice": selected_judge.get("predicted_correct_choice", ""),
                "provided_correct_choice": selected_judge.get("provided_correct_choice", ""),
            }
        )
        if release_status == "release_ready":
            export_row = {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            # 일부 legacy objective row는 problem_id가 비어 있어 final export에서 중복 ID처럼 보인다.
            # release-facing split에는 내부 row id를 승격해 row 식별성을 보존한다.
            if not export_row.get("problem_id"):
                export_row["problem_id"] = rid
            final_rows.append(export_row)
    return final_rows, validation_rows


def export_task(row: dict[str, Any]) -> str:
    task = row.get("_task")
    if task in {"objective", "descriptive"}:
        return task
    return "descriptive" if row.get("problem_task_type", "").startswith("descriptive") else "objective"


def write_split_exports(root: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in ("objective", "descriptive"):
        for split in SPLITS:
            grouped[(task, split)] = []
    for row in rows:
        task = export_task(row)
        split = row.get("_split") or row.get("split", "")
        if split not in SPLITS:
            continue
        grouped[(task, split)].append(row)
    for (task, split), split_rows in grouped.items():
        write_jsonl(root / task / f"{split}.jsonl", split_rows)


def write_release_supporting_artifacts(
    release_root: Path,
    final_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    dataset_rows: list[dict[str, Any]] = []
    for row in final_rows:
        dataset_rows.append(
            {
                "problem_id": row.get("problem_id", ""),
                "problem_task_type": row.get("problem_task_type", ""),
                "task": export_task(row),
                "split": row.get("split", ""),
                "doc_type_name": row.get("doc_type_name", ""),
                "family_id": row.get("family_id", ""),
                "seed_sample_id": row.get("seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "run_name": row.get("run_name", ""),
                "release_status": "release_ready",
            }
        )
    write_csv(
        release_root / "manifests/dataset_manifest.csv",
        dataset_rows,
        [
            "problem_id",
            "problem_task_type",
            "task",
            "split",
            "doc_type_name",
            "family_id",
            "seed_sample_id",
            "source_subset",
            "run_name",
            "release_status",
        ],
    )

    package_counter = Counter((row["task"], row["run_name"]) for row in dataset_rows)
    package_rows = [
        {"task": task, "run_name": run_name, "release_ready_count": count}
        for (task, run_name), count in sorted(package_counter.items())
    ]
    write_csv(
        release_root / "manifests/package_source_manifest.csv",
        package_rows,
        ["task", "run_name", "release_ready_count"],
    )

    split_counts = Counter((row["task"], row["doc_type_name"], row["split"]) for row in dataset_rows)
    status_counts = Counter(row["release_status"] for row in validation_rows)
    type_report = [
        f"# {manifest['release_id']} type balance report",
        "",
        "## release-ready split counts",
        "",
        "| task | doc_type | train | dev | test | total |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task in ("objective", "descriptive"):
        doc_types = sorted({doc for row_task, doc, _ in split_counts if row_task == task})
        for doc_type in doc_types:
            train = split_counts.get((task, doc_type, "train"), 0)
            dev = split_counts.get((task, doc_type, "dev"), 0)
            test = split_counts.get((task, doc_type, "test"), 0)
            type_report.append(f"| `{task}` | `{doc_type}` | `{train}` | `{dev}` | `{test}` | `{train + dev + test}` |")
    (release_root / "reports/type_balance_report.md").write_text("\n".join(type_report) + "\n", encoding="utf-8")

    methodology = [
        f"# {manifest['release_id']} methodology note",
        "",
        "이번 release는 counted snapshot을 그대로 최종 데이터셋으로 승격하지 않고,",
        "split-lock quarantine, deterministic stem repair, strict final Judge, bounded repair, rejudge, derived release gate를 통과한 row만 컴파일했다.",
        "",
        "## gate summary",
        "",
        f"- release_ready: `{status_counts.get('release_ready', 0)}`",
        f"- manual_review: `{status_counts.get('manual_review', 0)}`",
        f"- quarantine: `{status_counts.get('quarantine', 0)}`",
        f"- repair_attempts_completed: `{manifest.get('repair_attempts_completed', 0)}`",
        "",
        "candidate/count/release를 분리했기 때문에, manual_review와 quarantine row는 provenance에는 남지만 model-facing final split에는 들어가지 않는다.",
    ]
    (release_root / "reports/methodology_note.md").write_text("\n".join(methodology) + "\n", encoding="utf-8")

    release_notes = [
        f"# {manifest['release_id']} release notes",
        "",
        f"- source snapshot: `{manifest['snapshot_id']}`",
        f"- repaired snapshot: `{manifest['repaired_snapshot_id']}`",
        f"- strict Judge log: `{manifest['strict_judge_log']}`",
        "- final_dataset에는 `release_ready` row만 포함한다.",
        "- `manual_review`와 `quarantine` row는 `manifests/validation_manifest.csv`에서 추적한다.",
    ]
    (release_root / "reports/release_notes.md").write_text("\n".join(release_notes) + "\n", encoding="utf-8")


def write_reports(
    release_root: Path,
    repaired_root: Path,
    validation_rows: list[dict[str, Any]],
    revision_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    status_counts = Counter(row["release_status"] for row in validation_rows)
    task_status_counts = Counter((row["task"], row["release_status"]) for row in validation_rows)
    report = [
        f"# {manifest['release_id']} final validation report",
        "",
        "## summary",
        "",
        f"- snapshot_id: `{manifest['snapshot_id']}`",
        f"- release_id: `{manifest['release_id']}`",
        f"- release_ready: `{status_counts.get('release_ready', 0)}`",
        f"- manual_review: `{status_counts.get('manual_review', 0)}`",
        f"- quarantine: `{status_counts.get('quarantine', 0)}`",
        f"- strict_judge_api_executed: `{manifest['strict_judge_api_executed']}`",
        f"- repair_api_executed: `{manifest['repair_api_executed']}`",
        f"- derived_release_gate_applied: `{manifest.get('derived_release_gate_applied', True)}`",
        f"- repair_attempts_completed: `{manifest.get('repair_attempts_completed', 0)}`",
        "",
        "## task/status",
        "",
        "| task | status | count |",
        "| --- | --- | ---: |",
    ]
    for (task, status), count in sorted(task_status_counts.items()):
        report.append(f"| `{task}` | `{status}` | `{count}` |")
    report.extend(
        [
            "",
            "## release policy",
            "",
            "Only `release_ready` rows are written to `final_dataset` split JSONL files.",
            "`repair_planned`, `manual_review`, and `quarantine` rows remain in validation manifests and are not model-facing release rows.",
        ]
    )
    (release_root / "reports/final_validation_report.md").parent.mkdir(parents=True, exist_ok=True)
    (release_root / "reports/final_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    repair_report = [
        f"# {manifest['release_id']} repair report",
        "",
        f"- revision rows: `{len(revision_rows)}`",
        f"- repaired snapshot root: `{repaired_root}`",
    ]
    (repaired_root / "reports/repair_report.md").parent.mkdir(parents=True, exist_ok=True)
    (repaired_root / "reports/repair_report.md").write_text("\n".join(repair_report) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> None:
    snapshot_root = DATASET_CANDIDATES / args.snapshot_id
    repaired_root = DATASET_CANDIDATES / args.repaired_snapshot_id
    release_root = FINAL_DATASET_ROOT / args.release_id
    logs_root = snapshot_root / "final_validation_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    rows = load_snapshot_rows(snapshot_root)
    quarantine_ids = load_quarantine_ids(snapshot_root)
    pre_repaired_rows, deterministic_revision_rows = apply_prefinal_repairs(rows, quarantine_ids)
    rows_by_id = {row["_row_id"]: row for row in pre_repaired_rows}

    # repaired_snapshot은 Judge 전 deterministic repair와 split-lock quarantine을 반영한 중간 산출물이다.
    write_split_exports(repaired_root, pre_repaired_rows)

    strict_judge_log = logs_root / "strict_final_judge.jsonl"
    judge_rows = run_judge_batch(pre_repaired_rows, strict_judge_log, args.judge_workers)

    # Judge가 pass라고 해도 derived gate에서 repair_planned가 될 수 있으므로,
    # rejudge 결과를 현재 Judge 상태로 덮어가며 최대 시도 횟수만큼 bounded repair를 수행한다.
    current_judge_rows = dict(judge_rows)
    api_revision_rows: list[dict[str, Any]] = []
    repair_logs: list[str] = []
    rejudge_logs: list[str] = []
    repair_attempts_completed = 0
    for attempt in range(1, max(1, args.max_repair_attempts) + 1):
        repair_log = logs_root / f"bounded_repair_attempt{attempt}.jsonl"
        repaired_by_api, attempt_revision_rows = run_repairs(
            rows_by_id,
            current_judge_rows,
            repair_log,
            args.repair_workers,
            attempt,
        )
        repair_logs.append(str(repair_log))
        api_revision_rows.extend(attempt_revision_rows)
        if not repaired_by_api:
            break
        repair_attempts_completed = attempt
        for row_id, repaired in repaired_by_api.items():
            rows_by_id[row_id] = repaired

        rejudge_log = logs_root / f"repair_rejudge_attempt{attempt}.jsonl"
        rejudge_logs.append(str(rejudge_log))
        rejudge_rows = run_judge_batch(
            list(repaired_by_api.values()),
            rejudge_log,
            args.judge_workers,
            judge_stage=f"repair_rejudge_attempt{attempt}",
        )
        current_judge_rows.update(rejudge_rows)

    final_rows, validation_rows = build_release_rows(
        list(rows_by_id.values()),
        current_judge_rows,
        max_repair_exhausted=repair_attempts_completed >= max(1, args.max_repair_attempts),
    )
    write_split_exports(release_root, final_rows)

    revision_rows = deterministic_revision_rows + api_revision_rows
    write_csv(
        repaired_root / "manifests/revision_manifest.csv",
        revision_rows,
        [
            "problem_id",
            "revision_id",
            "parent_problem_id",
            "repair_stage",
            "repair_attempt",
            "repair_reason",
            "repair_action",
            "applied_patterns",
            "before_text",
            "after_text",
            "validation_status",
        ],
    )
    write_csv(
        release_root / "manifests/validation_manifest.csv",
        validation_rows,
        [
            "row_id",
            "problem_id",
            "task",
            "split",
            "doc_type_name",
            "family_id",
            "release_status",
            "release_reason",
            "derived_gate_reasons",
            "repair_allowed",
            "judge_stage_used",
            "pass_or_fail",
            "critical_fail",
            "hard_fail_tags",
            "repair_action",
            "predicted_correct_choice",
            "provided_correct_choice",
        ],
    )

    release_counts = Counter((row.get("problem_task_type", "").startswith("descriptive") and "descriptive" or "objective", row.get("split", "")) for row in final_rows)
    status_counts = Counter(row["release_status"] for row in validation_rows)
    manifest = {
        "snapshot_id": args.snapshot_id,
        "repaired_snapshot_id": args.repaired_snapshot_id,
        "release_id": args.release_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strict_judge_api_executed": True,
        "repair_api_executed": repair_attempts_completed > 0,
        "derived_release_gate_applied": True,
        "repair_attempts_completed": repair_attempts_completed,
        "judge_model_candidates": list(JUDGE_MODEL_CANDIDATES),
        "repair_model_candidates": list(REPAIR_MODEL_CANDIDATES),
        "release_counts": {f"{task}/{split}": count for (task, split), count in release_counts.items()},
        "release_status_counts": dict(status_counts),
        "strict_judge_log": str(strict_judge_log),
        "bounded_repair_logs": repair_logs,
        "repair_rejudge_logs": rejudge_logs,
        "final_dataset_created": True,
    }
    write_json(release_root / "manifests/release_manifest.json", manifest)
    write_json(repaired_root / "manifests/repaired_snapshot_manifest.json", manifest)
    write_release_supporting_artifacts(release_root, final_rows, validation_rows, manifest)
    write_reports(release_root, repaired_root, validation_rows, revision_rows, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default="kcc_2026_04_27_counted_snapshot_v1")
    parser.add_argument("--repaired-snapshot-id", default="kcc_2026_04_27_repaired_snapshot_v1")
    parser.add_argument("--release-id", default="kcc_2026_04_27_validated_v1")
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--repair-workers", type=int, default=32)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute(args)


if __name__ == "__main__":
    main()
