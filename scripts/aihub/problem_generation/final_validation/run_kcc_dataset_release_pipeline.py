#!/usr/bin/env python3
"""Prepare final validation manifests for a counted snapshot.

The current stop line is preparation, not release. Therefore this script only
performs local deterministic preflight and writes validation scaffolding; it
does not call any LLM Judge API.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_CANDIDATES = REPO_ROOT / "data/processed/aihub/problem_generation/dataset_candidates"
SPLITS = ("train", "dev", "test")


STEM_HYGIENE_PATTERNS = [
    ("무엇인가 를", "조사 오류"),
    ("무엇인가 옳은", "중복/어색한 명령"),
    ("기간은 얼마인가요의 범위", "깨진 문장 결합"),
    ("가장 옳은 것은.*고르시오", "중복 명령"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_identifier(row: dict[str, Any], task: str, split: str, index: int) -> str:
    return (
        row.get("problem_id")
        or row.get("candidate_id")
        or row.get("seed_sample_id")
        or f"{task}_{split}_{index:06d}"
    )


def stem_text(row: dict[str, Any]) -> str:
    return row.get("generated_stem") or row.get("generated_problem") or row.get("question") or ""


def deterministic_status(row: dict[str, Any], task: str) -> tuple[str, list[str]]:
    findings: list[str] = []
    if not row.get("doc_type_name"):
        findings.append("doc_type_name_missing")
    if not row.get("family_id"):
        findings.append("family_id_missing")
    if task == "objective":
        choices = [row.get(f"choice_{label}") for label in ("a", "b", "c", "d")]
        if any(not choice for choice in choices):
            findings.append("objective_choice_missing")
        if row.get("correct_choice") not in {"A", "B", "C", "D"}:
            findings.append("correct_choice_invalid")
        text = stem_text(row)
        for pattern, reason in STEM_HYGIENE_PATTERNS:
            if re.search(pattern, text):
                findings.append(f"stem_hygiene:{reason}")
    else:
        if not (row.get("generated_problem") or row.get("question")):
            findings.append("descriptive_question_missing")
        if not (row.get("gold_short_answer") or row.get("gold_reference_explanation")):
            findings.append("descriptive_answer_reference_missing")
    return ("deterministic_review_required" if findings else "deterministic_ready", findings)


def load_split_lock_counts(snapshot_root: Path) -> Counter[str]:
    path = snapshot_root / "manifests/split_lock_manifest.csv"
    if not path.exists():
        return Counter({"missing_split_lock_manifest": 1})
    with path.open(encoding="utf-8", newline="") as f:
        return Counter(row["split_lock_status"] for row in csv.DictReader(f))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_rank(split: str) -> int:
    # final release에서는 평가 누수를 더 보수적으로 막기 위해 test, dev,
    # train 순으로 보존 우선순위를 둔다.
    return {"test": 3, "dev": 2, "train": 1}.get(split, 0)


def build_split_lock_disposition(snapshot_root: Path, snapshot_id: str, release_id: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    split_lock_rows = read_csv(snapshot_root / "manifests/split_lock_manifest.csv")
    manifest_rows = read_csv(snapshot_root / "manifests/dataset_manifest.csv")
    rows_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        family_id = row.get("family_id") or row.get("seed_sample_id") or row.get("problem_id", "")
        rows_by_family[family_id].append(row)

    disposition_rows: list[dict[str, Any]] = []
    disposition_counter: Counter[str] = Counter()
    for split_lock in split_lock_rows:
        if split_lock.get("split_lock_status") != "cross_split_review_required":
            continue
        family_id = split_lock.get("family_id", "")
        family_rows = rows_by_family.get(family_id, [])
        splits = sorted({row.get("split", "") for row in family_rows})
        keep_split = max(splits, key=split_rank) if splits else ""
        if {"train"} & set(splits) and {"dev", "test"} & set(splits):
            disposition_reason = "train_eval_boundary_overlap"
        elif {"dev", "test"} <= set(splits):
            disposition_reason = "dev_test_boundary_overlap"
        else:
            disposition_reason = "multi_split_family"

        for row in family_rows:
            split = row.get("split", "")
            action = "keep_for_validation" if split == keep_split else "quarantine_before_final_release"
            # Snapshot은 유지하되 final_dataset 승격 전에 row-level release action을
            # 명시해 Judge API와 release compiler가 같은 결정을 재사용하게 한다.
            disposition_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "release_id": release_id,
                    "family_id": family_id,
                    "problem_id": row.get("problem_id", ""),
                    "snapshot_task": row.get("snapshot_task", ""),
                    "doc_type_name": row.get("doc_type_name", ""),
                    "split": split,
                    "snapshot_package": row.get("snapshot_package", ""),
                    "split_group": "|".join(splits),
                    "recommended_keep_split": keep_split,
                    "split_lock_disposition": action,
                    "disposition_reason": disposition_reason,
                    "api_required": "아니오",
                    "final_release_blocker_until_applied": "예" if action != "keep_for_validation" else "아니오",
                }
            )
            disposition_counter[action] += 1
    return disposition_rows, disposition_counter


def build_stem_hygiene_manifest(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    hygiene_rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    for row in rows:
        findings = [item for item in row.get("deterministic_findings", "").split("|") if item.startswith("stem_hygiene:")]
        if not findings:
            continue
        action = "repair_before_final_release"
        for finding in findings:
            counter[finding] += 1
        hygiene_rows.append(
            {
                "snapshot_id": row.get("snapshot_id", ""),
                "release_id": row.get("release_id", ""),
                "row_id": row.get("row_id", ""),
                "task": row.get("task", ""),
                "split": row.get("split", ""),
                "doc_type_name": row.get("doc_type_name", ""),
                "family_id": row.get("family_id", ""),
                "stem_hygiene_findings": "|".join(findings),
                "stem_hygiene_disposition": action,
                "api_required": "일부",
                "final_release_blocker_until_applied": "예",
            }
        )
    return hygiene_rows, counter


def run_preflight(snapshot_id: str, release_id: str, judge_workers: int, repair_workers: int) -> None:
    snapshot_root = DATASET_CANDIDATES / snapshot_id
    rows: list[dict[str, Any]] = []
    task_split_counts: Counter[tuple[str, str]] = Counter()
    row_ids_by_task: dict[str, list[str]] = {"objective": [], "descriptive": []}
    finding_counter: Counter[str] = Counter()

    for task in ("objective", "descriptive"):
        for split in SPLITS:
            split_rows = read_jsonl(snapshot_root / task / f"{split}.jsonl")
            task_split_counts[(task, split)] = len(split_rows)
            for index, row in enumerate(split_rows, start=1):
                row_id = row_identifier(row, task, split, index)
                status, findings = deterministic_status(row, task)
                for finding in findings:
                    finding_counter[finding] += 1
                row_ids_by_task[task].append(row_id)
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "release_id": release_id,
                        "row_id": row_id,
                        "task": task,
                        "split": split,
                        "doc_type_name": row.get("doc_type_name", ""),
                        "family_id": row.get("family_id", ""),
                        "deterministic_status": status,
                        "deterministic_findings": "|".join(findings),
                        "strict_final_judge_status": "pending",
                        "repair_status": "not_started",
                        "release_status": "not_final_dataset",
                    }
                )

    duplicate_counts = {
        task: sum(count - 1 for count in Counter(ids).values() if count > 1)
        for task, ids in row_ids_by_task.items()
    }
    split_lock_counts = load_split_lock_counts(snapshot_root)

    write_csv(
        snapshot_root / "manifests/deterministic_validation_manifest.csv",
        rows,
        [
            "snapshot_id",
            "release_id",
            "row_id",
            "task",
            "split",
            "doc_type_name",
            "family_id",
            "deterministic_status",
            "deterministic_findings",
            "strict_final_judge_status",
            "repair_status",
            "release_status",
        ],
    )

    split_lock_disposition_rows, split_lock_disposition_counts = build_split_lock_disposition(
        snapshot_root, snapshot_id, release_id
    )
    write_csv(
        snapshot_root / "manifests/split_lock_disposition_manifest.csv",
        split_lock_disposition_rows,
        [
            "snapshot_id",
            "release_id",
            "family_id",
            "problem_id",
            "snapshot_task",
            "doc_type_name",
            "split",
            "snapshot_package",
            "split_group",
            "recommended_keep_split",
            "split_lock_disposition",
            "disposition_reason",
            "api_required",
            "final_release_blocker_until_applied",
        ],
    )

    stem_hygiene_rows, stem_hygiene_counts = build_stem_hygiene_manifest(rows)
    write_csv(
        snapshot_root / "manifests/stem_hygiene_manifest.csv",
        stem_hygiene_rows,
        [
            "snapshot_id",
            "release_id",
            "row_id",
            "task",
            "split",
            "doc_type_name",
            "family_id",
            "stem_hygiene_findings",
            "stem_hygiene_disposition",
            "api_required",
            "final_release_blocker_until_applied",
        ],
    )

    summary = {
        "snapshot_id": snapshot_id,
        "release_id": release_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "validation_preparation_only",
        "judge_workers": judge_workers,
        "repair_workers": repair_workers,
        "task_split_counts": {f"{task}/{split}": count for (task, split), count in task_split_counts.items()},
        "duplicate_row_id_overflow_by_task": duplicate_counts,
        "deterministic_finding_counts": dict(finding_counter),
        "split_lock_counts": dict(split_lock_counts),
        "split_lock_disposition_counts": dict(split_lock_disposition_counts),
        "split_lock_disposition_manifest": str(
            snapshot_root / "manifests/split_lock_disposition_manifest.csv"
        ),
        "stem_hygiene_counts": dict(stem_hygiene_counts),
        "stem_hygiene_manifest": str(snapshot_root / "manifests/stem_hygiene_manifest.csv"),
        "llm_judge_executed": False,
        "final_dataset_created": False,
    }
    (snapshot_root / "manifests/release_pipeline_preflight.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = f"""# {snapshot_id} release pipeline preflight

이 문서는 final Validation Judge 실행 전 준비 상태를 기록한다.
이 단계에서는 LLM Judge API를 호출하지 않았고, `final_dataset`도 생성하지 않았다.

## split count

| task/split | count |
| --- | ---: |
{chr(10).join(f"| `{task}/{split}` | `{count}` |" for (task, split), count in sorted(task_split_counts.items()))}

## deterministic finding counts

| finding | count |
| --- | ---: |
{chr(10).join(f"| `{key}` | `{value}` |" for key, value in sorted(finding_counter.items())) or "| `none` | `0` |"}

## duplicate row id overflow

| task | duplicate overflow |
| --- | ---: |
| `objective` | `{duplicate_counts.get('objective', 0)}` |
| `descriptive` | `{duplicate_counts.get('descriptive', 0)}` |

## split-lock precheck

| status | count |
| --- | ---: |
{chr(10).join(f"| `{key}` | `{value}` |" for key, value in sorted(split_lock_counts.items()))}

## split-lock disposition

| disposition | row count |
| --- | ---: |
{chr(10).join(f"| `{key}` | `{value}` |" for key, value in sorted(split_lock_disposition_counts.items())) or "| `none` | `0` |"}

`cross_split_review_required` family는 snapshot rollback 없이 유지하되,
`split_lock_disposition_manifest.csv`에 final release 전 keep/quarantine 권고를 남겼다.

## stem hygiene disposition

| finding | row count |
| --- | ---: |
{chr(10).join(f"| `{key}` | `{value}` |" for key, value in sorted(stem_hygiene_counts.items())) or "| `none` | `0` |"}

`stem_hygiene_manifest.csv`는 final Judge/repair 단계에서 처리할 row-level 입력이다.

## next

다음 단계는 strict final Judge API 실행, bounded repair, leakage graph check, release report 생성이다.
"""
    (snapshot_root / "reports/release_pipeline_preflight.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--repair-workers", type=int, default=32)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--mode", default="objective,descriptive")
    parser.add_argument("--require-evidence-green", action="store_true")
    parser.add_argument("--require-family-split-lock", action="store_true")
    parser.add_argument("--write-paper-reports", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="현재 stop line은 준비 단계이므로 API를 호출하지 않는 local preflight만 수행한다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preflight(args.snapshot_id, args.release_id, args.judge_workers, args.repair_workers)


if __name__ == "__main__":
    main()
