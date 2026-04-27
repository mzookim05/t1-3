#!/usr/bin/env python3
"""Build a counted snapshot candidate from already-counted production packages.

This script intentionally writes under ``dataset_candidates`` instead of
``final_dataset`` because the reviewer explicitly separated counted snapshots
from fully validated releases.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_ROOT = REPO_ROOT / "data/processed/aihub/problem_generation"


# Current objective counted line from project/core/problem_target_inventory.md.
# Keep this list explicit so the snapshot is reproducible and does not
# accidentally sweep failed or candidate-only production batches into release
# preparation.
OBJECTIVE_PACKAGES = [
    "v2_difficulty_patch_r2",
    "production_batches/pb2_objective_candidate",
    "production_batches/pb3_objective_current_r2",
    "production_batches/pb4_objective_current_r2",
    "production_batches/pb9_cslot_final_replacement_package",
    "production_batches/objective_judgment_repair_a_slot_replacement_package",
    "production_batches/objective_interpretation_repair_dslot_final_replacement_package",
    "production_batches/objective_judgment_small_overgeneration_pilot",
    "production_batches/objective_interpretation_small_overgeneration_pilot",
    "production_batches/objective_judgment_medium_overgeneration_pilot",
    "production_batches/objective_interpretation_constrained_overgeneration_pilot",
    "production_batches/objective_decision_medium_overgeneration_pilot",
    "production_batches/objective_decision_addon_overgeneration_pilot",
    "production_batches/objective_non_law_eval_aware_superwave_judgment",
    "production_batches/objective_non_law_eval_aware_superwave_decision",
    "production_batches/objective_balanced_deficit_recovery_judgment",
    "production_batches/objective_balanced_deficit_recovery_interpretation_relaxed",
    "production_batches/objective_balanced_deficit_recovery_decision_eval_addon",
    "production_batches/objective_law_source_relaxed_repair_superwave",
    "production_batches/objective_interpretation_source_relaxed_repeat",
    "production_batches/objective_eval_micro_addon_decision",
    "production_batches/objective_type_deficit_overclosure_law_addon",
    "production_batches/objective_type_deficit_overclosure_judgment_eval_micro",
    "production_batches/objective_type_deficit_overclosure_interpretation_train_micro",
]


# Current descriptive counted line from project/core/problem_target_inventory.md.
# Historical v1 descriptive remains reference-only, so it is deliberately
# excluded from this snapshot.
DESCRIPTIVE_PACKAGES = [
    "v3_split_descriptive",
    "production_batches/pb1_descriptive",
    "production_batches/descriptive_wave_v2_constrained",
    "production_batches/descriptive_wave_v2_followup_constrained",
    "production_batches/descriptive_wave_v2_second_followup_constrained",
    "production_batches/descriptive_wave_v2_split_lock_hotfix_next_api",
    "production_batches/descriptive_tail_manifest_sync_constrained_followup",
    "production_batches/descriptive_inventory_linter_pointer_sync_medium_primary",
    "production_batches/descriptive_medium_repeat_availability_aware",
    "production_batches/descriptive_emergency_candidate128_final80",
    "production_batches/descriptive_source_reallocated_emergency_candidate128_final80",
    "production_batches/descriptive_source_reallocated_eval_targeted_candidate128_final80",
    "production_batches/descriptive_deficit_closure_train_heavy_addon_candidate64_final40",
    "production_batches/descriptive_train_micro_closure_judgment_candidate16_final5",
    "production_batches/descriptive_type_balance_extension_candidate64_final40",
    "production_batches/descriptive_type_deficit_overclosure_bundle",
]


SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def package_name(rel_path: str) -> str:
    return rel_path.split("/")[-1]


def included_manifest_row(row: dict[str, str]) -> bool:
    split = row.get("split", "")
    if split not in SPLITS:
        return False
    if row.get("train_eligible") == "아니오":
        return False
    if row.get("count_allowed") == "아니오":
        return False
    if row.get("count_disposition") == "candidate_not_counted":
        return False
    return True


def collect_package(task: str, rel_path: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    package_dir = PROCESSED_ROOT / rel_path
    name = package_name(rel_path)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(package_dir / f"{split}.jsonl")
        split_rows[split] = rows
        split_counts[split] = len(rows)

    manifest_rows = []
    for row in read_manifest(package_dir / "dataset_manifest.csv"):
        if included_manifest_row(row):
            # Snapshot provenance belongs in manifest files rather than
            # model-facing JSONL rows.
            row = dict(row)
            row["snapshot_task"] = task
            row["snapshot_package"] = name
            row["snapshot_package_path"] = str(package_dir)
            manifest_rows.append(row)

    expected = sum(split_counts.values())
    if expected != len(manifest_rows):
        raise ValueError(
            f"{name}: jsonl split count {expected} != included manifest rows {len(manifest_rows)}"
        )

    package_summary = {
        "task": task,
        "package": name,
        "package_path": str(package_dir),
        "train": split_counts["train"],
        "dev": split_counts["dev"],
        "test": split_counts["test"],
        "total": expected,
    }
    return split_rows, manifest_rows, package_summary


def build_snapshot(snapshot_id: str, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    all_json_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "objective": {split: [] for split in SPLITS},
        "descriptive": {split: [] for split in SPLITS},
    }
    all_manifest_rows: list[dict[str, Any]] = []
    package_summaries: list[dict[str, Any]] = []

    for task, packages in {
        "objective": OBJECTIVE_PACKAGES,
        "descriptive": DESCRIPTIVE_PACKAGES,
    }.items():
        for rel_path in packages:
            split_rows, manifest_rows, summary = collect_package(task, rel_path)
            for split in SPLITS:
                all_json_rows[task][split].extend(split_rows[split])
            all_manifest_rows.extend(manifest_rows)
            package_summaries.append(summary)

    for task in ("objective", "descriptive"):
        for split in SPLITS:
            write_jsonl(output_root / task / f"{split}.jsonl", all_json_rows[task][split])

    write_csv(output_root / "manifests/dataset_manifest.csv", all_manifest_rows)
    write_csv(output_root / "manifests/package_source_manifest.csv", package_summaries)

    inventory_rows = build_inventory_rows(all_manifest_rows)
    write_csv(output_root / "manifests/count_inventory_snapshot.csv", inventory_rows)

    split_lock_rows = build_split_lock_rows(all_manifest_rows)
    write_csv(output_root / "manifests/split_lock_manifest.csv", split_lock_rows)

    validation_rows = build_validation_manifest(all_manifest_rows)
    write_csv(output_root / "manifests/validation_manifest.csv", validation_rows)

    # snapshot_build_manifest 자체만 읽어도 reviewer-facing count를 재검산할 수
    # 있도록 split별 수치를 함께 남긴다.
    split_counts = {
        task: {split: len(all_json_rows[task][split]) for split in SPLITS}
        for task in ("objective", "descriptive")
    }
    inventory_counts = {
        task: sum(split_counts[task].values())
        for task in ("objective", "descriptive")
    }

    build_manifest = {
        "snapshot_id": snapshot_id,
        "snapshot_stage": "counted_snapshot_release_candidate",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "split_counts": split_counts,
        "inventory_counts": inventory_counts,
        "objective_count": inventory_counts["objective"],
        "descriptive_count": inventory_counts["descriptive"],
        "package_total": len(package_summaries),
        "final_dataset_status": "not_final_dataset_pending_validation_judge_and_repair",
    }
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests/snapshot_build_manifest.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_reports(output_root, snapshot_id, package_summaries, inventory_rows, split_lock_rows)


def build_inventory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counter[(row["snapshot_task"], row.get("doc_type_name", ""), row.get("split", ""))] += 1

    result: list[dict[str, Any]] = []
    for (task, doc_type, split), count in sorted(counter.items()):
        result.append({"task": task, "doc_type_name": doc_type, "split": split, "count": count})

    for task in ("objective", "descriptive"):
        for split in SPLITS:
            result.append(
                {
                    "task": task,
                    "doc_type_name": "__TOTAL__",
                    "split": split,
                    "count": sum(c for (t, _d, s), c in counter.items() if t == task and s == split),
                }
            )
        result.append(
            {
                "task": task,
                "doc_type_name": "__TOTAL__",
                "split": "__TOTAL__",
                "count": sum(c for (t, _d, _s), c in counter.items() if t == task),
            }
        )
    return result


def build_split_lock_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_id = row.get("family_id") or row.get("seed_sample_id") or row.get("problem_id", "")
        groups[family_id].append(row)

    result: list[dict[str, Any]] = []
    for family_id, family_rows in sorted(groups.items()):
        splits = sorted({row.get("split", "") for row in family_rows})
        tasks = sorted({row.get("snapshot_task", "") for row in family_rows})
        doc_types = sorted({row.get("doc_type_name", "") for row in family_rows})
        result.append(
            {
                "family_id": family_id,
                "row_count": len(family_rows),
                "tasks": "|".join(tasks),
                "doc_type_names": "|".join(doc_types),
                "splits": "|".join(splits),
                "split_count": len(splits),
                "split_lock_status": "split_locked" if len(splits) == 1 else "cross_split_review_required",
            }
        )
    return result


def build_validation_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        problem_id = row.get("problem_id") or row.get("candidate_id") or f"snapshot_row_{index:06d}"
        result.append(
            {
                "snapshot_row_index": index,
                "problem_id": problem_id,
                "task": row.get("snapshot_task", ""),
                "doc_type_name": row.get("doc_type_name", ""),
                "split": row.get("split", ""),
                "family_id": row.get("family_id", ""),
                "snapshot_package": row.get("snapshot_package", ""),
                "validation_status": "pending_validation",
                "deterministic_gate_status": "pending",
                "strict_final_judge_status": "pending",
                "repair_status": "not_started",
                "release_status": "not_final_dataset",
            }
        )
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_reports(
    output_root: Path,
    snapshot_id: str,
    package_summaries: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    split_lock_rows: list[dict[str, Any]],
) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    total_rows = [row for row in inventory_rows if row["doc_type_name"] == "__TOTAL__"]
    split_lock_counter = Counter(row["split_lock_status"] for row in split_lock_rows)
    count_report = f"""# {snapshot_id} counted snapshot build report

이 산출물은 counted snapshot release candidate이며, 아직 최종 검증 데이터셋이 아니다.

## count 요약

{markdown_table(total_rows, ["task", "doc_type_name", "split", "count"])}

## package source

{markdown_table(package_summaries, ["task", "package", "train", "dev", "test", "total"])}

## split-lock 사전 점검

| status | count |
| --- | ---: |
| split_locked | {split_lock_counter.get("split_locked", 0)} |
| cross_split_review_required | {split_lock_counter.get("cross_split_review_required", 0)} |

`cross_split_review_required` row는 snapshot build blocker가 아니다.
다만 final Validation Judge와 leakage graph check에서 반드시 재검토해야 한다.
"""
    (reports / "count_build_report.md").write_text(count_report, encoding="utf-8")

    quality_debt = f"""# {snapshot_id} known quality debt

이 snapshot에는 counted `train/dev/test` row만 포함한다.
audit row, quality tail, failed row, quota surplus는 model-facing snapshot 파일에 병합하지 않는다.

## stage naming

- 이 산출물은 `dataset_candidates/{snapshot_id}`로만 부른다.
- 이 산출물을 `final_dataset`으로 부르지 않는다.
- final Validation Judge, bounded repair, leakage check, release reporting이 모두 닫힌 뒤에만 `final_dataset/kcc_2026_04_27_validated_v1`로 승격한다.

## 남은 validation 작업

- 객관식/서술형 strict final Judge
- 객관식 stem hygiene hard gate
- 객관식 answer-blind keyedness와 source-span proof
- 서술형 rubric/completeness validation
- revision lineage를 보존하는 bounded repair loop
- repaired row와 sampled row에 대한 metamorphic validation

## release compiler P3 hygiene

- `artifact_paths.evidence_summary` 같은 legacy alias는 final release compiler에서 formal evidence card path와 구분해 정리한다.
- model-facing export와 answer-key/provenance manifest를 분리한다.
- snapshot JSONL 내부의 `raw_path`, `label_path`, `run_name`, judge score, linter/evidence pointer 같은 internal provenance는 final model-facing row에 과도하게 노출하지 않는다.
- dev/test row는 채점용 answer를 보존하되, 모델 입력용 view에서는 정답과 해설을 가린다.
"""
    (reports / "known_quality_debt.md").write_text(quality_debt, encoding="utf-8")

    # Reviewer response의 validation-as-compilation 설계를 산출물 안에
    # 고정해 두어, 다음 실행자가 request.md 없이도 같은 stop line을 재현할 수
    # 있게 한다.
    validation_plan = f"""# {snapshot_id} final Validation Judge 준비

## Status

준비만 완료했다.
이 snapshot builder는 final Validation Judge API 호출을 실행하지 않는다.
`dataset_candidates/{snapshot_id}`는 counted snapshot release candidate이고,
`final_dataset/kcc_2026_04_27_validated_v1`은 validation, repair, leakage check,
evidence, reporting이 모두 닫힌 뒤에만 생성한다.

## Validation-as-Compilation 원칙

Counted Snapshot은 최종 데이터셋이 아니라 증명 가능한 release candidate다.
Final Dataset은 아래 컴파일 단계를 통과한 뒤에만 승격한다.

| layer | 역할 | 산출물 |
| --- | --- | --- |
| Count Compiler | counted row만 모아 snapshot 생성 | `counted_snapshot_v1` |
| Evidence Compiler | linter/evidence/card/provenance 정합성 확인 | `evidence_bundle` |
| Validation Compiler | Judge, deterministic gate, leakage 검증 | `validation_manifest.csv` |
| Repair Compiler | 실패 row를 bounded repair loop로 수정 | `revision_manifest.csv` |
| Release Compiler | 최종 통과 row만 release로 승격 | `validated_v1` |
| Paper Compiler | 방법론, 표, 품질 지표를 논문용으로 생성 | `methodology_note.md`, `type_balance_report.md` |

## 전체 pipeline

| 단계 | 이름 | 핵심 작업 | 성공 조건 |
| ---: | --- | --- | --- |
| `0` | P2 Sync Gate | `223038`, `224152`, inventory stale 표 수정 | linter/evidence all-green |
| `1` | Counted Snapshot Build | counted row 전부 수집 | `dataset_candidates/...counted_snapshot_v1` 생성 |
| `2` | Schema Canonicalization | 객관식/서술형 row 표준화 | 모든 row가 공통 schema 통과 |
| `3` | Leakage Graph Check | `family_id`, `raw_path`, `label_path`, source 기반 train/dev/test 누수 검사 | cross-split family overlap `0` 또는 reviewer-approved exception |
| `4` | Deterministic Validation | 정답 label, 선택지 수, 중복, null, metadata 검증 | hard deterministic fail `0` |
| `5` | Final Validation Judge | 객관식/서술형 최종 rubric Judge | pass 또는 repair 대상 분리 |
| `6` | Bounded Repair Loop | 실패 row 자동 수정, 최대 `2회` | pass 승격 또는 quarantine |
| `7` | Release Candidate Lock | 최종 train/dev/test export | objective/descriptive split 파일 생성 |
| `8` | Quality Certification | 품질 보고서, 유형별 표, failure taxonomy | 제출용 report 완성 |

## 필요한 queue

| queue | purpose | api |
| --- | --- | --- |
| deterministic_queue | schema, duplicate, stem hygiene, split-lock precheck | 아니오 |
| objective_judge_queue | objective strict final validation | 예 |
| descriptive_judge_queue | descriptive strict final validation | 예 |
| repair_generation_queue | failed row 전용 bounded repair generation | 예 |
| repair_judge_queue | repaired row rejudge | 예 |
| evidence_queue | reports and evidence bundle | 아니오 |

## row 상태 머신

```text
pending_validation
-> deterministic_pass
-> judge_pass
-> release_ready

pending_validation
-> deterministic_fail 또는 judge_fail
-> repair_planned
-> repair_generated
-> repair_rejudged
-> release_ready 또는 quarantine
```

## objective Judge gate

객관식은 평균 점수 기반이 아니라 critical gate 방식으로 판정한다.
아래 항목 중 하나라도 critical fail이면 `pass_or_fail = fail`로 둔다.

| 축 | 질문 | fail 조건 |
| --- | --- | --- |
| Source Grounding | 정답과 해설이 source에 직접 근거하는가 | 근거 불일치, 외부 지식 추가 |
| Answer-Blind Keyedness | 정답 label 없이 풀어도 같은 답이 나오는가 | Judge 예측 답과 `correct_choice` 불일치 |
| Single-Best Answer | 정답이 정확히 하나인가 | 복수정답, 정답 없음 |
| Distractor Refutation | 각 오답이 source 기준 명확히 틀렸는가 | 오답이 정답 가능 |
| Near-Miss Quality | 오답이 너무 허술하지 않고 같은 법적 anchor를 공유하는가 | 오답약함, 무관한 일반론 |
| Single-Axis Stem | 하나의 쟁점만 묻는가 | 복수 쟁점 혼합 |
| Stem Naturalness | 발문이 자연스럽고 법률 시험 문항처럼 읽히는가 | 어색한 조사, 중복 명령, 깨진 문장 |
| Answer Leakage | 발문이 정답 선택지를 과도하게 암시하지 않는가 | 정답 누설, 정답 직접 복사 |
| Explanation Completeness | 해설이 정답 이유와 오답 반박을 모두 포함하는가 | 정답만 말하고 오답 반박 없음 |
| Metadata Consistency | `correct_choice`, choices, explanation, validator metadata가 일치하는가 | label map 충돌, 선택지 중복 |

## stem hygiene hard gate

LLM Judge 전에 deterministic regex와 phrase list로 발문 hygiene을 먼저 잡는다.

| 패턴 | 처리 |
| --- | --- |
| `무엇인가 를` | hard repair |
| `무엇인가 옳은` | hard repair |
| `기간은 얼마인가요의 범위` | hard fail |
| `가장 옳은 것은 ... 고르시오` 중복 명령 | repair |
| `다음 중 ... 무엇인가?`와 `고르시오` 중복 | repair |
| stem이 선택지와 같은 문장을 거의 반복 | answer leakage 후보 |
| 질문 축이 `주체+기간+절차`처럼 여러 개 | single-axis fail |
| 법령명/기관명만 바꾼 단순 회상형 | weak item 후보 |
| 선택지 길이에서 정답만 과도하게 김 | answer leakage 후보 |

Judge prompt에는 아래 취지를 반드시 포함한다.

```text
발문이 한국어 법률 시험 문항으로 자연스럽지 않으면 Grounding이 맞더라도 fail로 판정하라.
특히 조사 오류, 중복 명령, 깨진 문장 결합, "무엇인가 를", "기간은 얼마인가요의 범위" 같은 표현은 hard fail 또는 repair_required로 표시하라.
```

## source-span proof obligation

객관식 final Judge는 단순 점수만 내지 말고, 각 선택지별 source-span proof를 남긴다.

| 필드 | 의미 |
| --- | --- |
| `predicted_correct_choice` | 정답 label 없이 Judge가 푼 답 |
| `correct_support_span` | 정답을 지지하는 source 문장/조문 요약 |
| `distractor_A_refutation` | A가 오답이면 왜 틀렸는지 |
| `distractor_B_refutation` | B가 오답이면 왜 틀렸는지 |
| `distractor_C_refutation` | C가 오답이면 왜 틀렸는지 |
| `distractor_D_refutation` | D가 오답이면 왜 틀렸는지 |
| `stem_axis` | 주체/요건/효과/절차/기한/적용범위 중 무엇을 묻는지 |
| `axis_consistency` | 모든 선택지가 같은 축에서 경쟁하는지 |
| `naturalness_score` | 한국어 문항 자연성 |
| `hard_fail_tags` | 있으면 final 제외 |

## strict final Judge output contract

first-pass는 multi-axis JSON Judge로 돌리고, 위험 row는 second Judge로 보낸다.

```json
{{
  "pass_or_fail": "fail",
  "critical_fail": true,
  "predicted_correct_choice": "B",
  "provided_correct_choice": "C",
  "axis": "적용범위",
  "scores": {{
    "source_grounding": 5,
    "answer_blind_keyedness": 1,
    "single_best_answer": 1,
    "distractor_refutation": 2,
    "near_miss_quality": 4,
    "stem_naturalness": 3,
    "explanation_completeness": 4
  }},
  "hard_fail_tags": ["정답 비유일", "오답이 정답 가능"],
  "repair_action": "replace_distractor_B",
  "one_sentence_reason": "B 선택지도 source 기준 정답으로 읽혀 C와 복수정답이 됩니다."
}}
```

## descriptive Judge gate

서술형은 정답 유일성이 아니라 채점 가능성과 답안 완결성을 본다.

| 축 | 질문 | fail 조건 |
| --- | --- | --- |
| Source Grounding | 답안이 source에서 직접 지지되는가 | 외부 사실, 근거 없음 |
| Answer Completeness | 필수 요건을 빠짐없이 포함하는가 | 핵심 요건 누락 |
| Rubric Atomicity | 채점 기준이 원자적 항목으로 나뉘는가 | 뭉뚱그린 rubric |
| Question Clarity | 질문이 무엇을 요구하는지 명확한가 | 복수 과제 혼합 |
| Evaluation Feasibility | 사람이 일관되게 채점 가능한가 | 정답 범위 불명확 |
| Explanation Quality | 모범답안/해설이 충분한가 | 짧고 추상적 |
| Korean Naturalness | 문장이 자연스럽고 학습자료로 적합한가 | 어색한 문장 |
| No Hallucination | source 밖 요건을 만들지 않는가 | 근거 없는 보강 |

## metamorphic validation

LLM 점수에만 의존하지 않도록 아래 검증을 추가한다.

| 테스트 | 방식 | 목적 |
| --- | --- | --- |
| `choice_permutation_test` | 선택지 순서를 섞고 `correct_choice`를 재계산 | label remap 안정성 |
| `answer_blind_solve` | 정답 label과 해설을 숨기고 Judge가 직접 풂 | keyedness 검증 |
| `distractor_flip_test` | 오답 하나를 정답처럼 바꾼 mutation fixture 투입 | Judge 민감도 확인 |
| `stem_only_leakage_test` | source 없이 stem/choices만 보고 정답이 드러나는지 확인 | 정답 누설 탐지 |
| `source_only_support_test` | 문제 없이 source에서 정답 근거가 충분한지 확인 | grounding 확인 |
| `repair_regression_test` | repair 전 fail tag가 repair 후 재발하지 않는지 확인 | 자동 수정 검증 |

| 대상 | 적용 |
| --- | --- |
| 모든 row | strict final multi-axis Judge |
| repaired row 전체 | metamorphic full test |
| 법령/해석례 row 전체 | answer-blind solve + source-span proof |
| 결정례/판결문 sample `20∼30%` | adversarial mutation |
| 랜덤 `10%` | full red-team Judge |

## bounded repair loop

Judge fail row는 무한 반복하지 않고, 실패 유형별 최대 시도와 lineage를 남긴다.

| 실패 유형 | 자동 repair 방식 | 최대 시도 | 최종 실패 시 |
| --- | --- | ---: | --- |
| `stem 어색함` | stem만 재작성 | `2` | `manual_review` |
| `형식 부적합` | 발문/선택지 길이·문장 다듬기 | `2` | `manual_review` |
| `오답약함` | 오답만 재생성 | `2` | quota surplus 대체 또는 `manual_review` |
| `정답 누설` | stem/정답 선택지 표현 완화 | `2` | `manual_review` |
| `해설 부족` | explanation/rubric 보강 | `2` | `manual_review` |
| `복수 쟁점 혼합` | stem을 단일 axis로 축소 | `1` | `quarantine` |
| `정답 비유일` | 선택지 전체 재구성 | `1` | `quarantine` |
| `오답이 정답 가능` | 해당 오답 교체 | `1` | `quarantine` |
| `근거 불일치` | 자동 수정 금지 또는 제한 `1회` | `0∼1` | `quarantine` |
| `metadata 충돌` | deterministic fix | `1` | `quarantine` |

repair prompt에는 아래 제약을 반드시 넣는다.

```text
원문 source의 법적 의미를 바꾸지 마라.
새로운 법적 사실을 추가하지 마라.
정답 label을 억지로 유지하지 마라.
수정 후에도 정답은 source 기준으로 유일해야 한다.
수정 대상 외의 필드는 가능한 한 보존하라.
```

수정본은 원본을 덮지 않고 `revision_manifest.csv` 또는 row-level revision field로 lineage를 남긴다.

## 구현 모듈 제안

| 모듈 | 역할 |
| --- | --- |
| `strict_final_judge_v1.py` | 객관식/서술형 multi-axis strict Judge 실행 |
| `stem_hygiene_gate.py` | 어색한 발문, 중복 명령, template 찌꺼기 deterministic 차단 |
| `answer_blind_solver.py` | 정답 label 없이 Judge가 직접 풀어 keyedness 검증 |
| `source_span_proof_judge.py` | 정답/오답별 source-span proof와 refutation 확인 |
| `explanation_quality_judge.py` | 객관식 해설과 서술형 rubric/모범답안 충분성 평가 |
| `metamorphic_validator.py` | choice shuffle, mutation fixture, leakage smoke test |
| `bounded_repair_loop.py` | 실패 row에만 제한적 repair와 rejudge 수행 |
| `judge_calibration_fixture.py` | known pass/fail fixture로 Judge 민감도 검산 |

## 최종 산출물 구조

```text
data/processed/aihub/problem_generation/dataset_candidates/
  kcc_2026_04_27_counted_snapshot_v1/
    objective/train.jsonl
    objective/dev.jsonl
    objective/test.jsonl
    descriptive/train.jsonl
    descriptive/dev.jsonl
    descriptive/test.jsonl
    manifests/dataset_manifest.csv
    manifests/package_source_manifest.csv
    manifests/count_inventory_snapshot.csv
    manifests/split_lock_manifest.csv
    reports/count_build_report.md
    reports/known_quality_debt.md

data/processed/aihub/problem_generation/dataset_candidates/
  kcc_2026_04_27_repaired_snapshot_v1/
    objective/train.jsonl
    objective/dev.jsonl
    objective/test.jsonl
    descriptive/train.jsonl
    descriptive/dev.jsonl
    descriptive/test.jsonl
    manifests/revision_manifest.csv
    manifests/validation_manifest.csv
    reports/repair_report.md

data/processed/aihub/problem_generation/final_dataset/
  kcc_2026_04_27_validated_v1/
    objective/train.jsonl
    objective/dev.jsonl
    objective/test.jsonl
    descriptive/train.jsonl
    descriptive/dev.jsonl
    descriptive/test.jsonl
    manifests/dataset_manifest.csv
    manifests/validation_manifest.csv
    manifests/package_source_manifest.csv
    reports/final_validation_report.md
    reports/type_balance_report.md
    reports/methodology_note.md
    reports/release_notes.md
```

## 논문/발표 프레이밍

```text
Evidence-Backed Dataset Compilation:
LLM-generated legal QA items with split-locked provenance,
bounded self-repair, and type-deficit overclosure.
```

핵심 기여는 package factory, count promotion contract, split-lock provenance,
type-deficit overclosure, evidence card, strict stem hygiene gate,
source-span proof Judge, bounded repair loop, metamorphic validation,
judge calibration fixture, release compiler로 설명한다.

## 지금 당장 실행 순서

| 순서 | 작업 |
| ---: | --- |
| `1` | P2 no-API sync 결과를 all-green으로 잠근다. |
| `2` | objective `48건` count reflection을 유지한다. |
| `3` | `problem_target_inventory.md` 최신 확보량/부족량 표를 source-of-truth로 잠근다. |
| `4` | `counted_snapshot_v1`을 빌드한다. |
| `5` | schema/stem/leakage deterministic gate를 실행한다. |
| `6` | strict final multi-axis Judge를 모든 row에 실행한다. |
| `7` | 법령/해석례 전체와 위험 row에 answer-blind/source-span Judge를 실행한다. |
| `8` | fail row를 `repair_planned`, `manual_review`, `quarantine`으로 분리한다. |
| `9` | bounded GPT repair를 실패 row에만 실행한다. |
| `10` | repaired row rejudge와 repair regression test를 실행한다. |
| `11` | metamorphic validation을 repaired row 전체와 sampled row에 실행한다. |
| `12` | `release_ready`, `manual_review`, `quarantine`을 확정한다. |
| `13` | repaired snapshot을 생성한다. |
| `14` | 최종 검증을 재실행한다. |
| `15` | `validated_v1` release를 생성한다. |
| `16` | `quality_validation_report.md`, `type_balance_report.md`, `methodology_note.md`, `judge_calibration_report.md`를 생성한다. |

## 다음 command sketch

```bash
python3 scripts/aihub/problem_generation/final_validation/run_kcc_dataset_release_pipeline.py \\
  --snapshot-id {snapshot_id} \\
  --release-id kcc_2026_04_27_validated_v1 \\
  --judge-workers 32 \\
  --repair-workers 32 \\
  --max-repair-attempts 2 \\
  --mode objective,descriptive \\
  --require-evidence-green \\
  --require-family-split-lock \\
  --write-paper-reports
```
"""
    (reports / "final_validation_judge_plan.md").write_text(validation_plan, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id", default="kcc_2026_04_27_counted_snapshot_v1")
    parser.add_argument(
        "--output-root",
        default=str(PROCESSED_ROOT / "dataset_candidates/kcc_2026_04_27_counted_snapshot_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_snapshot(args.snapshot_id, Path(args.output_root))


if __name__ == "__main__":
    main()
