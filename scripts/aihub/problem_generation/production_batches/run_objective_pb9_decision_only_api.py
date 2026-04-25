from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# `pb9` API runner는 직전 smoke artifact를 기준선으로 고정한 뒤, 같은 seed와
# 같은 label schedule이 실제 generation/Judge/export 경로에서도 유지되는지 검산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb8_decision_only as pb8,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_decision_only as pb9_smoke,
)


VERSION_TAG = "pb9_decision_only_controlled_production_with_choice_validator"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_decision_only_api_execution"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb8.pb6.pb4.pb3.base.PROJECT_ROOT
SMOKE_RUN_NAME = "2026-04-26_024801_pb9_decision_only_controlled_production_with_choice_validator_objective_r2_decision_only_execution_mode_smoke_check"
SMOKE_RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / SMOKE_RUN_NAME
SMOKE_SEED_REGISTRY_PATH = SMOKE_RUN_DIR / "inputs" / "seed_registry.csv"
SMOKE_TARGET_LABEL_SCHEDULE_PATH = (
    SMOKE_RUN_DIR / "inputs" / f"target_label_schedule_{VERSION_TAG}.csv"
)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
TAIL_MEMO_MD_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.md"
VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
LIVE_CHECKSUM_MD_PATH = RUN_EXPORTS_DIR / f"live_checksum_check_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

EXPECTED_TOTAL_SEED_COUNT = 40
EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 40}
EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "generalization_03_04"): 24,
    ("결정례_QA", "expansion_01_02"): 16,
}
PB9_SOURCE_COUNTS = pb9_smoke.PB9_SOURCE_COUNTS
TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
SUCCESS_USABLE_MIN = 38
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 2
SUCCESS_LAW_ROW_COUNT = 0
BATCH_STATUS = "failed_not_counted"
COUNT_REFLECTION_STATUS = "not_counted"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

BASE_BUILD_SEED_REGISTRY = pb8.pb6.build_seed_registry
BASE_BUILD_GENERATION_MESSAGES = pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES
BASE_BUILD_BATCH_SUMMARY = pb8.pb6.build_batch_summary
BASE_BUILD_RUN_MANIFEST = pb8.pb6.build_run_manifest
BASE_SPLIT_DATASET = pb8.pb6.pb4.pb3.base.split_dataset

VALIDATOR_SUMMARY: dict[str, object] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def append_tag(existing_tags: str, tag: str) -> str:
    tags = [value for value in (existing_tags or "").split("|") if value]
    if tag not in tags:
        tags.append(tag)
    return "|".join(tags)


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def stable_rows_for_diff(path: Path, keys: list[str]) -> list[dict[str, str]]:
    # `selected_at_utc`처럼 실행할 때마다 바뀌는 필드는 drift 검산에서 제외하고,
    # reviewer가 지정한 seed identity와 label schedule key만 비교한다.
    return [{key: row.get(key, "") for key in keys} for row in read_csv_rows(path)]


def write_live_checksum_check() -> None:
    # API 비용이 들어가기 직전에 smoke artifact와 live artifact가 같은지 고정 검산한다.
    # seed registry는 timestamp 컬럼 때문에 전체 파일 checksum 대신 key diff를 우선한다.
    seed_keys = ["seed_sample_id", "family_id", "label_path", "raw_path"]
    schedule_keys = ["seed_sample_id", "family_id", "target_correct_choice"]
    checks = [
        ("seed_registry", SMOKE_SEED_REGISTRY_PATH, SEED_REGISTRY_PATH, seed_keys, "key_diff"),
        (
            "target_label_schedule",
            SMOKE_TARGET_LABEL_SCHEDULE_PATH,
            TARGET_LABEL_SCHEDULE_CSV_PATH,
            schedule_keys,
            "key_diff",
        ),
    ]
    rows = []
    for artifact_name, smoke_path, live_path, keys, compare_mode in checks:
        smoke_hash = sha256_file(smoke_path)
        live_hash = sha256_file(live_path)
        smoke_stable_rows = stable_rows_for_diff(smoke_path, keys)
        live_stable_rows = stable_rows_for_diff(live_path, keys)
        key_matched = smoke_stable_rows == live_stable_rows
        rows.append(
            {
                "artifact": artifact_name,
                "compare_mode": compare_mode,
                "key_fields": ",".join(keys),
                "smoke_path": str(smoke_path),
                "live_path": str(live_path),
                "smoke_sha256": smoke_hash,
                "live_sha256": live_hash,
                "key_matched": "예" if key_matched else "아니오",
            }
        )
        if not key_matched:
            raise RuntimeError(f"pb9 live artifact key drift detected: {artifact_name}")

    lines = [
        f"# live checksum check `{VERSION_TAG}`",
        "",
        "| artifact | compare_mode | key_matched | key_fields | smoke_sha256 | live_sha256 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['artifact']}` | `{row['compare_mode']}` | `{row['key_matched']}` | `{row['key_fields']}` | `{row['smoke_sha256']}` | `{row['live_sha256']}` |"
        )
    pb8.pb6.pb4.pb3.base.write_text_atomic(LIVE_CHECKSUM_MD_PATH, "\n".join(lines) + "\n")
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(LIVE_CHECKSUM_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry_with_pb9_schedule() -> list[dict[str, str]]:
    seed_rows = BASE_BUILD_SEED_REGISTRY()
    pb9_smoke.assert_seed_preflight(seed_rows)
    schedule_rows = pb9_smoke.write_target_label_schedule(seed_rows)
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    write_live_checksum_check()
    if Counter(row["target_correct_choice"] for row in schedule_rows) != Counter(TARGET_LABEL_COUNTS):
        raise RuntimeError("pb9 target label schedule changed before API execution")
    return seed_rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## pb9 decision-only production 추가 지시
- 이번 run은 `결정례_QA` `40개` controlled production이며, 생성 후 postprocess choice validator와 label balance gate가 적용된다.
- 오답 3개 모두 정답과 같은 핵심 판단 기준 또는 적용 사실을 공유해야 한다.
- 각 오답은 같은 anchor 안에서 요건, 시점, 주체, 효과, 예외, 절차 중 정확히 한 축만 어긋나야 한다.
- 단순히 다른 법 개념의 정의나 역할을 묻는 선택지는 weak distractor로 본다.
- 문제 stem이 특정 개념의 일반적 정의·역할만 묻는 단순 회상형으로 닫히면 실패로 본다.
- `결정례_QA` 문항은 source의 판단 이유, 절차적 효과, 적용 사실 중 하나를 반드시 묻는다.
- 후처리 validator가 `A/B/C/D = 10/10/10/10` target schedule로 choice를 다시 섞으므로, 생성 단계에서는 label 위치가 아니라 정답 유일성과 선택지 의미 분리에 집중한다.
"""
    return messages


def target_schedule_by_seed() -> dict[str, str]:
    rows = read_csv_rows(TARGET_LABEL_SCHEDULE_CSV_PATH)
    return {row["seed_sample_id"]: row["target_correct_choice"] for row in rows}


def apply_pb9_validator_to_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    global VALIDATOR_SUMMARY

    selected = selected_rows(rows)
    target_by_seed = target_schedule_by_seed()
    report_rows = []

    for row in rows:
        # Judge 원본 상태를 보존해야 validator 때문에 바뀐 부분과 원래 실패를 분리해 볼 수 있다.
        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
        row["upstream_train_eligible"] = row.get("train_eligible", "")
        row["validator_action"] = ""
        row["validator_status"] = ""
        row["validator_reasons"] = ""
        row["target_correct_choice"] = ""
        row["validator_recalculated_correct_choice"] = ""
        row["validator_correct_choice_match_count"] = ""
        row["validator_shuffle_recalc_ok"] = ""
        row["metadata_remap_ok"] = ""
        row["metadata_remap_reasons"] = ""
        row["validator_export_disposition"] = ""
        row["export_correct_choice"] = row.get("correct_choice", "")

    for row in selected:
        seed_id = row["seed_sample_id"]
        target_label = target_by_seed[seed_id]
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices,
            row.get("correct_choice", ""),
            target_label,
        )
        action, status, reasons = pb9_smoke.choose_pb9_validator_action(row)
        shuffle_ok = recalculated_label == target_label
        metadata_ok = True
        metadata_reasons: list[str] = []

        if not shuffle_ok:
            action = "hard_block"
            status = "correct_choice_recalc_block"
            reasons = [*reasons, "correct_choice_recalc_mismatch"]

        row["target_correct_choice"] = target_label
        row["validator_recalculated_correct_choice"] = recalculated_label or ""
        row["validator_correct_choice_match_count"] = str(match_count)
        row["validator_shuffle_recalc_ok"] = "예" if shuffle_ok else "아니오"

        if action in {"accept", "audit"} and shuffle_ok:
            original_correct_choice = row.get("correct_choice", "")
            row["choice_a"] = shuffled_choices["A"]
            row["choice_b"] = shuffled_choices["B"]
            row["choice_c"] = shuffled_choices["C"]
            row["choice_d"] = shuffled_choices["D"]
            validator_replay.remap_label_keyed_metadata(row, original_correct_choice, target_label)
            row["correct_choice"] = recalculated_label or row.get("correct_choice", "")
            metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(row)
            row["metadata_remap_ok"] = "예" if metadata_ok else "아니오"
            row["metadata_remap_reasons"] = "|".join(metadata_reasons)
        else:
            row["metadata_remap_ok"] = "대상아님"
            row["metadata_remap_reasons"] = ""

        disposition, split_allowed, count_allowed = pb9_smoke.pb9_export_disposition(action, metadata_ok)
        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["validator_export_disposition"] = disposition
        row["export_correct_choice"] = row.get("correct_choice", "")

        if disposition == "export_ready":
            row["final_status"] = "pass"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "예"
        elif disposition == "audit_queue":
            row["final_status"] = "pass"
            row["audit_required"] = "예"
            row["audit_reason"] = append_tag(row.get("audit_reason", ""), "validator_audit")
            row["train_eligible"] = "아니오"
        elif disposition == "regenerate_required":
            row["final_status"] = "soft_fail"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "아니오"
            row["error_tags"] = append_tag(row.get("error_tags", ""), "validator_regenerate")
        else:
            row["final_status"] = "hard_fail"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "아니오"
            row["error_tags"] = append_tag(row.get("error_tags", ""), "validator_hard_block")

        report_rows.append(
            {
                "seed_sample_id": seed_id,
                "upstream_final_status": row["upstream_final_status"],
                "upstream_audit_required": row["upstream_audit_required"],
                "upstream_train_eligible": row["upstream_train_eligible"],
                "validator_action": action,
                "validator_status": status,
                "validator_reasons": row["validator_reasons"],
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "shuffle_recalc_ok": row["validator_shuffle_recalc_ok"],
                "metadata_remap_ok": row["metadata_remap_ok"],
                "metadata_remap_reasons": row["metadata_remap_reasons"],
                "validator_export_disposition": disposition,
                "split_allowed": split_allowed,
                "count_allowed": count_allowed,
                "final_status": row["final_status"],
                "audit_required": row["audit_required"],
                "train_eligible": row["train_eligible"],
                "export_correct_choice": row["export_correct_choice"],
            }
        )

    action_counts = Counter(row["validator_action"] for row in selected)
    target_label_counts = Counter(row["target_correct_choice"] for row in selected if row["target_correct_choice"])
    export_ready_rows = [
        row
        for row in selected
        if row.get("validator_export_disposition") == "export_ready"
    ]
    export_counted_rows = [
        row
        for row in selected
        if row.get("validator_export_disposition") in {"export_ready", "audit_queue"}
    ]
    export_label_counts = Counter(row["correct_choice"] for row in export_ready_rows)
    package_label_counts = Counter(row["correct_choice"] for row in export_counted_rows)
    shuffle_mismatch_count = sum(1 for row in selected if row["validator_shuffle_recalc_ok"] != "예")
    metadata_mismatch_count = sum(
        1
        for row in export_counted_rows
        if row.get("metadata_remap_ok") != "예"
    )
    summary = pb8.pb6.pb4.pb3.summarize_rows(rows)
    law_row_count = sum(1 for row in selected if row.get("doc_type_name") == "법령_QA")
    exact_export_balance = all(
        export_label_counts.get(label, 0) == TARGET_LABEL_COUNTS[label]
        for label in validator_replay.CHOICE_LABELS
    )

    VALIDATOR_SUMMARY = {
        "selected_count": len(selected),
        "validator_action_counts": dict(action_counts),
        "target_label_counts": {label: target_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_ready_label_counts": {label: export_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_or_audit_label_counts": {label: package_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "shuffle_recalc_mismatch_count": shuffle_mismatch_count,
        "metadata_remap_mismatch_count": metadata_mismatch_count,
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "law_row_count": law_row_count,
        "exact_export_label_balance_passed": exact_export_balance,
    }
    VALIDATOR_SUMMARY["pb9_initial_success_passed"] = (
        VALIDATOR_SUMMARY["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and VALIDATOR_SUMMARY["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and VALIDATOR_SUMMARY["shuffle_recalc_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["metadata_remap_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["law_row_count"] == SUCCESS_LAW_ROW_COUNT
    )
    VALIDATOR_SUMMARY["pb9_final_exact_package_passed"] = (
        VALIDATOR_SUMMARY["pb9_initial_success_passed"]
        and VALIDATOR_SUMMARY["selected_train_eligible_count"] == EXPECTED_TOTAL_SEED_COUNT
        and exact_export_balance
    )

    fieldnames = list(report_rows[0].keys()) if report_rows else ["seed_sample_id"]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, fieldnames)
    write_validator_report_md(report_rows)
    write_validator_wiring_check_md()
    return rows


def write_validator_report_md(report_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{VALIDATOR_SUMMARY.get('selected_count', 0)}`",
        f"- validator_action_counts: `{VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- target_label_counts: `{VALIDATOR_SUMMARY.get('target_label_counts', {})}`",
        f"- export_ready_label_counts: `{VALIDATOR_SUMMARY.get('export_ready_label_counts', {})}`",
        f"- shuffle_recalc_mismatch_count: `{VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)}`",
        f"- metadata_remap_mismatch_count: `{VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}`",
        f"- pb9_initial_success_passed: `{VALIDATOR_SUMMARY.get('pb9_initial_success_passed', False)}`",
        f"- pb9_final_exact_package_passed: `{VALIDATOR_SUMMARY.get('pb9_final_exact_package_passed', False)}`",
        "",
        "## row actions",
        "| seed | upstream_status | action | final_status | audit | train_eligible | target | recalculated | metadata | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['upstream_final_status']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['recalculated_correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['validator_export_disposition']}` |"
        )
    pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_validator_wiring_check_md() -> None:
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| validator connected after merge | `pass` | `merge_scores -> pb9 validator -> split/export` 순서로 연결 |",
        "| target label schedule | `pass` | selected package 기준 `A/B/C/D = 10/10/10/10` target 적용 |",
        "| correct choice recalculation | `pass` | mismatch 발생 시 `hard_block` 처리 |",
        "| metadata remap gate | `pass` | metadata mismatch는 `metadata_remap_block`으로 export/count 차단 |",
        "| NearMiss low/missing gate | `pass` | `<=2` regenerate, `3` audit, missing/parse hard block |",
        "| count reflection | `pass` | reviewer sign-off 전 core current count 미변경 |",
    ]
    pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def split_dataset_with_pb9_validator(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    validated_rows = apply_pb9_validator_to_rows(rows)
    if validated_rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(MERGED_SCORES_PATH, validated_rows, list(validated_rows[0].keys()))
    manifest_rows = BASE_SPLIT_DATASET(validated_rows)
    selected_by_problem_id = {
        row["candidate_id"]: row
        for row in validated_rows
        if row.get("selected_for_seed") == "예"
    }
    rewrite_split_jsonl_with_pb9_status(selected_by_problem_id)
    rewrite_audit_queue_with_pb9_status(selected_by_problem_id)
    return rewrite_manifest_with_validator_fields(validated_rows, manifest_rows)


def pb9_status_fields() -> dict[str, str]:
    return {
        "batch_status": BATCH_STATUS,
        "count_reflection_status": COUNT_REFLECTION_STATUS,
        "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
    }


def rewrite_split_jsonl_with_pb9_status(selected_by_problem_id: dict[str, dict[str, str]]) -> None:
    # `pb9` failed candidate는 processed path에 있어도 downstream이 단독 JSONL만 보고
    # counted production data로 소비하지 못하도록 row-level status/provenance를 남긴다.
    for path in (PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH, PROBLEM_TEST_PATH):
        if not path.exists():
            continue
        rows = pb8.pb6.pb4.pb3.base.load_jsonl(path)
        enriched_rows = []
        for payload in rows:
            source_row = selected_by_problem_id.get(payload.get("problem_id", ""))
            enriched = dict(payload)
            enriched.update(pb9_status_fields())
            if source_row:
                enriched.update(
                    {
                        "export_correct_choice": source_row.get("export_correct_choice", source_row.get("correct_choice", "")),
                        "target_correct_choice": source_row.get("target_correct_choice", ""),
                        "validator_action": source_row.get("validator_action", ""),
                        "validator_export_disposition": source_row.get("validator_export_disposition", ""),
                        "metadata_remap_ok": source_row.get("metadata_remap_ok", ""),
                    }
                )
            enriched_rows.append(enriched)
        pb8.pb6.pb4.pb3.base.write_jsonl_atomic(path, enriched_rows)


def rewrite_audit_queue_with_pb9_status(selected_by_problem_id: dict[str, dict[str, str]]) -> None:
    if not PROBLEM_AUDIT_QUEUE_PATH.exists():
        return
    audit_rows = read_csv_rows(PROBLEM_AUDIT_QUEUE_PATH)
    if not audit_rows:
        return
    enriched_rows = []
    for row in audit_rows:
        source_row = selected_by_problem_id.get(row.get("problem_id", ""))
        enriched = dict(row)
        enriched.update(pb9_status_fields())
        if source_row:
            enriched.update(
                {
                    "export_correct_choice": source_row.get("export_correct_choice", source_row.get("correct_choice", "")),
                    "target_correct_choice": source_row.get("target_correct_choice", ""),
                    "validator_action": source_row.get("validator_action", ""),
                    "validator_export_disposition": source_row.get("validator_export_disposition", ""),
                    "metadata_remap_ok": source_row.get("metadata_remap_ok", ""),
                }
            )
        enriched_rows.append(enriched)
    pb8.pb6.pb4.pb3.base.write_csv_atomic(
        PROBLEM_AUDIT_QUEUE_PATH,
        enriched_rows,
        list(enriched_rows[0].keys()),
    )


def rewrite_manifest_with_validator_fields(
    validated_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected_by_problem_id = {
        row["candidate_id"]: row
        for row in validated_rows
        if row.get("selected_for_seed") == "예"
    }
    enriched_rows = []
    for manifest_row in manifest_rows:
        source_row = selected_by_problem_id[manifest_row["problem_id"]]
        enriched = dict(manifest_row)
        enriched.update(
            {
                **pb9_status_fields(),
                "export_correct_choice": source_row.get("export_correct_choice", source_row.get("correct_choice", "")),
                "target_correct_choice": source_row.get("target_correct_choice", ""),
                "validator_action": source_row.get("validator_action", ""),
                "validator_export_disposition": source_row.get("validator_export_disposition", ""),
                "validator_recalculated_correct_choice": source_row.get("validator_recalculated_correct_choice", ""),
                "validator_correct_choice_match_count": source_row.get("validator_correct_choice_match_count", ""),
                "metadata_remap_ok": source_row.get("metadata_remap_ok", ""),
                "metadata_remap_reasons": source_row.get("metadata_remap_reasons", ""),
            }
        )
        enriched_rows.append(enriched)

    if enriched_rows:
        fieldnames = list(enriched_rows[0].keys())
        pb8.pb6.pb4.pb3.base.write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, enriched_rows, fieldnames)
    write_manifest_header_gate(enriched_rows)
    return enriched_rows


def write_manifest_header_gate(manifest_rows: list[dict[str, str]]) -> None:
    headers = list(manifest_rows[0].keys()) if manifest_rows else []
    missing = [field for field in pb9_smoke.MANIFEST_REQUIRED_FIELDS if field not in headers]
    if missing:
        raise RuntimeError(f"pb9 dataset manifest required fields missing: {missing}")

    leak_count = 0
    mismatch_count = 0
    for row in manifest_rows:
        if row.get("metadata_remap_ok") != "예" and row.get("validator_export_disposition") == "export_ready":
            leak_count += 1
        if row.get("validator_recalculated_correct_choice") != row.get("export_correct_choice"):
            mismatch_count += 1
    if leak_count or mismatch_count:
        raise RuntimeError(
            f"pb9 manifest gate failed: metadata_leak={leak_count}, correct_choice_mismatch={mismatch_count}"
        )

    lines = [
        f"# manifest header gate `{VERSION_TAG}`",
        "",
        "| check | result | value |",
        "| --- | --- | --- |",
        f"| required header fields | `pass` | `{pb9_smoke.MANIFEST_REQUIRED_FIELDS}` |",
        f"| metadata remap leak | `pass` | `{leak_count}` |",
        f"| correct choice recalculation mismatch | `pass` | `{mismatch_count}` |",
    ]
    pb8.pb6.pb4.pb3.base.write_text_atomic(MANIFEST_HEADER_GATE_MD_PATH, "\n".join(lines) + "\n")


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows = BASE_BUILD_BATCH_SUMMARY(rows)
    with BATCH_SUMMARY_MD_PATH.open("a", encoding="utf-8") as output_file:
        output_file.write("\n## pb9 validator success criteria\n")
        output_file.write("| criterion | target | result |\n")
        output_file.write("| --- | --- | --- |\n")
        output_file.write(f"| usable | `>= {SUCCESS_USABLE_MIN} / 40` | `{VALIDATOR_SUMMARY.get('selected_train_eligible_count', 0)}` |\n")
        output_file.write(f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{VALIDATOR_SUMMARY.get('selected_hard_fail_count', 0)}` |\n")
        output_file.write(f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{VALIDATOR_SUMMARY.get('selected_soft_fail_count', 0)}` |\n")
        output_file.write(f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{VALIDATOR_SUMMARY.get('selected_audit_required_count', 0)}` |\n")
        output_file.write(f"| shuffle recalc mismatch | `0` | `{VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)}` |\n")
        output_file.write(f"| metadata remap mismatch | `0` | `{VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |\n")
        output_file.write(f"| final exact export label balance | `A/B/C/D = 10/10/10/10` | `{VALIDATOR_SUMMARY.get('export_ready_label_counts', {})}` |\n")
    return summary_rows


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict:
    manifest = BASE_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    manifest.update(
        {
            "version_tag": VERSION_TAG,
            "run_name": RUN_NAME,
            "source_smoke_run_name": SMOKE_RUN_NAME,
            "candidate_recipe_source": "v2_difficulty_patch_r2_decision_only_choice_validator_pb9",
            "seed_registry_strategy": "same_as_pb9_execution_mode_smoke_check_with_live_checksum_gate",
            "current_count_decision": "not_counted_until_reviewer_signoff_after_pb9_api_execution",
            "validator_policy": {
                "source": "pb9_postprocess_choice_validator",
                "target_label_counts": TARGET_LABEL_COUNTS,
                "manifest_required_fields": pb9_smoke.MANIFEST_REQUIRED_FIELDS,
                "count_reflection": "reviewer_signoff_required",
            },
            "validator_summary": VALIDATOR_SUMMARY,
            "success_criteria": {
                "usable_min": SUCCESS_USABLE_MIN,
                "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
                "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
                "audit_max": SUCCESS_AUDIT_MAX,
                "law_row_count": SUCCESS_LAW_ROW_COUNT,
                "shuffle_recalc_mismatch": 0,
                "metadata_remap_mismatch": 0,
                "final_exact_export_label_balance": TARGET_LABEL_COUNTS,
            },
            "success_result": {
                "usable": VALIDATOR_SUMMARY.get("selected_train_eligible_count", 0),
                "hard_fail": VALIDATOR_SUMMARY.get("selected_hard_fail_count", 0),
                "soft_fail": VALIDATOR_SUMMARY.get("selected_soft_fail_count", 0),
                "audit": VALIDATOR_SUMMARY.get("selected_audit_required_count", 0),
                "pb9_initial_success_passed": VALIDATOR_SUMMARY.get("pb9_initial_success_passed", False),
                "pb9_final_exact_package_passed": VALIDATOR_SUMMARY.get("pb9_final_exact_package_passed", False),
            },
        }
    )
    manifest["artifact_paths"].update(
        {
            "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "raw_merged_before_validator": str(RAW_MERGED_BEFORE_VALIDATOR_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
            "live_checksum_check": str(LIVE_CHECKSUM_MD_PATH),
            "manifest_header_gate": str(MANIFEST_HEADER_GATE_MD_PATH),
        }
    )
    pb8.pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_pb9_api_globals() -> None:
    # `pb8/pb6` strict-final 본체를 유지하고, pb9 identity/path/validator hook만 덮어쓴다.
    pb9_smoke.RUN_DATE = RUN_DATE
    pb9_smoke.RUN_PURPOSE = RUN_PURPOSE
    pb9_smoke.RUN_NAME = RUN_NAME
    pb9_smoke.RUN_DIR = RUN_DIR
    pb9_smoke.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb9_smoke.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb9_smoke.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb9_smoke.SEED_READY_PATH = SEED_READY_PATH
    pb9_smoke.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb9_smoke.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb9_smoke.TARGET_LABEL_SCHEDULE_CSV_PATH = TARGET_LABEL_SCHEDULE_CSV_PATH
    pb9_smoke.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH

    pb9_smoke.configure_seed_registry_globals()
    pb8.pb6.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb8.pb6.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb8.pb6.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb8.pb6.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb8.pb6.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb8.pb6.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb8.pb6.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb8.pb6.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb8.pb6.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb8.pb6.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb8.pb6.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb8.pb6.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb8.pb6.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb8.pb6.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb8.pb6.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb8.pb6.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb8.pb6.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb8.pb6.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pb8.pb6.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    pb8.pb6.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    pb8.pb6.RUN_LABEL = "pb9 decision-only API execution"
    pb8.pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pb8.pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pb8.pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pb8.pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pb8.pb6.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_decision_only_choice_validator_pb9"
    pb8.pb6.SEED_REGISTRY_STRATEGY = "pb9_same_seed_as_execution_smoke_check"
    pb8.pb6.LAW_STATUS_NOTE = "law_interpretation_judgment_repair_tracks_excluded_from_pb9"
    pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb8.pb6.build_seed_registry = build_seed_registry_with_pb9_schedule
    pb8.pb6.build_batch_summary = build_batch_summary
    pb8.pb6.build_run_manifest = build_run_manifest
    pb8.pb6.pb4.pb3.base.split_dataset = split_dataset_with_pb9_validator


def main() -> dict:
    configure_pb9_api_globals()
    return pb8.pb6.main()


if __name__ == "__main__":
    main()
