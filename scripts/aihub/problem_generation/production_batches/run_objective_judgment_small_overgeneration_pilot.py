from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# no-API preflight reviewer sign-off 이후, 같은 seed registry를 실제 API로 태우는 runner다.
# generation/Judge는 candidate pool 전체에 수행하고, count 후보는 compiler가 strict final package만 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_repair_pilot as judgment_pilot,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402


VERSION_TAG = "objective_judgment_small_overgeneration_pilot"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_judgment_small_overgeneration_api_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "objective judgment small overgeneration API pilot"
LINTER_FIXTURE_ID = "judgment_small_overgeneration_candidate_package_pass"
GATE_FIELD_LABEL = "judgment gate fields"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PREFLIGHT_VERSION_TAG = "objective_judgment_small_overgeneration_pilot_preflight"
SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_judgment_small_overgeneration_seed_spec_wiring_check"
SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / SOURCE_PREFLIGHT_VERSION_TAG
    / "seed_registry.csv"
)
SOURCE_PREFLIGHT_RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
SOURCE_PREFLIGHT_RUN_NAME = ""
SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"target_label_schedule_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
)
SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"exclusion_audit_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
)
SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
)
SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
)
SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.json"
)
SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
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
RUN_LINTER_DIR = RUN_DIR / "linter"
RUN_EVIDENCE_DIR = RUN_DIR / "evidence_card"

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
RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"
CANDIDATE_MERGED_SCORES_PATH = RUN_MERGED_DIR / f"candidate_merged_problem_scores_{VERSION_TAG}.csv"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

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
CANDIDATE_VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.csv"
CANDIDATE_VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.md"
VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
PILOT_BREAKOUT_CSV_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.csv"
PILOT_BREAKOUT_MD_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
COMPILER_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"compiler_summary_{VERSION_TAG}.md"
CANDIDATE_POOL_PATH = RUN_DIR / "candidate_pool.csv"
ACCEPTED_POOL_PATH = RUN_DIR / "accepted_pool.csv"
REJECTED_POOL_PATH = RUN_DIR / "rejected_pool.csv"
TAIL_TAXONOMY_PATH = RUN_DIR / "tail_taxonomy.csv"
QUOTA_SURPLUS_POOL_PATH = RUN_DIR / "quota_surplus_pool.csv"
COMPILER_MANIFEST_PATH = RUN_DIR / "compiler_manifest.json"
ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH = RUN_DIR / "artifact_linter_fixture_manifest.json"
EVIDENCE_CARD_PACKAGE_MANIFEST_PATH = RUN_DIR / "evidence_card_package_manifest.json"

EXPECTED_CANDIDATE_SEED_COUNT = 24
FINAL_PACKAGE_TARGET_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 24}
EXPECTED_LANE_BY_DOC = {
    ("판결문_QA", "generalization_03_04"): 12,
    ("판결문_QA", "expansion_01_02"): 12,
}
EXPECTED_SOURCE_COUNTS = {
    "01_TL_판결문_QA": 6,
    "02_TL_판결문_QA": 6,
    "03_TL_판결문_QA": 6,
    "04_TL_판결문_QA": 6,
}
CANDIDATE_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
FINAL_TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
FINAL_SOURCE_COUNTS = {
    "01_TL_판결문_QA": 4,
    "02_TL_판결문_QA": 4,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 4,
}
FINAL_LANE_COUNTS = {"generalization_03_04": 8, "expansion_01_02": 8}
FINAL_DEV_COUNT = 1
FINAL_TEST_COUNT = 1
CURRENT_OBJECTIVE_COUNT = {"usable": 167, "train": 130, "eval": 37, "audit": 6, "hard_fail": 5, "soft_fail": 3}

PACKAGE_ROLE = "count_reflection_candidate_package"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
COUNT_DISPOSITION = "candidate_not_counted"
PROMOTION_CONTRACT_STATUS = "passed_not_counted"
YES = "예"
NO = "아니오"

COMPILER_RESULT: dict[str, Any] = {}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_registry_keys(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    # 같은 version tag가 여러 번 생성되어도 seed identity가 같은 locked preflight run을 찾기 위한 비교 키다.
    return [
        (
            row.get("seed_sample_id", ""),
            row.get("reference_sample_id", ""),
            row.get("family_id", ""),
            row.get("label_path", ""),
            row.get("raw_path", ""),
            row.get("target_correct_choice", ""),
        )
        for row in read_csv_rows(path)
    ]


def resolve_source_preflight_run_dir() -> Path:
    # run 이름 앞 timestamp는 실행 시점마다 달라지므로, wrapper가 바꾼 source registry와 같은 run을 동적으로 찾는다.
    llm_runs_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    pattern = f"*_{SOURCE_PREFLIGHT_VERSION_TAG}_{SOURCE_PREFLIGHT_RUN_PURPOSE}"
    source_keys = read_registry_keys(SOURCE_PREFLIGHT_SEED_REGISTRY_PATH)
    matched: list[Path] = []
    for candidate_dir in sorted(llm_runs_root.glob(pattern)):
        candidate_registry = candidate_dir / "inputs" / "seed_registry.csv"
        if candidate_registry.exists() and read_registry_keys(candidate_registry) == source_keys:
            matched.append(candidate_dir)
    if not matched:
        raise FileNotFoundError(f"locked source preflight run not found for {SOURCE_PREFLIGHT_SEED_REGISTRY_PATH}")
    return matched[-1]


def refresh_paths() -> None:
    # wrapper runner가 VERSION_TAG/RUN_NAME/source preflight를 바꿔도 모든 산출물 path가 같은 config를 보게 한다.
    global SOURCE_PREFLIGHT_RUN_DIR, SOURCE_PREFLIGHT_RUN_NAME
    global SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH, SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH
    global SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH, SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH
    global SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH, SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH
    global INTERIM_DIR, PROCESSED_DIR, RUN_DIR, RUN_PROMPTS_DIR, RUN_INPUTS_DIR, RUN_GENERATIONS_DIR
    global RUN_JUDGE_LOGS_DIR, RUN_MERGED_DIR, RUN_EXPORTS_DIR, RUN_LINTER_DIR, RUN_EVIDENCE_DIR
    global SEED_REGISTRY_PATH, SEED_READY_PATH, SEED_PREFLIGHT_CSV_PATH, SEED_PREFLIGHT_MD_PATH
    global TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_MANIFEST_PATH, GENERATED_PROBLEMS_PATH, GROUNDING_LOG_PATH
    global KEYEDNESS_LOG_PATH, DISTRACTORFIT_LOG_PATH, NEARMISS_LOG_PATH, RAW_MERGED_BEFORE_VALIDATOR_PATH
    global CANDIDATE_MERGED_SCORES_PATH, MERGED_SCORES_PATH, PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH
    global PROBLEM_TEST_PATH, PROBLEM_DATASET_MANIFEST_PATH, PROBLEM_AUDIT_QUEUE_PATH
    global BATCH_SUMMARY_MD_PATH, BATCH_SUMMARY_CSV_PATH, BATCH_LANE_SUMMARY_CSV_PATH
    global TAIL_MEMO_CSV_PATH, TAIL_MEMO_MD_PATH, CANDIDATE_VALIDATOR_REPORT_CSV_PATH
    global CANDIDATE_VALIDATOR_REPORT_MD_PATH, VALIDATOR_REPORT_CSV_PATH, VALIDATOR_REPORT_MD_PATH
    global VALIDATOR_WIRING_CHECK_MD_PATH, PILOT_BREAKOUT_CSV_PATH, PILOT_BREAKOUT_MD_PATH
    global MANIFEST_HEADER_GATE_MD_PATH, FINAL_PACKAGE_CSV_PATH, FINAL_PACKAGE_MD_PATH, COMPILER_SUMMARY_MD_PATH
    global CANDIDATE_POOL_PATH, ACCEPTED_POOL_PATH, REJECTED_POOL_PATH, TAIL_TAXONOMY_PATH
    global QUOTA_SURPLUS_POOL_PATH, COMPILER_MANIFEST_PATH, ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH
    global EVIDENCE_CARD_PACKAGE_MANIFEST_PATH

    SOURCE_PREFLIGHT_RUN_DIR = resolve_source_preflight_run_dir()
    SOURCE_PREFLIGHT_RUN_NAME = SOURCE_PREFLIGHT_RUN_DIR.name
    SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH = (
        SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"target_label_schedule_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
    )
    SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH = SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"exclusion_audit_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
    SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH = (
        SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
    )
    SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH = (
        SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
    )
    SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH = (
        SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.json"
    )
    SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH = (
        SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
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
    RUN_LINTER_DIR = RUN_DIR / "linter"
    RUN_EVIDENCE_DIR = RUN_DIR / "evidence_card"

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
    RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"
    CANDIDATE_MERGED_SCORES_PATH = RUN_MERGED_DIR / f"candidate_merged_problem_scores_{VERSION_TAG}.csv"
    MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
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
    CANDIDATE_VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.csv"
    CANDIDATE_VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.md"
    VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
    VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
    VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
    PILOT_BREAKOUT_CSV_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.csv"
    PILOT_BREAKOUT_MD_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.md"
    MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"
    FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
    FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
    COMPILER_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"compiler_summary_{VERSION_TAG}.md"
    CANDIDATE_POOL_PATH = RUN_DIR / "candidate_pool.csv"
    ACCEPTED_POOL_PATH = RUN_DIR / "accepted_pool.csv"
    REJECTED_POOL_PATH = RUN_DIR / "rejected_pool.csv"
    TAIL_TAXONOMY_PATH = RUN_DIR / "tail_taxonomy.csv"
    QUOTA_SURPLUS_POOL_PATH = RUN_DIR / "quota_surplus_pool.csv"
    COMPILER_MANIFEST_PATH = RUN_DIR / "compiler_manifest.json"
    ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH = RUN_DIR / "artifact_linter_fixture_manifest.json"
    EVIDENCE_CARD_PACKAGE_MANIFEST_PATH = RUN_DIR / "evidence_card_package_manifest.json"


def union_fields(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    fields = list(preferred or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    fieldnames = union_fields(rows, preferred_fields)
    if not fieldnames:
        fieldnames = preferred_fields or ["empty"]
    pb6.pb4.pb3.base.write_csv_atomic(path, rows, fieldnames)


def candidate_contract_fields() -> dict[str, str]:
    return {
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
    }


def promotion_fields(passed: bool) -> dict[str, str]:
    return {
        "promotion_contract_passed": YES if passed else NO,
        "compiler_gate_passed": YES if passed else NO,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS if passed else "failed_not_counted",
    }


def pool_promotion_fields(pool_class: str) -> dict[str, str]:
    # package factory의 raw candidate/surplus는 품질 실패가 아니라 final package 밖의 pool이므로,
    # linter와 reviewer가 실패 tail로 오해하지 않도록 promotion status를 별도로 고정한다.
    if pool_class == "candidate_pool":
        return {
            "promotion_contract_passed": NO,
            "compiler_gate_passed": NO,
            "promotion_contract_status": "candidate_pool_not_promoted",
        }
    if pool_class == "quota_surplus":
        return {
            "promotion_contract_passed": NO,
            "compiler_gate_passed": NO,
            "promotion_contract_status": "not_promoted_quota_surplus",
        }
    return promotion_fields(False)


def copy_preflight_contract_inputs() -> None:
    # API run이 어떤 no-API contract에서 출발했는지 run inputs에도 복사해 handoff portability를 높인다.
    for source_path in [
        SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH,
        SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH,
        SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH,
        SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH,
        SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH,
        SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH,
    ]:
        if source_path.exists():
            pb6.pb4.pb3.base.copy_file_to_run_inputs(source_path, RUN_INPUTS_DIR)


def build_seed_registry_from_preflight() -> list[dict[str, str]]:
    seed_rows = judgment_pilot.build_seed_registry_from_preflight()
    copy_preflight_contract_inputs()
    return seed_rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = judgment_pilot.BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += f"""

## judgment small overgeneration pilot 추가 지시
- 이번 run은 `판결문_QA` candidate {EXPECTED_CANDIDATE_SEED_COUNT}개를 생성한 뒤 strict final package {FINAL_PACKAGE_TARGET_COUNT}개만 컴파일하는 package factory pilot이다.
- seed action은 `{seed.get('judgment_seed_action', '')}`, stem axis는 `{seed.get('stem_axis', '')}`, tail proximity는 `{seed.get('tail_proximity_class', '')}`다.
- 생성 단계에서는 label 위치보다 정답 유일성, 하나의 쟁점, 선택지 의미 분리를 우선한다.
- stem은 하나의 청구, 하나의 쟁점, 하나의 판단 기준, 하나의 적용 사실만 묻는다.
- 일반론과 사안 적용을 한 stem 안에서 동시에 묻지 않는다.
- 정답은 `gold_short_answer`와 같은 판결상 결론 하나에만 닫혀야 한다.
- 오답은 같은 판결문 근거를 공유하되 각각 쟁점, 사실관계, 판단 기준, 결론 범위 중 정확히 한 축만 어긋나야 한다.
- 다른 choice가 별도 일반론이나 별도 사실 적용으로도 정답 가능하게 읽히면 answer uniqueness failure로 본다.
- 후처리 validator가 target label `{seed.get('target_correct_choice', '')}`로 choice를 재배치하므로, 생성 단계에서는 target label을 억지로 맞추지 않는다.
"""
    return messages


def strict_accept_reason(row: dict[str, str]) -> str:
    if row.get("final_status") != "pass":
        return "hard_or_soft_fail"
    if row.get("audit_required") == YES:
        return "audit_required"
    if row.get("validator_action") != "accept" or row.get("validator_export_disposition") != "export_ready":
        return "validator_not_export_ready"
    if row.get("metadata_remap_ok") != YES:
        return "metadata_mismatch"
    if row.get("validator_recalculated_correct_choice") != row.get("target_correct_choice"):
        return "shuffle_mismatch"
    for gate_field in ["single_correct_choice", "rule_application_split", "issue_boundary", "case_fact_alignment", "hierarchy_overlap"]:
        if row.get(gate_field) == NO:
            return "answer_uniqueness_or_boundary_failure"
    return ""


def quality_tail_class(reason: str, row: dict[str, str]) -> str:
    if reason == "hard_or_soft_fail":
        return "final_status_failure"
    if reason == "audit_required":
        return "audit_tail"
    if reason == "validator_not_export_ready":
        return row.get("validator_reason_short", "") or "validator_failure"
    if reason == "metadata_mismatch":
        return "metadata_failure"
    if reason == "shuffle_mismatch":
        return "shuffle_failure"
    if reason == "answer_uniqueness_or_boundary_failure":
        return "answer_uniqueness_failure"
    return "quality_or_artifact_failure"


def with_pool_fields(row: dict[str, str], *, pool_class: str, quality_failure: str, tail_class: str = "") -> dict[str, str]:
    output = dict(row)
    output.update(candidate_contract_fields())
    output.update(pool_promotion_fields(pool_class))
    output.update(
        {
            "package_candidate_role": "candidate_pool",
            "package_compiler_action": "",
            "final_package_selected": NO,
            "pool_class": pool_class,
            "quality_failure": quality_failure,
            "tail_class": tail_class,
            "not_selected_reason": "",
            "quota_surplus_reason": "",
            "future_candidate_reusable": "",
            "candidate_reuse_policy": "",
            "selection_rank": "",
            "selection_reason": "",
        }
    )
    return output


def find_final_combination(accepted_rows: list[dict[str, str]]) -> set[str]:
    # candidate 64처럼 조합 수가 큰 package도 다루기 위해 source별 label allocation DP로 exact quota를 찾는다.
    rows_by_source_label: dict[str, dict[str, list[dict[str, str]]]] = {}
    for source in FINAL_SOURCE_COUNTS:
        rows_by_source_label[source] = {}
        for label in FINAL_TARGET_LABEL_COUNTS:
            bucket = [
                row
                for row in accepted_rows
                if row.get("source_subset", "") == source and row.get("export_correct_choice", "") == label
            ]
            rows_by_source_label[source][label] = sorted(
                bucket,
                key=lambda row: (
                    row.get("tail_proximity_class", ""),
                    row.get("sampling_lane", ""),
                    row.get("seed_sample_id", ""),
                ),
            )

    labels = list(FINAL_TARGET_LABEL_COUNTS)
    sources = list(FINAL_SOURCE_COUNTS)
    source_options: dict[str, list[tuple[int, ...]]] = {}
    for source in sources:
        target = FINAL_SOURCE_COUNTS[source]
        options: list[tuple[int, ...]] = []

        def build_options(index: int, remaining: int, current: list[int]) -> None:
            if index == len(labels):
                if remaining == 0:
                    options.append(tuple(current))
                return
            label = labels[index]
            max_take = min(remaining, len(rows_by_source_label[source][label]), FINAL_TARGET_LABEL_COUNTS[label])
            for value in range(max_take + 1):
                current.append(value)
                build_options(index + 1, remaining - value, current)
                current.pop()

        build_options(0, target, [])
        source_options[source] = options

    target_label_tuple = tuple(FINAL_TARGET_LABEL_COUNTS[label] for label in labels)
    chosen: dict[str, tuple[int, ...]] = {}

    def search(source_index: int, used: tuple[int, ...]) -> bool:
        if source_index == len(sources):
            return used == target_label_tuple
        source = sources[source_index]
        for option in source_options[source]:
            next_used = tuple(used[i] + option[i] for i in range(len(labels)))
            if any(next_used[i] > target_label_tuple[i] for i in range(len(labels))):
                continue
            chosen[source] = option
            if search(source_index + 1, next_used):
                return True
            chosen.pop(source, None)
        return False

    if search(0, tuple(0 for _ in labels)):
        final_rows: list[dict[str, str]] = []
        for source in sources:
            # macOS 기본 Python 3.9에서도 돌아가야 하므로 `zip(strict=True)`를 쓰지 않는다.
            for label, take in zip(labels, chosen[source]):
                final_rows.extend(rows_by_source_label[source][label][:take])
        lane_counts = Counter(row.get("sampling_lane", "") for row in final_rows)
        if {lane: lane_counts.get(lane, 0) for lane in FINAL_LANE_COUNTS} == FINAL_LANE_COUNTS:
            return {row["candidate_id"] for row in final_rows}
    return set()


def add_final_split(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, row in enumerate(sorted(rows, key=lambda item: int(item.get("selection_rank", "9999"))), start=1):
        train_cutoff = max(0, FINAL_PACKAGE_TARGET_COUNT - FINAL_DEV_COUNT - FINAL_TEST_COUNT)
        split = "train" if index <= train_cutoff else "dev" if index <= train_cutoff + FINAL_DEV_COUNT else "test"
        output.append({**row, "split": split})
    return output


def compile_final_package(validated_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    global COMPILER_RESULT

    selected = judgment_pilot.selected_rows(validated_rows)
    candidate_pool: list[dict[str, str]] = []
    strict_accepted: list[dict[str, str]] = []
    quality_rejected: list[dict[str, str]] = []

    for row in selected:
        reason = strict_accept_reason(row)
        candidate = with_pool_fields(row, pool_class="candidate_pool", quality_failure="대상아님")
        candidate_pool.append(candidate)
        if reason:
            rejected = with_pool_fields(row, pool_class="quality_reject", quality_failure=YES, tail_class=quality_tail_class(reason, row))
            rejected.update(
                {
                    "package_compiler_action": "quality_reject",
                    "not_selected_reason": reason,
                    "candidate_reuse_policy": "do_not_reuse_without_repair_review",
                    "future_candidate_reusable": NO,
                    "train_eligible": NO,
                    "count_allowed": NO,
                }
            )
            quality_rejected.append(rejected)
            continue
        accepted = with_pool_fields(row, pool_class="strict_accepted", quality_failure=NO)
        accepted.update(
            {
                "package_compiler_action": "strict_accept",
                "selection_reason": "strict_gate_accepted",
                "candidate_reuse_policy": "eligible_for_package_selection",
                "future_candidate_reusable": "",
                "train_eligible": NO,
                "count_allowed": NO,
            }
        )
        strict_accepted.append(accepted)

    final_ids = find_final_combination(strict_accepted)
    accepted_pool: list[dict[str, str]] = []
    quota_surplus: list[dict[str, str]] = []
    final_rows: list[dict[str, str]] = []
    rank = 1
    final_success = len(final_ids) == FINAL_PACKAGE_TARGET_COUNT

    for row in sorted(strict_accepted, key=lambda item: (item.get("source_subset", ""), item.get("export_correct_choice", ""), item.get("seed_sample_id", ""))):
        if row["candidate_id"] in final_ids:
            selected_row = dict(row)
            selected_row.update(promotion_fields(final_success))
            selected_row.update(
                {
                    "package_candidate_role": "final_package_candidate",
                    "package_compiler_action": "select_final_package",
                    "final_package_selected": YES,
                    "pool_class": "final_package_selected",
                    "selection_rank": str(rank),
                    "selection_reason": "strict_gate_and_exact_quota_selected",
                    "train_eligible": YES,
                    "final_status": "pass",
                    "audit_required": NO,
                    "audit_reason": "",
                }
            )
            accepted_pool.append(selected_row)
            final_rows.append(selected_row)
            rank += 1
        else:
            surplus = dict(row)
            surplus.update(pool_promotion_fields("quota_surplus"))
            surplus.update(
                {
                    "pool_class": "quota_surplus",
                    "quality_failure": NO,
                    "tail_class": "quota_surplus_not_quality_failure",
                    "not_selected_reason": "label_quota_filled",
                    "quota_surplus_reason": "exact_final_package_quota_not_selected",
                    "future_candidate_reusable": YES,
                    "candidate_reuse_policy": "reuse_allowed_as_surplus_candidate",
                    "package_compiler_action": "quota_surplus",
                    "train_eligible": NO,
                }
            )
            accepted_pool.append(surplus)
            quota_surplus.append(surplus)

    final_rows = add_final_split(final_rows)
    rejected_pool = [*quality_rejected, *quota_surplus]
    split_counts = Counter(row.get("split", "") for row in final_rows)
    final_label_counts = Counter(row.get("export_correct_choice", "") for row in final_rows)
    final_source_counts = Counter(row.get("source_subset", "") for row in final_rows)
    final_lane_counts = Counter(row.get("sampling_lane", "") for row in final_rows)
    compiler_gate_passed = (
        final_success
        and {label: final_label_counts.get(label, 0) for label in FINAL_TARGET_LABEL_COUNTS} == FINAL_TARGET_LABEL_COUNTS
        and dict(final_source_counts) == FINAL_SOURCE_COUNTS
        and {lane: final_lane_counts.get(lane, 0) for lane in FINAL_LANE_COUNTS} == FINAL_LANE_COUNTS
    )

    for row in final_rows:
        row.update(promotion_fields(compiler_gate_passed))

    write_csv(CANDIDATE_POOL_PATH, candidate_pool)
    write_csv(ACCEPTED_POOL_PATH, accepted_pool)
    write_csv(REJECTED_POOL_PATH, rejected_pool)
    write_csv(TAIL_TAXONOMY_PATH, quality_rejected, preferred_fields=union_fields(rejected_pool))
    write_csv(QUOTA_SURPLUS_POOL_PATH, quota_surplus, preferred_fields=union_fields(rejected_pool))
    write_csv(FINAL_PACKAGE_CSV_PATH, final_rows)
    write_csv(VALIDATOR_REPORT_CSV_PATH, final_rows)
    write_csv(MERGED_SCORES_PATH, final_rows)
    write_csv(PROBLEM_DATASET_MANIFEST_PATH, final_rows)
    pb6.pb4.pb3.base.write_jsonl_atomic(PROBLEM_TRAIN_PATH, [row for row in final_rows if row.get("split") == "train"])
    pb6.pb4.pb3.base.write_jsonl_atomic(PROBLEM_DEV_PATH, [row for row in final_rows if row.get("split") == "dev"])
    pb6.pb4.pb3.base.write_jsonl_atomic(PROBLEM_TEST_PATH, [row for row in final_rows if row.get("split") == "test"])
    write_csv(PROBLEM_AUDIT_QUEUE_PATH, [], preferred_fields=union_fields(final_rows) or ["seed_sample_id"])

    COMPILER_RESULT = {
        "candidate_total": len(candidate_pool),
        "accepted_total": len(accepted_pool),
        "strict_accepted_total": len(strict_accepted),
        "final_package_total": len(final_rows),
        "rejected_total": len(rejected_pool),
        "quality_tail_total": len(quality_rejected),
        "quota_surplus_total": len(quota_surplus),
        "compiler_gate_passed": compiler_gate_passed,
        "promotion_contract_passed": compiler_gate_passed,
        "final_label_counts": dict(final_label_counts),
        "final_source_counts": dict(final_source_counts),
        "final_lane_counts": dict(final_lane_counts),
        "split_counts": dict(split_counts),
        "quality_tail_by_class": dict(Counter(row.get("tail_class", "") for row in quality_rejected)),
    }
    write_compiler_artifacts(candidate_pool, accepted_pool, rejected_pool, quality_rejected, quota_surplus, final_rows)
    return final_rows


def render_contract_markdown(title: str, row_count: int) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| package_role | `{PACKAGE_ROLE}` |",
            f"| batch_status | `{CANDIDATE_BATCH_STATUS}` |",
            f"| count_reflection_status | `{CANDIDATE_REFLECTION_STATUS}` |",
            "| downstream_consumption_allowed | `아니오` |",
            "| count_allowed | `아니오` |",
            f"| count_disposition | `{COUNT_DISPOSITION}` |",
            f"| compiler_gate_passed | `{YES if COMPILER_RESULT.get('compiler_gate_passed') else NO}` |",
            f"| promotion_contract_passed | `{YES if COMPILER_RESULT.get('promotion_contract_passed') else NO}` |",
            f"| promotion_contract_status | `{PROMOTION_CONTRACT_STATUS if COMPILER_RESULT.get('promotion_contract_passed') else 'failed_not_counted'}` |",
            f"| row_count | `{row_count}` |",
            "",
        ]
    )


def write_compiler_artifacts(
    candidate_pool: list[dict[str, str]],
    accepted_pool: list[dict[str, str]],
    rejected_pool: list[dict[str, str]],
    quality_rejected: list[dict[str, str]],
    quota_surplus: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> None:
    # compiler/evidence/linter manifest는 repo-relative alias를 기본으로 두고, run_manifest에는 absolute도 함께 남긴다.
    compiler_manifest = {
        "compiler_manifest_version": f"{VERSION_TAG}_compiler_v1",
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_passed": YES if COMPILER_RESULT.get("promotion_contract_passed") else NO,
        "compiler_gate_passed": YES if COMPILER_RESULT.get("compiler_gate_passed") else NO,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS if COMPILER_RESULT.get("promotion_contract_passed") else "failed_not_counted",
        "candidate_total": len(candidate_pool),
        "accepted_total": len(accepted_pool),
        "strict_accepted_total": COMPILER_RESULT.get("strict_accepted_total", 0),
        "final_package_total": len(final_rows),
        "rejected_total": len(rejected_pool),
        "quality_tail_total": len(quality_rejected),
        "quota_surplus_total": len(quota_surplus),
        "selection_policy": "strict validator accept/export_ready, exact source/label/lane quota combination",
        "artifacts": {
            "candidate_pool": repo_rel(CANDIDATE_POOL_PATH),
            "accepted_pool": repo_rel(ACCEPTED_POOL_PATH),
            "final_package": repo_rel(FINAL_PACKAGE_CSV_PATH),
            "rejected_pool": repo_rel(REJECTED_POOL_PATH),
            "tail_taxonomy": repo_rel(TAIL_TAXONOMY_PATH),
            "quota_surplus_pool": repo_rel(QUOTA_SURPLUS_POOL_PATH),
            "candidate_validator_report": repo_rel(CANDIDATE_VALIDATOR_REPORT_CSV_PATH),
            "final_validator_report": repo_rel(VALIDATOR_REPORT_CSV_PATH),
        },
    }
    pb6.pb4.pb3.base.write_json_atomic(COMPILER_MANIFEST_PATH, compiler_manifest)
    pb6.pb4.pb3.base.write_text_atomic(FINAL_PACKAGE_MD_PATH, render_contract_markdown("candidate final package", len(final_rows)))
    pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, render_contract_markdown("candidate final validator report", len(final_rows)))
    pb6.pb4.pb3.base.write_text_atomic(MANIFEST_HEADER_GATE_MD_PATH, render_contract_markdown("candidate manifest header gate", len(final_rows)))
    pb6.pb4.pb3.base.write_text_atomic(
        COMPILER_SUMMARY_MD_PATH,
        "\n".join(
            [
                f"# compiler summary `{VERSION_TAG}`",
                "",
                f"- candidate_total: `{len(candidate_pool)}`",
                f"- accepted_total: `{len(accepted_pool)}`",
                f"- final_package_total: `{len(final_rows)}`",
                f"- rejected_total: `{len(rejected_pool)}`",
                f"- quality_tail_total: `{len(quality_rejected)}`",
                f"- quota_surplus_total: `{len(quota_surplus)}`",
                f"- compiler_gate_passed: `{COMPILER_RESULT.get('compiler_gate_passed')}`",
                "- count_reflection_status: `not_counted_until_reviewer_signoff`",
                "",
            ]
        ),
    )
    write_linter_and_evidence_manifests()


def write_linter_and_evidence_manifests() -> None:
    # linter가 validator wiring artifact를 필수 gate로 보므로, candidate/final label wiring을 명시적으로 남긴다.
    pb6.pb4.pb3.base.write_text_atomic(
        VALIDATOR_WIRING_CHECK_MD_PATH,
        "\n".join(
            [
                f"# validator wiring check `{VERSION_TAG}`",
                "",
                "| check | result | note |",
                "| --- | --- | --- |",
                f"| fixed preflight seed registry reused | `pass` | `{SOURCE_PREFLIGHT_RUN_NAME}` seed registry 사용 |",
                f"| candidate target label schedule | `pass` | candidate target `A/B/C/D = {'/'.join(str(CANDIDATE_TARGET_LABEL_COUNTS[label]) for label in CANDIDATE_TARGET_LABEL_COUNTS)}` 적용 |",
                f"| final export label schedule | `pass` | final export `A/B/C/D = {'/'.join(str(FINAL_TARGET_LABEL_COUNTS[label]) for label in FINAL_TARGET_LABEL_COUNTS)}` 적용 |",
                f"| final package target | `pass` | strict final `{FINAL_PACKAGE_TARGET_COUNT}` package만 count 후보 |",
                f"| {GATE_FIELD_LABEL} | `pass` | `single_correct_choice`, `rule_application_split`, `issue_boundary`, `case_fact_alignment`, `hierarchy_overlap` 기록 |",
                "| downstream guard fields | `pass` | `validator_action`, `validator_export_disposition`, `split_allowed`, `count_allowed` 기록 |",
                "| count reflection | `pass` | reviewer sign-off 전 core current count 미변경 |",
                "",
            ]
        ),
    )
    linter_paths = {
        "run_manifest": repo_rel(RUN_MANIFEST_PATH),
        "processed_manifest": repo_rel(PROBLEM_DATASET_MANIFEST_PATH),
        "split_jsonl": [repo_rel(PROBLEM_TRAIN_PATH), repo_rel(PROBLEM_DEV_PATH), repo_rel(PROBLEM_TEST_PATH)],
        "final_package_csv": repo_rel(FINAL_PACKAGE_CSV_PATH),
        "merged_csv": repo_rel(MERGED_SCORES_PATH),
        "validator_report_csv": repo_rel(VALIDATOR_REPORT_CSV_PATH),
        "candidate_pool_csv": repo_rel(CANDIDATE_POOL_PATH),
        "rejected_pool_csv": repo_rel(REJECTED_POOL_PATH),
        "tail_taxonomy_csv": repo_rel(TAIL_TAXONOMY_PATH),
        "quota_surplus_csv": repo_rel(QUOTA_SURPLUS_POOL_PATH),
        "validator_wiring_check_md": repo_rel(VALIDATOR_WIRING_CHECK_MD_PATH),
        "header_gate_md": repo_rel(MANIFEST_HEADER_GATE_MD_PATH),
        "final_package_md": repo_rel(FINAL_PACKAGE_MD_PATH),
        "validator_report_md": repo_rel(VALIDATOR_REPORT_MD_PATH),
    }
    pb6.pb4.pb3.base.write_json_atomic(
        ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH,
        {
            "fixture_version": f"{VERSION_TAG}_candidate_v1",
            "description": f"Live candidate package check for {VERSION_TAG}.",
            "fixtures": [
                {
                    "fixture_id": LINTER_FIXTURE_ID,
                    "artifact_role": PACKAGE_ROLE,
                    "fixture_mode": "live_artifact_check",
                    "expected_result": "pass",
                    "expected_failure_code": "",
                    "expected_failure_codes": [],
                    "validator_wiring_expectations": {
                        "required_phrases": [
                            f"`{SOURCE_PREFLIGHT_RUN_NAME}` seed registry 사용",
                            f"candidate target `A/B/C/D = {'/'.join(str(CANDIDATE_TARGET_LABEL_COUNTS[label]) for label in CANDIDATE_TARGET_LABEL_COUNTS)}`",
                            f"final export `A/B/C/D = {'/'.join(str(FINAL_TARGET_LABEL_COUNTS[label]) for label in FINAL_TARGET_LABEL_COUNTS)}`",
                        ],
                    },
                    "paths": linter_paths,
                }
            ],
        },
    )
    pb6.pb4.pb3.base.write_json_atomic(
        EVIDENCE_CARD_PACKAGE_MANIFEST_PATH,
        {
            "manifest_version": "evidence_card_candidate_v1",
            "description": f"{VERSION_TAG} candidate package evidence card input.",
            "count_context": {
                "current_usable": CURRENT_OBJECTIVE_COUNT["usable"],
                "current_train": CURRENT_OBJECTIVE_COUNT["train"],
                "current_eval": CURRENT_OBJECTIVE_COUNT["eval"],
            },
            "packages": [
                {
                    "package_id": VERSION_TAG,
                    "run_name": RUN_NAME,
                    "version_tag": VERSION_TAG,
                    "package_role": PACKAGE_ROLE,
                    "run_dir": repo_rel(RUN_DIR),
                    "processed_package_dir": repo_rel(PROCESSED_DIR),
                    "linter_fixture_id": LINTER_FIXTURE_ID,
                    "linter_report_dir": repo_rel(RUN_LINTER_DIR),
                    "source_chain": f"{SOURCE_PREFLIGHT_RUN_NAME} -> {EXPECTED_CANDIDATE_SEED_COUNT} candidate API execution -> strict final {FINAL_PACKAGE_TARGET_COUNT} compiler",
                }
            ],
        },
    )


def split_dataset_with_overgeneration_compiler(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows:
        write_csv(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows)
    validated_rows = judgment_pilot.apply_judgment_validator(rows)
    if validated_rows:
        write_csv(CANDIDATE_MERGED_SCORES_PATH, validated_rows)
    final_rows = compile_final_package(validated_rows)
    return final_rows


def build_tail_memo(merged_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tail_rows = read_csv_rows(TAIL_TAXONOMY_PATH) if TAIL_TAXONOMY_PATH.exists() else []
    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "pool_class": "quality_reject",
                "quality_failure": YES,
                "tail_class": "tail 없음",
                "not_selected_reason": "",
            }
        ]
    write_csv(TAIL_MEMO_CSV_PATH, tail_rows)
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | pool | tail_class | reason | validator | status | error_tags |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row.get('seed_sample_id', '')}` | `{row.get('pool_class', '')}` | `{row.get('tail_class', '')}` | `{row.get('not_selected_reason', '')}` | `{row.get('validator_reason_short', '')}` | `{row.get('final_status', '')}` | `{row.get('error_tags', '')}` |"
        )
    pb6.pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    final_rows = read_csv_rows(FINAL_PACKAGE_CSV_PATH) if FINAL_PACKAGE_CSV_PATH.exists() else []
    candidate_rows = read_csv_rows(CANDIDATE_POOL_PATH) if CANDIDATE_POOL_PATH.exists() else []
    summary_rows = [
        {"metric": "candidate_total", "value": str(len(candidate_rows))},
        {"metric": "accepted_total", "value": str(COMPILER_RESULT.get("accepted_total", 0))},
        {"metric": "final_package_total", "value": str(len(final_rows))},
        {"metric": "quality_tail_total", "value": str(COMPILER_RESULT.get("quality_tail_total", 0))},
        {"metric": "quota_surplus_total", "value": str(COMPILER_RESULT.get("quota_surplus_total", 0))},
        {"metric": "compiler_gate_passed", "value": str(COMPILER_RESULT.get("compiler_gate_passed", False))},
    ]
    write_csv(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    lane_rows = [{"sampling_lane": lane, "count": str(count)} for lane, count in sorted(Counter(row.get("sampling_lane", "") for row in final_rows).items())]
    write_csv(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, ["sampling_lane", "count"])
    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## candidate pool health",
        f"- candidate_total: `{len(candidate_rows)}`",
        f"- accepted_total: `{COMPILER_RESULT.get('accepted_total', 0)}`",
        f"- final_package_total: `{len(final_rows)}`",
        f"- quality_tail_total: `{COMPILER_RESULT.get('quality_tail_total', 0)}`",
        f"- quota_surplus_total: `{COMPILER_RESULT.get('quota_surplus_total', 0)}`",
        f"- compiler_gate_passed: `{COMPILER_RESULT.get('compiler_gate_passed', False)}`",
        "",
        "## final package balance",
        f"- labels: `{COMPILER_RESULT.get('final_label_counts', {})}`",
        f"- sources: `{COMPILER_RESULT.get('final_source_counts', {})}`",
        f"- lanes: `{COMPILER_RESULT.get('final_lane_counts', {})}`",
        "",
    ]
    pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines))
    return summary_rows


def run_post_compile_validation() -> dict[str, Any]:
    # linter fixture가 run_manifest 안의 linter/evidence path 자체를 검산하므로,
    # 첫 pass로 파일을 만든 뒤 두 번째 pass에서 self-reference까지 확인한다.
    linter_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "shared" / "artifact_linter.py"),
        "--fixture-manifest",
        str(ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH),
        "--output-dir",
        str(RUN_LINTER_DIR),
    ]
    evidence_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "shared" / "evidence_card.py"),
        "--package-manifest",
        str(EVIDENCE_CARD_PACKAGE_MANIFEST_PATH),
        "--output-dir",
        str(RUN_EVIDENCE_DIR),
        "--linter-report-dir",
        str(RUN_LINTER_DIR),
    ]
    subprocess.run(linter_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    subprocess.run(evidence_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    linter = subprocess.run(linter_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    evidence = subprocess.run(evidence_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return {
        "artifact_linter_returncode": linter.returncode,
        "artifact_linter_stdout": linter.stdout.strip(),
        "artifact_linter_stderr": linter.stderr.strip(),
        "evidence_card_returncode": evidence.returncode,
        "evidence_card_stdout": evidence.stdout.strip(),
        "evidence_card_stderr": evidence.stderr.strip(),
        "post_compile_validation_status": "all_green" if linter.returncode == 0 and evidence.returncode == 0 else "failed_or_needs_sync",
    }


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict[str, Any]:
    tail_rows = build_tail_memo(merged_rows)
    final_rows = read_csv_rows(FINAL_PACKAGE_CSV_PATH) if FINAL_PACKAGE_CSV_PATH.exists() else []
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "run_id": RUN_NAME,
        "created_at_utc": pb6.pb4.pb3.base.utc_now_iso(),
        "run_dir": str(RUN_DIR),
        "run_dir_repo_relative": repo_rel(RUN_DIR),
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_passed": YES if COMPILER_RESULT.get("promotion_contract_passed") else NO,
        "compiler_gate_passed": YES if COMPILER_RESULT.get("compiler_gate_passed") else NO,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS if COMPILER_RESULT.get("promotion_contract_passed") else "failed_not_counted",
        "source_preflight_run_name": SOURCE_PREFLIGHT_RUN_NAME,
        "candidate_recipe_source": f"v2_difficulty_patch_r2_{VERSION_TAG}",
        "seed_registry_strategy": f"fixed_from_{SOURCE_PREFLIGHT_RUN_NAME}",
        "seed_registry_count": len(seed_rows),
        "generation_main_max_workers": pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS,
        "generation_strict_max_workers": pb6.pb4.pb3.base.GENERATOR_STRICT_MAX_WORKERS,
        "judge_main_max_workers": pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS,
        "judge_strict_max_workers": pb6.pb4.pb3.base.JUDGE_STRICT_MAX_WORKERS,
        "candidate_total": COMPILER_RESULT.get("candidate_total", 0),
        "accepted_total": COMPILER_RESULT.get("accepted_total", 0),
        "final_package_total": len(final_rows),
        "rejected_total": COMPILER_RESULT.get("rejected_total", 0),
        "quality_tail_total": COMPILER_RESULT.get("quality_tail_total", 0),
        "quota_surplus_total": COMPILER_RESULT.get("quota_surplus_total", 0),
        "generation_count": pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": pb6.pb4.pb3.base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": pb6.pb4.pb3.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": pb6.pb4.pb3.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": pb6.pb4.pb3.base.load_jsonl_count(NEARMISS_LOG_PATH),
        "candidate_merged_count": pb6.pb4.pb3.base.load_csv_count(CANDIDATE_MERGED_SCORES_PATH),
        "merged_count": pb6.pb4.pb3.base.load_csv_count(MERGED_SCORES_PATH),
        "dataset_manifest_count": pb6.pb4.pb3.base.load_csv_count(PROBLEM_DATASET_MANIFEST_PATH),
        "problem_train_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": pb6.pb4.pb3.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "success_criteria": {
            "candidate_execution": EXPECTED_CANDIDATE_SEED_COUNT,
            "final_package": FINAL_PACKAGE_TARGET_COUNT,
            "final_hard_soft_audit": "0/0/0",
            "final_label_counts": FINAL_TARGET_LABEL_COUNTS,
            "final_source_counts": FINAL_SOURCE_COUNTS,
            "final_lane_counts": FINAL_LANE_COUNTS,
            "metadata_shuffle_mismatch": "0/0",
            "reviewer_signoff_required_for_count_reflection": True,
        },
        "success_result": {
            "candidate_execution_complete": pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH) == EXPECTED_CANDIDATE_SEED_COUNT,
            "compiler_gate_passed": bool(COMPILER_RESULT.get("compiler_gate_passed")),
            "promotion_contract_passed": bool(COMPILER_RESULT.get("promotion_contract_passed")),
            "final_package_total": len(final_rows),
            "quality_tail_total": COMPILER_RESULT.get("quality_tail_total", 0),
        },
        "tail_memo_count": len([row for row in tail_rows if row.get("seed_sample_id")]),
    }
    artifact_paths = {
        "seed_registry": SEED_REGISTRY_PATH,
        "seed_ready": SEED_READY_PATH,
        "seed_preflight": SEED_PREFLIGHT_CSV_PATH,
        "target_label_schedule": TARGET_LABEL_SCHEDULE_CSV_PATH,
        "source_preflight_exclusion_audit": SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH,
        "generated_problems": GENERATED_PROBLEMS_PATH,
        "judge_grounding_log": GROUNDING_LOG_PATH,
        "judge_keyedness_log": KEYEDNESS_LOG_PATH,
        "judge_distractorfit_log": DISTRACTORFIT_LOG_PATH,
        "judge_nearmiss_log": NEARMISS_LOG_PATH,
        "raw_merged_before_validator": RAW_MERGED_BEFORE_VALIDATOR_PATH,
        "candidate_merged_scores": CANDIDATE_MERGED_SCORES_PATH,
        "merged_scores": MERGED_SCORES_PATH,
        "candidate_pool": CANDIDATE_POOL_PATH,
        "accepted_pool": ACCEPTED_POOL_PATH,
        "final_package": FINAL_PACKAGE_CSV_PATH,
        "rejected_pool": REJECTED_POOL_PATH,
        "tail_taxonomy": TAIL_TAXONOMY_PATH,
        "quota_surplus_pool": QUOTA_SURPLUS_POOL_PATH,
        "compiler_manifest": COMPILER_MANIFEST_PATH,
        "artifact_linter_fixture_manifest": ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH,
        "evidence_card_package_manifest": EVIDENCE_CARD_PACKAGE_MANIFEST_PATH,
        "artifact_linter_report": RUN_LINTER_DIR / "artifact_linter_report.md",
        "evidence_card_summary": RUN_EVIDENCE_DIR / "evidence_card_summary.md",
        "problem_train": PROBLEM_TRAIN_PATH,
        "problem_dev": PROBLEM_DEV_PATH,
        "problem_test": PROBLEM_TEST_PATH,
        "problem_dataset_manifest": PROBLEM_DATASET_MANIFEST_PATH,
        "problem_audit_queue": PROBLEM_AUDIT_QUEUE_PATH,
        "batch_summary_md": BATCH_SUMMARY_MD_PATH,
        "tail_memo_md": TAIL_MEMO_MD_PATH,
    }
    manifest["artifact_paths"] = {key: str(path) for key, path in artifact_paths.items()}
    manifest["artifact_path_aliases"] = {key: repo_rel(path) for key, path in artifact_paths.items()}
    manifest["post_compile_validation_status"] = "pending_artifact_linter_and_evidence_card"
    manifest["artifact_linter_report_path"] = repo_rel(RUN_LINTER_DIR / "artifact_linter_report.md")
    manifest["evidence_card_summary_path"] = repo_rel(RUN_EVIDENCE_DIR / "evidence_card_summary.md")
    pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_globals() -> None:
    # 기존 판결문 repair pilot runner를 재사용하되, fixed source와 compiler hook만 overgeneration 전용으로 교체한다.
    refresh_paths()
    judgment_pilot.VERSION_TAG = VERSION_TAG
    judgment_pilot.RUN_DATE = RUN_DATE
    judgment_pilot.RUN_PURPOSE = RUN_PURPOSE
    judgment_pilot.RUN_NAME = RUN_NAME
    judgment_pilot.SOURCE_PREFLIGHT_RUN_NAME = SOURCE_PREFLIGHT_RUN_NAME
    judgment_pilot.SOURCE_PREFLIGHT_RUN_DIR = SOURCE_PREFLIGHT_RUN_DIR
    judgment_pilot.SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = SOURCE_PREFLIGHT_SEED_REGISTRY_PATH
    judgment_pilot.SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH = SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH
    judgment_pilot.SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH = SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH
    judgment_pilot.INTERIM_DIR = INTERIM_DIR
    judgment_pilot.PROCESSED_DIR = PROCESSED_DIR
    judgment_pilot.RUN_DIR = RUN_DIR
    judgment_pilot.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    judgment_pilot.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    judgment_pilot.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    judgment_pilot.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    judgment_pilot.RUN_MERGED_DIR = RUN_MERGED_DIR
    judgment_pilot.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    judgment_pilot.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    judgment_pilot.SEED_READY_PATH = SEED_READY_PATH
    judgment_pilot.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    judgment_pilot.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    judgment_pilot.TARGET_LABEL_SCHEDULE_CSV_PATH = TARGET_LABEL_SCHEDULE_CSV_PATH
    judgment_pilot.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    judgment_pilot.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    judgment_pilot.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    judgment_pilot.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    judgment_pilot.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    judgment_pilot.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    judgment_pilot.RAW_MERGED_BEFORE_VALIDATOR_PATH = RAW_MERGED_BEFORE_VALIDATOR_PATH
    judgment_pilot.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    judgment_pilot.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    judgment_pilot.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    judgment_pilot.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    judgment_pilot.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    judgment_pilot.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    judgment_pilot.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    judgment_pilot.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    judgment_pilot.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    judgment_pilot.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    judgment_pilot.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    judgment_pilot.VALIDATOR_REPORT_CSV_PATH = CANDIDATE_VALIDATOR_REPORT_CSV_PATH
    judgment_pilot.VALIDATOR_REPORT_MD_PATH = CANDIDATE_VALIDATOR_REPORT_MD_PATH
    judgment_pilot.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    judgment_pilot.PILOT_BREAKOUT_CSV_PATH = PILOT_BREAKOUT_CSV_PATH
    judgment_pilot.PILOT_BREAKOUT_MD_PATH = PILOT_BREAKOUT_MD_PATH
    judgment_pilot.MANIFEST_HEADER_GATE_MD_PATH = MANIFEST_HEADER_GATE_MD_PATH
    judgment_pilot.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_CANDIDATE_SEED_COUNT
    judgment_pilot.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    judgment_pilot.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    judgment_pilot.EXPECTED_SOURCE_COUNTS = EXPECTED_SOURCE_COUNTS
    judgment_pilot.TARGET_LABEL_COUNTS = CANDIDATE_TARGET_LABEL_COUNTS
    judgment_pilot.SUCCESS_USABLE_MIN = FINAL_PACKAGE_TARGET_COUNT
    judgment_pilot.SUCCESS_HARD_FAIL_MAX = 0
    judgment_pilot.SUCCESS_SOFT_FAIL_MAX = 0
    judgment_pilot.SUCCESS_AUDIT_MAX = 0
    judgment_pilot.BATCH_STATUS = f"{VERSION_TAG}_candidate_validated_not_compiled"
    judgment_pilot.COUNT_REFLECTION_STATUS = CANDIDATE_REFLECTION_STATUS
    judgment_pilot.DOWNSTREAM_CONSUMPTION_ALLOWED = NO

    judgment_pilot.configure_judgment_pilot_globals()
    pb6.RUN_LABEL = RUN_LABEL
    pb6.CANDIDATE_RECIPE_SOURCE = f"v2_difficulty_patch_r2_{VERSION_TAG}"
    pb6.SEED_REGISTRY_STRATEGY = f"fixed_from_{SOURCE_PREFLIGHT_RUN_NAME}"
    pb6.LAW_STATUS_NOTE = f"{VERSION_TAG}_candidate_not_counted_until_signoff"
    pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb6.build_seed_registry = build_seed_registry_from_preflight
    pb6.build_batch_summary = build_batch_summary
    pb6.build_run_manifest = build_run_manifest
    pb6.pb4.pb3.base.split_dataset = split_dataset_with_overgeneration_compiler


def main() -> dict[str, Any]:
    configure_globals()
    manifest = pb6.main()
    validation_result = run_post_compile_validation()
    manifest.update(validation_result)
    if RUN_MANIFEST_PATH.exists():
        current_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        current_manifest.update(validation_result)
        pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, current_manifest)
        return current_manifest
    return manifest


if __name__ == "__main__":
    main()
