from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 package factory 전환의 첫 실제 seed/spec/wiring stop line이다.
# API를 호출하지 않고 판결문_QA candidate 24개와 final 16개 quota 계약만 검산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_repair_pilot_seed_preflight as judgment_base,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402


VERSION_TAG = "objective_judgment_small_overgeneration_pilot_preflight"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_judgment_small_overgeneration_seed_spec_wiring_check"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "judgment small overgeneration preflight"

PROJECT_ROOT = judgment_base.PROJECT_ROOT
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
EXCLUSION_AUDIT_CSV_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.csv"
EXCLUSION_AUDIT_MD_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
FINAL_PACKAGE_SPEC_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_spec_{VERSION_TAG}.csv"
FINAL_PACKAGE_SPEC_MD_PATH = RUN_EXPORTS_DIR / f"final_package_spec_{VERSION_TAG}.md"
VALIDATOR_SCHEMA_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.csv"
VALIDATOR_SCHEMA_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.md"
PACKAGE_COMPILER_CONTRACT_JSON_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.json"
PACKAGE_COMPILER_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"

EXPECTED_CANDIDATE_SEED_COUNT = 24
FINAL_PACKAGE_TARGET_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 24}
EXPECTED_LANE_BY_DOC = {
    ("판결문_QA", "generalization_03_04"): 12,
    ("판결문_QA", "expansion_01_02"): 12,
}
JUDGMENT_SOURCE_COUNTS = {
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

COUNTED_EXCLUSION_COMPONENTS = {
    "r2": 16,
    "pb2": 13,
    "pb3": 40,
    "pb4": 40,
    "pb9_final_package": 40,
    "judgment_final_package": 16,
    "interpretation_final_package": 16,
}

# package factory pilot은 이미 counted된 package뿐 아니라 failed/candidate run의 seen seed도 제외한다.
# 기존 판결문 preflight base 목록에 judgment/interpretation counted package까지 명시적으로 더한다.
REFERENCE_SEED_REGISTRY_PATHS = [
    *judgment_base.REFERENCE_SEED_REGISTRY_PATHS,
    PROJECT_ROOT
    / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_pilot_seed_preflight/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_pilot/seed_registry.csv",
    PROJECT_ROOT
    / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_a_slot_replacement_package/seed_registry.csv",
    PROJECT_ROOT
    / "data/interim/aihub/problem_generation/production_batches/objective_interpretation_repair_pilot_seed_preflight/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_interpretation_repair_pilot/seed_registry.csv",
    PROJECT_ROOT
    / "data/interim/aihub/problem_generation/production_batches/objective_interpretation_repair_dslot_replacement_package/seed_registry.csv",
    PROJECT_ROOT
    / "data/interim/aihub/problem_generation/production_batches/objective_interpretation_repair_dslot_final_replacement_package/seed_registry.csv",
]

VALIDATOR_REQUIRED_FIELDS = [
    *judgment_base.VALIDATOR_REQUIRED_FIELDS,
    "package_candidate_role",
    "package_compiler_action",
    "final_package_selected",
    "quota_surplus_reason",
]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collect_excluded_rows() -> list[dict[str, str]]:
    # base preflight의 exclusion registry를 확장해 current counted 181 seed와 failed/candidate seen seed를 함께 막는다.
    rows: list[dict[str, str]] = []
    for path in REFERENCE_SEED_REGISTRY_PATHS:
        rows.extend(judgment_base.load_csv_rows_if_exists(path))
    # 일부 과거 split artifact는 seed registry보다 더 넓은 provenance를 담고 있으므로,
    # seed 선택 단계에서도 processed/audit/tail identifier를 flatten해 한 번 더 제외한다.
    for paths in judgment_base.audit_source_paths().values():
        for path in paths:
            for row in judgment_base.load_audit_source_rows(path):
                flattened = {
                    field: judgment_base.normalized_value(row, field)
                    for field in ["seed_sample_id", "reference_sample_id", "family_id", "label_path", "raw_path"]
                }
                if any(flattened.values()):
                    rows.append(flattened)
    return rows


def build_seed_row(record: dict) -> dict[str, str]:
    row = judgment_base.ORIGINAL_PB6_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_judgment_small_overgeneration_candidate_seed"
    row["selection_note"] = "판결문_QA package factory no-API candidate seed; target 16 final package를 위해 candidate 24개를 먼저 고정"
    row["pb6_seed_filter_note"] = "judgment_only_small_overgeneration_seed_filter"
    row["non_law_scope_note"] = "judgment_small_overgeneration_preflight_no_api_candidate_not_counted"
    row["package_candidate_role"] = "candidate_pool"
    row["count_reflection_status"] = "candidate_not_counted"
    return row


def configure_judgment_seed_selector() -> None:
    # pb6 selector를 재사용해 raw/label pairing과 family_id 생성 규칙을 그대로 유지한다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = RUN_LABEL
    pb6.SEED_ID_PREFIX = "judgment_overgen_preflight"
    pb6.SEED_SELECTION_ROLE = "objective_judgment_small_overgeneration_candidate_seed"
    pb6.SEED_SELECTION_NOTE = "판결문_QA package factory no-API candidate seed"
    pb6.SEED_FILTER_NOTE = "judgment_only_small_overgeneration_seed_filter"
    pb6.SCOPE_NOTE = "판결문_QA only; API 전 candidate 24 / final 16 seed-spec-wiring check"
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_CANDIDATE_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.PB6_SOURCE_COUNTS = JUDGMENT_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.OVERLAP_CHECK_LABEL = "no current/candidate/failed/held-out/audit overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        "`181`은 current counted objective seed package pool이다.",
        "failed/candidate seen seed까지 추가 제외해 package factory pilot의 fresh seed 성격을 유지한다.",
    ]
    pb6.INTERIM_DIR = INTERIM_DIR
    pb6.PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
    pb6.RUN_DIR = RUN_DIR
    pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb6.SEED_READY_PATH = SEED_READY_PATH
    pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb6.collect_excluded_rows = collect_excluded_rows
    pb6.passes_pb6_seed_filter = judgment_base.passes_judgment_seed_filter
    pb6.build_seed_row = build_seed_row


def build_preflight_rows(seed_rows: list[dict[str, str]], exclusion_sets: dict) -> list[dict[str, str]]:
    return judgment_base.build_preflight_rows(seed_rows, exclusion_sets)


def assert_preflight(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    # base assertion은 configured globals를 보므로, 이 run의 24/12/6/6 기준으로 동작하도록 맞춘다.
    judgment_base.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_CANDIDATE_SEED_COUNT
    judgment_base.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    judgment_base.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    judgment_base.JUDGMENT_SOURCE_COUNTS = JUDGMENT_SOURCE_COUNTS
    judgment_base.TARGET_LABEL_COUNTS = CANDIDATE_TARGET_LABEL_COUNTS
    judgment_base.assert_judgment_preflight(seed_rows, preflight_rows)


def write_preflight_report(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["judgment_seed_action"] for row in seed_rows)
    tail_counts = Counter(row["tail_proximity_class"] for row in seed_rows)
    write_csv(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- candidate_seed_count: `{len(seed_rows)}`",
        f"- target_final_package_count: `{FINAL_PACKAGE_TARGET_COUNT}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- source_subset_counts: `{dict(source_counts)}`",
        f"- target_label_counts: `{dict(target_counts)}`",
        f"- judgment_seed_action_counts: `{dict(action_counts)}`",
        f"- tail_proximity_counts: `{dict(tail_counts)}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        "| candidate seed count is 24 | `pass` |",
        "| final package target is 16 | `pass` |",
        "| doc type is 판결문_QA only | `pass` |",
        "| candidate source split is 01/02/03/04 = 6/6/6/6 | `pass` |",
        "| candidate lane split is 12/12 | `pass` |",
        "| candidate target label schedule is A/B/C/D = 6/6/6/6 | `pass` |",
        "| no batch duplicate | `pass` |",
        "| no prior/candidate/failed/held-out/audit overlap | `pass` |",
        "| no excluded seed leaked into registry | `pass` |",
    ]
    write_text(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_target_label_schedule(seed_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "candidate_slot": str(index + 1),
            "target_correct_choice": row["target_correct_choice"],
            "final_package_candidate_state": "candidate_pool_not_counted",
        }
        for index, row in enumerate(seed_rows)
    ]
    write_csv(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)


def write_final_package_spec() -> None:
    rows: list[dict[str, str]] = []
    for source_subset, count in FINAL_SOURCE_COUNTS.items():
        rows.append({"quota_type": "source_subset", "quota_key": source_subset, "target_count": str(count)})
    for lane, count in FINAL_LANE_COUNTS.items():
        rows.append({"quota_type": "lane", "quota_key": lane, "target_count": str(count)})
    for label, count in FINAL_TARGET_LABEL_COUNTS.items():
        rows.append({"quota_type": "export_correct_choice", "quota_key": label, "target_count": str(count)})
    rows.extend(
        [
            {"quota_type": "final_gate", "quota_key": "hard_fail", "target_count": "0"},
            {"quota_type": "final_gate", "quota_key": "soft_fail", "target_count": "0"},
            {"quota_type": "final_gate", "quota_key": "audit", "target_count": "0"},
            {"quota_type": "final_gate", "quota_key": "metadata_mismatch", "target_count": "0"},
            {"quota_type": "final_gate", "quota_key": "shuffle_mismatch", "target_count": "0"},
            {"quota_type": "final_gate", "quota_key": "validator_accept_export_ready", "target_count": "16"},
        ]
    )
    write_csv(FINAL_PACKAGE_SPEC_CSV_PATH, rows, ["quota_type", "quota_key", "target_count"])
    lines = [
        f"# final package spec `{VERSION_TAG}`",
        "",
        "| quota_type | quota_key | target_count |",
        "| --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| `{row['quota_type']}` | `{row['quota_key']}` | `{row['target_count']}` |")
    write_text(FINAL_PACKAGE_SPEC_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(FINAL_PACKAGE_SPEC_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(FINAL_PACKAGE_SPEC_MD_PATH, RUN_INPUTS_DIR)


def write_validator_schema() -> None:
    rows = [{"field": field, "required": "예"} for field in VALIDATOR_REQUIRED_FIELDS]
    write_csv(VALIDATOR_SCHEMA_CSV_PATH, rows, ["field", "required"])
    lines = [
        f"# validator report schema `{VERSION_TAG}`",
        "",
        "| field | required |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| `{row['field']}` | `{row['required']}` |")
    write_text(VALIDATOR_SCHEMA_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(VALIDATOR_SCHEMA_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(VALIDATOR_SCHEMA_MD_PATH, RUN_INPUTS_DIR)


def write_package_compiler_contract() -> None:
    contract = {
        "contract_name": VERSION_TAG,
        "api_calls": 0,
        "candidate_generation_target": EXPECTED_CANDIDATE_SEED_COUNT,
        "final_package_target": FINAL_PACKAGE_TARGET_COUNT,
        "count_state_before_reviewer_signoff": "candidate_not_counted",
        "candidate_quota": {
            "doc_type_counts": EXPECTED_DOC_TYPE_COUNTS,
            "source_subset_counts": JUDGMENT_SOURCE_COUNTS,
            "lane_counts": {"generalization_03_04": 12, "expansion_01_02": 12},
            "target_label_counts": CANDIDATE_TARGET_LABEL_COUNTS,
        },
        "final_package_quota": {
            "source_subset_counts": FINAL_SOURCE_COUNTS,
            "lane_counts": FINAL_LANE_COUNTS,
            "export_correct_choice_counts": FINAL_TARGET_LABEL_COUNTS,
        },
        "strict_final_gate": {
            "hard_fail": 0,
            "soft_fail": 0,
            "audit": 0,
            "validator_action": "accept_only",
            "validator_export_disposition": "export_ready_only",
            "metadata_mismatch": 0,
            "shuffle_mismatch": 0,
        },
        "post_compile_validation_aliases": {
            "post_compile_validation_status": "required_after_api_execution",
            "artifact_linter_report_path": "required_after_package_compiler",
            "evidence_card_summary_path": "required_after_package_compiler",
        },
    }
    write_json(PACKAGE_COMPILER_CONTRACT_JSON_PATH, contract)
    lines = [
        f"# package compiler contract `{VERSION_TAG}`",
        "",
        f"- candidate_generation_target: `{EXPECTED_CANDIDATE_SEED_COUNT}`",
        f"- final_package_target: `{FINAL_PACKAGE_TARGET_COUNT}`",
        "- count_state_before_reviewer_signoff: `candidate_not_counted`",
        "- strict_final_gate: `hard/soft/audit 0`, `validator accept/export-ready only`, `metadata/shuffle mismatch 0`",
        "- post_compile_validation_aliases: `post_compile_validation_status`, `artifact_linter_report_path`, `evidence_card_summary_path`",
        "",
    ]
    write_text(PACKAGE_COMPILER_CONTRACT_MD_PATH, "\n".join(lines))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(PACKAGE_COMPILER_CONTRACT_JSON_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(PACKAGE_COMPILER_CONTRACT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry() -> list[dict[str, str]]:
    configure_judgment_seed_selector()
    pb6.pb4.pb3.base.ensure_dirs(INTERIM_DIR, RUN_DIR, RUN_INPUTS_DIR, RUN_EXPORTS_DIR)
    records, exclusion_sets = pb6.select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["source_subset"], row["seed_sample_id"]))
    for index, row in enumerate(seed_rows):
        judgment_base.augment_seed_row(row, index)
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_preflight(seed_rows, preflight_rows)
    write_csv(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    write_target_label_schedule(seed_rows)
    write_final_package_spec()
    write_validator_schema()
    write_package_compiler_contract()
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def write_exclusion_audit(seed_rows: list[dict[str, str]]) -> dict[str, int]:
    # base exclusion audit 구현을 쓰되, 이 runner의 path/global 기준으로 바꾼 뒤 호출한다.
    judgment_base.PROJECT_ROOT = PROJECT_ROOT
    judgment_base.VERSION_TAG = VERSION_TAG
    judgment_base.REFERENCE_SEED_REGISTRY_PATHS = REFERENCE_SEED_REGISTRY_PATHS
    judgment_base.EXCLUSION_AUDIT_CSV_PATH = EXCLUSION_AUDIT_CSV_PATH
    judgment_base.EXCLUSION_AUDIT_MD_PATH = EXCLUSION_AUDIT_MD_PATH
    judgment_base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    judgment_base.COUNTED_EXCLUSION_COMPONENTS = COUNTED_EXCLUSION_COMPONENTS
    summary = judgment_base.write_exclusion_audit(seed_rows)
    # base 문구는 원래 16-slot repair preflight용이라, overgeneration 24-slot 의미로 동기화한다.
    md = EXCLUSION_AUDIT_MD_PATH.read_text(encoding="utf-8")
    md = md.replace(
        "current 16 judgment preflight seeds",
        "current 24 judgment small overgeneration preflight seeds",
    )
    write_text(EXCLUSION_AUDIT_MD_PATH, md)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_MD_PATH, RUN_INPUTS_DIR)
    return summary


def write_run_manifest(seed_rows: list[dict[str, str]], exclusion_audit_summary: dict[str, int]) -> None:
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["judgment_seed_action"] for row in seed_rows)
    manifest = {
        "run_name": RUN_NAME,
        "version_tag": VERSION_TAG,
        "run_type": "no_api_seed_spec_wiring_check",
        "api_calls": 0,
        "objective_current_count_reference": {
            "usable": 167,
            "train": 130,
            "eval": 37,
            "audit": 6,
            "hard_fail": 5,
            "soft_fail": 3,
        },
        "counted_exclusion_components": COUNTED_EXCLUSION_COMPONENTS,
        "next_seed_planning_exclusion_pool": sum(COUNTED_EXCLUSION_COMPONENTS.values()),
        "candidate_seed_count": len(seed_rows),
        "final_package_target_count": FINAL_PACKAGE_TARGET_COUNT,
        "doc_type_counts": {"판결문_QA": len(seed_rows)},
        "source_subset_counts": dict(source_counts),
        "lane_counts": dict(lane_counts),
        "candidate_target_label_counts": dict(target_counts),
        "final_target_label_counts": FINAL_TARGET_LABEL_COUNTS,
        "judgment_seed_action_counts": dict(action_counts),
        "count_state": "candidate_not_counted",
        "preflight_result": {
            "passed": True,
            "api_execution_allowed_by_this_run": False,
            "next_stop_line": "reviewer_signoff_then_candidate_24_api_execution",
            "interim_seed_registry_overlap_count": exclusion_audit_summary["interim_seed_registry_overlap_count"],
            "processed_split_overlap_count": exclusion_audit_summary["processed_train_dev_test_overlap_count"],
            "audit_queue_overlap_count": exclusion_audit_summary["processed_audit_queue_overlap_count"],
            "tail_memo_overlap_count": exclusion_audit_summary["analysis_tail_memo_overlap_count"],
        },
        "future_api_pilot_contract": {
            "candidate_generation": 24,
            "final_package": 16,
            "accepted_final_gate": "hard_soft_audit_0_validator_accept_export_ready_metadata_shuffle_0",
            "count_reflection": "not_counted_until_reviewer_signoff",
        },
        "artifacts": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight_csv": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
            "exclusion_audit_csv": str(EXCLUSION_AUDIT_CSV_PATH),
            "exclusion_audit_md": str(EXCLUSION_AUDIT_MD_PATH),
            "target_label_schedule_csv": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "final_package_spec_csv": str(FINAL_PACKAGE_SPEC_CSV_PATH),
            "final_package_spec_md": str(FINAL_PACKAGE_SPEC_MD_PATH),
            "validator_schema_csv": str(VALIDATOR_SCHEMA_CSV_PATH),
            "validator_schema_md": str(VALIDATOR_SCHEMA_MD_PATH),
            "package_compiler_contract_json": str(PACKAGE_COMPILER_CONTRACT_JSON_PATH),
            "package_compiler_contract_md": str(PACKAGE_COMPILER_CONTRACT_MD_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def main() -> None:
    seed_rows = build_seed_registry()
    exclusion_audit_summary = write_exclusion_audit(seed_rows)
    write_run_manifest(seed_rows, exclusion_audit_summary)
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "candidate_seed_count": len(seed_rows),
                "final_package_target_count": FINAL_PACKAGE_TARGET_COUNT,
                "source_subset_counts": dict(Counter(row["source_subset"] for row in seed_rows)),
                "target_label_counts": dict(Counter(row["target_correct_choice"] for row in seed_rows)),
                "api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
