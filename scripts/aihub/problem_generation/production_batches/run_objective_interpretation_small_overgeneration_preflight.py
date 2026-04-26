from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 해석례_QA에 package factory를 적용하기 전 no-API stop line이다.
# API 비용을 쓰기 전에 candidate/final quota, exclusion, validator/source contract를 먼저 잠근다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_repair_pilot_seed_preflight as interpretation_base,
)


VERSION_TAG = "objective_interpretation_small_overgeneration_pilot_preflight"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_interpretation_small_overgeneration_seed_spec_wiring_check"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "interpretation small overgeneration preflight"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
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
SOURCE_FIELD_CONTRACT_CSV_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.csv"
SOURCE_FIELD_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.md"
PACKAGE_COMPILER_CONTRACT_JSON_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.json"
PACKAGE_COMPILER_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"

EXPECTED_CANDIDATE_SEED_COUNT = 28
FINAL_PACKAGE_TARGET_COUNT = 16
SEED_ID_PREFIX = "interpretation_overgen_preflight"
SEED_SELECTION_ROLE = "objective_interpretation_small_overgeneration_candidate_seed"
SEED_SELECTION_NOTE = "해석례_QA package factory no-API candidate seed"
SEED_FILTER_NOTE = "interpretation_only_small_overgeneration_seed_filter"
NON_LAW_SCOPE_NOTE = "interpretation_small_overgeneration_preflight_no_api_candidate_not_counted"
EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 28}
EXPECTED_LANE_BY_DOC = {
    ("해석례_QA", "generalization_03_04"): 14,
    ("해석례_QA", "expansion_01_02"): 14,
}
INTERPRETATION_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 7,
    "02_TL_유권해석_QA": 7,
    "03_TL_해석례_QA": 7,
    "04_TL_해석례_QA": 7,
}
CANDIDATE_TARGET_LABEL_COUNTS = {"A": 7, "B": 7, "C": 7, "D": 7}
FINAL_TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
FINAL_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 4,
    "02_TL_유권해석_QA": 4,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 4,
}
FINAL_LANE_COUNTS = {"generalization_03_04": 8, "expansion_01_02": 8}
CURRENT_OBJECTIVE_COUNT = {
    "usable": 183,
    "train": 144,
    "eval": 39,
    "audit": 6,
    "hard_fail": 5,
    "soft_fail": 3,
}
COUNTED_EXCLUSION_COMPONENTS = {
    "r2": 16,
    "pb2": 13,
    "pb3": 40,
    "pb4": 40,
    "pb9_final_package": 40,
    "judgment_a_slot_final_package": 16,
    "interpretation_dslot_final_package": 16,
    "judgment_small_overgeneration_final_package": 16,
}

ORIGINAL_PB6_BUILD_SEED_ROW = pb6.build_seed_row

VALIDATOR_REQUIRED_FIELDS = [
    *interpretation_base.VALIDATOR_REQUIRED_FIELDS,
    "package_candidate_role",
    "package_compiler_action",
    "final_package_selected",
    "quota_surplus_reason",
]


def expected_lane_counts() -> dict[str, int]:
    # wrapper가 candidate 규모를 바꿔도 reviewer-facing lane 총량을 같은 기준으로 렌더링한다.
    counts: Counter[str] = Counter()
    for (_doc_type, lane), count in EXPECTED_LANE_BY_DOC.items():
        counts[lane] += count
    return dict(counts)


def refresh_paths() -> None:
    # medium wrapper가 VERSION_TAG/RUN_NAME을 바꿀 때 small runner의 artifact path가 stale해지지 않도록 재계산한다.
    global INTERIM_DIR, RUN_DIR, RUN_INPUTS_DIR, RUN_EXPORTS_DIR
    global SEED_REGISTRY_PATH, SEED_READY_PATH, SEED_PREFLIGHT_CSV_PATH, SEED_PREFLIGHT_MD_PATH
    global EXCLUSION_AUDIT_CSV_PATH, EXCLUSION_AUDIT_MD_PATH, TARGET_LABEL_SCHEDULE_CSV_PATH
    global FINAL_PACKAGE_SPEC_CSV_PATH, FINAL_PACKAGE_SPEC_MD_PATH, VALIDATOR_SCHEMA_CSV_PATH
    global VALIDATOR_SCHEMA_MD_PATH, SOURCE_FIELD_CONTRACT_CSV_PATH, SOURCE_FIELD_CONTRACT_MD_PATH
    global PACKAGE_COMPILER_CONTRACT_JSON_PATH, PACKAGE_COMPILER_CONTRACT_MD_PATH, RUN_MANIFEST_PATH
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
    SOURCE_FIELD_CONTRACT_CSV_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.csv"
    SOURCE_FIELD_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.md"
    PACKAGE_COMPILER_CONTRACT_JSON_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.json"
    PACKAGE_COMPILER_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"package_compiler_contract_{VERSION_TAG}.md"
    RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def normalized_value(row: dict, field: str) -> str:
    # split JSONL은 식별자가 metadata에 들어갈 수 있어서 top-level과 metadata를 같이 본다.
    for key, value in row.items():
        if str(key).lstrip("\ufeff") == field and value not in (None, ""):
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if str(key).lstrip("\ufeff") == field and value not in (None, ""):
                return str(value)
    return ""


def row_identifiers(row: dict) -> dict[str, str]:
    return {
        field: normalized_value(row, field)
        for field in ["seed_sample_id", "reference_sample_id", "family_id", "label_path", "raw_path"]
    }


def reference_seed_registry_paths() -> list[Path]:
    # 현재 runner 자신의 산출물은 rerun 안정성을 위해 exclusion source에서 제외한다.
    seed_root = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation"
    return sorted(path for path in seed_root.glob("**/seed_registry.csv") if path != SEED_REGISTRY_PATH)


def load_audit_source_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl_rows(path)
    if path.suffix == ".csv":
        return read_csv_rows(path)
    return []


def audit_source_paths() -> dict[str, list[Path]]:
    processed_root = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation"
    analysis_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    split_paths: list[Path] = []
    for name in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        split_paths.extend(processed_root.glob(f"**/{name}"))
    return {
        "interim_seed_registry": reference_seed_registry_paths(),
        "processed_train_dev_test": sorted(split_paths),
        "processed_audit_queue": sorted(processed_root.glob("**/audit_queue.csv")),
        "analysis_tail_memo": sorted(analysis_root.glob("**/exports/*tail*.csv")),
    }


def collect_excluded_rows() -> list[dict[str, str]]:
    # 기존 current/failed/candidate seed registry뿐 아니라 processed split과 tail memo까지 펼쳐 중복 사용을 막는다.
    rows: list[dict[str, str]] = []
    for path in reference_seed_registry_paths():
        rows.extend(load_csv_rows_if_exists(path))
    for source_group, paths in audit_source_paths().items():
        if source_group == "interim_seed_registry":
            continue
        for path in paths:
            for row in load_audit_source_rows(path):
                flattened = row_identifiers(row)
                if any(flattened.values()):
                    rows.append(flattened)
    return rows


def build_seed_row(record: dict) -> dict[str, str]:
    row = ORIGINAL_PB6_BUILD_SEED_ROW(record)
    row["selection_role"] = SEED_SELECTION_ROLE
    row["selection_note"] = (
        f"{SEED_SELECTION_NOTE}; target {FINAL_PACKAGE_TARGET_COUNT} final package를 위해 "
        f"candidate {EXPECTED_CANDIDATE_SEED_COUNT}개를 먼저 고정"
    )
    row["pb6_seed_filter_note"] = SEED_FILTER_NOTE
    row["non_law_scope_note"] = NON_LAW_SCOPE_NOTE
    row["package_candidate_role"] = "candidate_pool"
    row["count_reflection_status"] = "candidate_not_counted"
    return row


def target_label_for_index(index: int) -> str:
    return ["A", "B", "C", "D"][index % 4]


def augment_seed_row(row: dict[str, str], index: int) -> None:
    action, flags = interpretation_base.interpretation_seed_action(row)
    text = " ".join([row.get("transformed_problem", ""), row.get("short_answer", ""), row.get("generated_explanation", "")])
    row["interpretation_seed_action"] = action
    row["interpretation_axis"] = interpretation_base.classify_interpretation_axis(text)
    row["interpretation_risk_flags"] = flags
    row["target_correct_choice"] = target_label_for_index(index)
    row["validator_report_schema_required"] = "예"
    row["source_field_contract_required"] = "예"


def configure_seed_selector() -> None:
    # pb6 selector의 raw/label 연결과 family_id 생성 규칙을 그대로 재사용하되, scope와 quota만 해석례 overgeneration용으로 바꾼다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = RUN_LABEL
    pb6.SEED_ID_PREFIX = SEED_ID_PREFIX
    pb6.SEED_SELECTION_ROLE = SEED_SELECTION_ROLE
    pb6.SEED_SELECTION_NOTE = SEED_SELECTION_NOTE
    pb6.SEED_FILTER_NOTE = SEED_FILTER_NOTE
    pb6.SCOPE_NOTE = (
        f"해석례_QA only; API 전 candidate {EXPECTED_CANDIDATE_SEED_COUNT} / "
        f"final {FINAL_PACKAGE_TARGET_COUNT} seed-spec-wiring check"
    )
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_CANDIDATE_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.PB6_SOURCE_COUNTS = INTERPRETATION_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.OVERLAP_CHECK_LABEL = "no current/candidate/failed/held-out/audit/tail overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        f"`{sum(COUNTED_EXCLUSION_COMPONENTS.values())}`은 current counted objective seed package pool이다.",
        "failed/candidate/intermediate seen seed도 제외해 해석례 package factory pilot의 fresh seed 성격을 유지한다.",
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
    pb6.passes_pb6_seed_filter = interpretation_base.passes_interpretation_seed_filter
    pb6.build_seed_row = build_seed_row


def build_preflight_rows(seed_rows: list[dict[str, str]], exclusion_sets: dict[str, set[str]]) -> list[dict[str, str]]:
    return interpretation_base.build_preflight_rows(seed_rows, exclusion_sets)


def assert_preflight(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    if len(seed_rows) != EXPECTED_CANDIDATE_SEED_COUNT:
        raise RuntimeError(f"interpretation candidate seed count mismatch: {len(seed_rows)}")
    if dict(doc_counts) != EXPECTED_DOC_TYPE_COUNTS:
        raise RuntimeError(f"interpretation doc type mismatch: {dict(doc_counts)}")
    for key, expected in EXPECTED_LANE_BY_DOC.items():
        if lane_by_doc.get(key, 0) != expected:
            raise RuntimeError(f"interpretation lane mismatch: {key}={lane_by_doc.get(key, 0)}")
    for source_subset, expected in INTERPRETATION_SOURCE_COUNTS.items():
        if source_counts.get(source_subset, 0) != expected:
            raise RuntimeError(f"interpretation source split mismatch: {source_subset}={source_counts.get(source_subset, 0)}")
    if dict(target_counts) != CANDIDATE_TARGET_LABEL_COUNTS:
        raise RuntimeError(f"interpretation target label mismatch: {dict(target_counts)}")
    for row in preflight_rows:
        overlap_flags = [
            row["seed_sample_id_duplicate_in_batch"],
            row["reference_sample_id_duplicate_in_batch"],
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
            row["seed_sample_id_overlap_with_prior"],
            row["reference_sample_id_overlap_with_prior"],
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        if "예" in overlap_flags:
            raise RuntimeError(f"interpretation overgeneration preflight overlap failed: {row['seed_sample_id']}")


def write_preflight_report(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["interpretation_seed_action"] for row in seed_rows)
    axis_counts = Counter(row["interpretation_axis"] for row in seed_rows)
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
        f"- interpretation_seed_action_counts: `{dict(action_counts)}`",
        f"- interpretation_axis_counts: `{dict(axis_counts)}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        f"| candidate seed count is {EXPECTED_CANDIDATE_SEED_COUNT} | `pass` |",
        f"| final package target is {FINAL_PACKAGE_TARGET_COUNT} | `pass` |",
        "| doc type is 해석례_QA only | `pass` |",
        f"| candidate source split is `{dict(source_counts)}` | `pass` |",
        f"| candidate lane split is `{dict(lane_counts)}` | `pass` |",
        f"| candidate target label schedule is `{dict(target_counts)}` | `pass` |",
        "| no batch duplicate | `pass` |",
        "| no prior/candidate/failed/held-out/audit/tail overlap | `pass` |",
        "| source field contract fields checked | `pass` |",
        "| validator report schema fields defined | `pass` |",
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
            "interpretation_axis": row["interpretation_axis"],
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
            {
                "quota_type": "final_gate",
                "quota_key": "validator_accept_export_ready",
                "target_count": str(FINAL_PACKAGE_TARGET_COUNT),
            },
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


def source_field_contract_row(row: dict[str, str]) -> dict[str, str]:
    return interpretation_base.source_field_contract_row(row)


def write_source_field_contract(seed_rows: list[dict[str, str]]) -> dict[str, int]:
    rows = [source_field_contract_row(row) for row in seed_rows]
    write_csv(SOURCE_FIELD_CONTRACT_CSV_PATH, rows, list(rows[0].keys()))
    pass_count = sum(1 for row in rows if row["source_field_contract_pass"] == "예")
    raw_exists_count = sum(1 for row in rows if row["raw_path_exists"] == "예")
    label_exists_count = sum(1 for row in rows if row["label_path_exists"] == "예")
    lines = [
        f"# source field contract `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(rows)}`",
        f"- source_field_contract_pass: `{pass_count}`",
        f"- raw_path_exists: `{raw_exists_count}`",
        f"- label_path_exists: `{label_exists_count}`",
        "",
        "## per-row contract",
        "| seed | source_subset | source_text_field | raw_exists | gold_reference | fallback | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{seed_sample_id}` | `{source_subset}` | `{source_text_field}` | `{raw_path_exists}` | `{gold_reference_explanation_present}` | `{fallback_grounding_present}` | `{source_field_contract_pass}` |".format(
                **row
            )
        )
    write_text(SOURCE_FIELD_CONTRACT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SOURCE_FIELD_CONTRACT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SOURCE_FIELD_CONTRACT_MD_PATH, RUN_INPUTS_DIR)
    if pass_count != len(rows):
        raise RuntimeError(f"source field contract failed: {pass_count}/{len(rows)}")
    return {
        "source_field_contract_pass_count": pass_count,
        "raw_path_exists_count": raw_exists_count,
        "label_path_exists_count": label_exists_count,
    }


def write_package_compiler_contract() -> None:
    contract = {
        "contract_name": VERSION_TAG,
        "api_calls": 0,
        "candidate_generation_target": EXPECTED_CANDIDATE_SEED_COUNT,
        "final_package_target": FINAL_PACKAGE_TARGET_COUNT,
        "count_state_before_reviewer_signoff": "candidate_not_counted",
        "candidate_quota": {
            "doc_type_counts": EXPECTED_DOC_TYPE_COUNTS,
            "source_subset_counts": INTERPRETATION_SOURCE_COUNTS,
            "lane_counts": expected_lane_counts(),
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
    }
    write_json(PACKAGE_COMPILER_CONTRACT_JSON_PATH, contract)
    lines = [
        f"# package compiler contract `{VERSION_TAG}`",
        "",
        f"- candidate_generation_target: `{EXPECTED_CANDIDATE_SEED_COUNT}`",
        f"- final_package_target: `{FINAL_PACKAGE_TARGET_COUNT}`",
        "- count_state_before_reviewer_signoff: `candidate_not_counted`",
        "- strict_final_gate: `hard/soft/audit 0`, `validator accept/export-ready only`, `metadata/shuffle mismatch 0`",
        "- source/validator contract: `해석례_QA 01/02 유권해석 expansion`과 `03/04 해석례 generalization`을 breakout으로 유지",
        "",
    ]
    write_text(PACKAGE_COMPILER_CONTRACT_MD_PATH, "\n".join(lines))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(PACKAGE_COMPILER_CONTRACT_JSON_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(PACKAGE_COMPILER_CONTRACT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry() -> list[dict[str, str]]:
    configure_seed_selector()
    pb6.pb4.pb3.base.ensure_dirs(INTERIM_DIR, RUN_DIR, RUN_INPUTS_DIR, RUN_EXPORTS_DIR)
    records, exclusion_sets = pb6.select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["source_subset"], row["seed_sample_id"]))
    for index, row in enumerate(seed_rows):
        augment_seed_row(row, index)
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
    seed_identifier_values = {
        field: {row_identifiers(row)[field] for row in seed_rows if row_identifiers(row)[field]}
        for field in ["seed_sample_id", "reference_sample_id", "family_id", "label_path", "raw_path"]
    }
    audit_rows: list[dict[str, str]] = []
    summary_counts: dict[str, int] = {}
    for source_group, paths in audit_source_paths().items():
        group_any_overlap = 0
        group_row_count = 0
        for path in paths:
            rows = load_audit_source_rows(path)
            group_row_count += len(rows)
            overlap_counts = {field: 0 for field in seed_identifier_values}
            any_overlap_count = 0
            for row in rows:
                identifiers = row_identifiers(row)
                row_has_overlap = False
                for field, seed_values in seed_identifier_values.items():
                    if identifiers[field] and identifiers[field] in seed_values:
                        overlap_counts[field] += 1
                        row_has_overlap = True
                if row_has_overlap:
                    any_overlap_count += 1
            group_any_overlap += any_overlap_count
            audit_rows.append(
                {
                    "source_group": source_group,
                    "source_path": str(path.relative_to(PROJECT_ROOT)),
                    "row_count": str(len(rows)),
                    "seed_sample_id_overlap_count": str(overlap_counts["seed_sample_id"]),
                    "reference_sample_id_overlap_count": str(overlap_counts["reference_sample_id"]),
                    "family_id_overlap_count": str(overlap_counts["family_id"]),
                    "label_path_overlap_count": str(overlap_counts["label_path"]),
                    "raw_path_overlap_count": str(overlap_counts["raw_path"]),
                    "any_overlap_count": str(any_overlap_count),
                }
            )
        summary_counts[f"{source_group}_source_count"] = len(paths)
        summary_counts[f"{source_group}_row_count"] = group_row_count
        summary_counts[f"{source_group}_overlap_count"] = group_any_overlap
    write_csv(
        EXCLUSION_AUDIT_CSV_PATH,
        audit_rows,
        [
            "source_group",
            "source_path",
            "row_count",
            "seed_sample_id_overlap_count",
            "reference_sample_id_overlap_count",
            "family_id_overlap_count",
            "label_path_overlap_count",
            "raw_path_overlap_count",
            "any_overlap_count",
        ],
    )
    lines = [
        f"# exclusion audit `{VERSION_TAG}`",
        "",
        "## summary",
        "| source_group | source_count | row_count | overlap_count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source_group in audit_source_paths():
        source_count = summary_counts[f"{source_group}_source_count"]
        row_count = summary_counts[f"{source_group}_row_count"]
        overlap_count = summary_counts[f"{source_group}_overlap_count"]
        lines.append(f"| `{source_group}` | `{source_count}` | `{row_count}` | `{overlap_count}` |")
    lines.extend(
        [
            "",
            "## interpretation",
            f"- `overlap_count = 0` means the current {EXPECTED_CANDIDATE_SEED_COUNT} interpretation overgeneration preflight seeds did not overlap with that source group by `seed_sample_id`, `reference_sample_id`, `family_id`, `label_path`, or `raw_path`.",
            f"- The named `{sum(COUNTED_EXCLUSION_COMPONENTS.values())}` pool is the current counted/package seed pool; this audit also checks failed/candidate/intermediate artifacts to avoid hidden seed reuse before API execution.",
        ]
    )
    write_text(EXCLUSION_AUDIT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_MD_PATH, RUN_INPUTS_DIR)
    return summary_counts


def write_run_manifest(
    seed_rows: list[dict[str, str]],
    exclusion_audit_summary: dict[str, int],
    source_contract_summary: dict[str, int],
) -> None:
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["interpretation_seed_action"] for row in seed_rows)
    axis_counts = Counter(row["interpretation_axis"] for row in seed_rows)
    manifest = {
        "run_name": RUN_NAME,
        "version_tag": VERSION_TAG,
        "run_type": "no_api_seed_spec_wiring_check",
        "api_calls": 0,
        "objective_current_count_reference": CURRENT_OBJECTIVE_COUNT,
        "counted_exclusion_components": COUNTED_EXCLUSION_COMPONENTS,
        "current_next_seed_planning_exclusion_pool": sum(COUNTED_EXCLUSION_COMPONENTS.values()),
        "actual_reference_seed_registry_path_count": len(reference_seed_registry_paths()),
        "candidate_seed_count": len(seed_rows),
        "final_package_target_count": FINAL_PACKAGE_TARGET_COUNT,
        "doc_type_counts": {"해석례_QA": len(seed_rows)},
        "source_subset_counts": dict(source_counts),
        "lane_counts": dict(lane_counts),
        "candidate_target_label_counts": dict(target_counts),
        "final_target_label_counts": FINAL_TARGET_LABEL_COUNTS,
        "interpretation_seed_action_counts": dict(action_counts),
        "interpretation_axis_counts": dict(axis_counts),
        "count_state": "candidate_not_counted",
        "preflight_result": {
            "passed": True,
            "api_execution_allowed_by_this_run": False,
            "next_stop_line": f"reviewer_signoff_then_candidate_{EXPECTED_CANDIDATE_SEED_COUNT}_api_execution",
            "interim_seed_registry_overlap_count": exclusion_audit_summary["interim_seed_registry_overlap_count"],
            "processed_split_overlap_count": exclusion_audit_summary["processed_train_dev_test_overlap_count"],
            "audit_queue_overlap_count": exclusion_audit_summary["processed_audit_queue_overlap_count"],
            "tail_memo_overlap_count": exclusion_audit_summary["analysis_tail_memo_overlap_count"],
            "source_field_contract_pass_count": source_contract_summary["source_field_contract_pass_count"],
        },
        "future_api_pilot_contract": {
            "candidate_generation": EXPECTED_CANDIDATE_SEED_COUNT,
            "final_package": FINAL_PACKAGE_TARGET_COUNT,
            "accepted_final_gate": "hard_soft_audit_0_validator_accept_export_ready_metadata_shuffle_0",
            "count_reflection": "not_counted_until_reviewer_signoff",
            "required_breakout": [
                "source_subset",
                "sampling_lane",
                "target_correct_choice",
                "interpretation_seed_action",
                "interpretation_axis",
                "tail_class",
                "quota_surplus",
            ],
        },
        "exclusion_audit_summary": exclusion_audit_summary,
        "source_field_contract_summary": source_contract_summary,
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
            "source_field_contract_csv": str(SOURCE_FIELD_CONTRACT_CSV_PATH),
            "source_field_contract_md": str(SOURCE_FIELD_CONTRACT_MD_PATH),
            "package_compiler_contract_json": str(PACKAGE_COMPILER_CONTRACT_JSON_PATH),
            "package_compiler_contract_md": str(PACKAGE_COMPILER_CONTRACT_MD_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def main() -> None:
    refresh_paths()
    seed_rows = build_seed_registry()
    exclusion_audit_summary = write_exclusion_audit(seed_rows)
    source_contract_summary = write_source_field_contract(seed_rows)
    write_run_manifest(seed_rows, exclusion_audit_summary, source_contract_summary)
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
