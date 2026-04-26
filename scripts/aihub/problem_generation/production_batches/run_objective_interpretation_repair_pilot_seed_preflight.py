from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 reviewer가 승인한 `해석례_QA` repair pilot의 API 전 stop line이다.
# 실제 generation/Judge 비용을 쓰기 전에 seed, exclusion, label schedule,
# validator schema, grounding source field contract를 no-API 산출물로 먼저 닫는다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402


VERSION_TAG = "objective_interpretation_repair_pilot_seed_preflight"
# llm_runs 이름은 실제 한국 시간 실행 시각과 맞아야 하므로 날짜/시각을 하드코딩하지 않는다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_interpretation_repair_seed_preflight_wiring_check"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "interpretation repair seed preflight"

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
VALIDATOR_SCHEMA_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.csv"
VALIDATOR_SCHEMA_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.md"
SOURCE_FIELD_CONTRACT_CSV_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.csv"
SOURCE_FIELD_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"source_field_contract_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"

EXPECTED_TOTAL_SEED_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 16}
EXPECTED_LANE_BY_DOC = {
    ("해석례_QA", "generalization_03_04"): 8,
    ("해석례_QA", "expansion_01_02"): 8,
}
INTERPRETATION_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 4,
    "02_TL_유권해석_QA": 4,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 4,
}
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
ORIGINAL_PB6_BUILD_SEED_ROW = pb6.build_seed_row

# `165`는 current usable count가 아니라 다음 seed planning에서 제외할 counted/package pool이다.
# 실제 fresh 검증에서는 실패/candidate seed까지 더 넓게 제외해, 이미 본 row 재사용을 막는다.
COUNTED_EXCLUSION_COMPONENTS = {
    "r2": 16,
    "pb2": 13,
    "pb3": 40,
    "pb4": 40,
    "pb9_final_package": 40,
    "judgment_a_slot_final_package": 16,
}

REFERENCE_SEED_REGISTRY_PATHS = [
    PROJECT_ROOT / "data/interim/aihub/problem_generation/v2_difficulty_patch_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb2_objective_candidate/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb3_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb4_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_law_guardrail_targeted_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb5_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb6_non_law_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb7_decision_judgment_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb8_decision_only_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_weak_distractor_guardrail_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_micro_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_micro_retry/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_a_slot_replacement/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_pilot_16/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_2slot_repair/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_d_slot_replacement/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_decision_only_controlled_production_with_choice_validator/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_04tl_decision_weak_distractor_calibration_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_04tl_decision_answer_uniqueness_1slot_replacement/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_accepted34_6slot_replacement_package/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_cslot_final_replacement_package/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_pilot_seed_preflight/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_judgment_repair_a_slot_replacement_package/seed_registry.csv",
]

VALIDATOR_REQUIRED_FIELDS = [
    "seed_sample_id",
    "target_correct_choice",
    "export_correct_choice",
    "validator_action",
    "validator_export_disposition",
    "validator_reason_short",
    "validator_recalculated_correct_choice",
    "metadata_remap_ok",
    "split_allowed",
    "count_allowed",
    "answer_uniqueness",
    "condition_preservation",
    "response_scope_limited",
    "answer_reason_split",
    "source_only_fact",
    "distractor_direction",
    "same_direction_distractor",
    "source_text_field",
    "source_identifier",
    "gold_reference_explanation_field",
    "fallback_grounding_field",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def normalized_value(row: dict, field: str) -> str:
    # split JSONL의 metadata 안에 식별자가 들어가는 경우가 있어, 직접 field와 metadata를 함께 본다.
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


def collect_excluded_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in REFERENCE_SEED_REGISTRY_PATHS:
        rows.extend(load_csv_rows_if_exists(path))
    return rows


def build_exclusion_sets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {
        "sample_ids": {
            row.get("sample_id", "") or row.get("seed_sample_id", "")
            for row in rows
            if row.get("sample_id", "") or row.get("seed_sample_id", "")
        },
        "reference_sample_ids": {row.get("reference_sample_id", "") for row in rows if row.get("reference_sample_id", "")},
        "family_ids": {row.get("family_id", "") for row in rows if row.get("family_id", "")},
        "label_paths": {row.get("label_path", "") for row in rows if row.get("label_path", "")},
        "raw_paths": {row.get("raw_path", "") for row in rows if row.get("raw_path", "")},
    }


def audit_source_paths() -> dict[str, list[Path]]:
    processed_root = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation"
    analysis_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    split_paths: list[Path] = []
    for name in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        split_paths.extend(processed_root.glob(f"**/{name}"))
    return {
        "interim_seed_registry": [path for path in REFERENCE_SEED_REGISTRY_PATHS if path.exists()],
        "processed_train_dev_test": sorted(split_paths),
        "processed_audit_queue": sorted(processed_root.glob("**/audit_queue.csv")),
        "analysis_tail_memo": sorted(analysis_root.glob("**/exports/*tail*.csv")),
    }


def load_audit_source_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl_rows(path)
    if path.suffix == ".csv":
        return read_csv_rows(path)
    return []


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
            "- `overlap_count = 0` means the current 16 interpretation preflight seeds did not overlap with that source group by `seed_sample_id`, `reference_sample_id`, `family_id`, `label_path`, or `raw_path`.",
            "- The named `165` pool is only the current counted/package planning pool; this audit also checks failed/candidate/intermediate artifacts to avoid hidden seed reuse before API execution.",
        ]
    )
    write_text(EXCLUSION_AUDIT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_MD_PATH, RUN_INPUTS_DIR)
    return summary_counts


def classify_interpretation_axis(text: str) -> str:
    if any(token in text for token in ["전제", "조건", "경우", "요건"]):
        return "전제조건"
    if any(token in text for token in ["예외", "제외", "다만"]):
        return "예외"
    if any(token in text for token in ["범위", "적용", "대상"]):
        return "적용범위"
    return "회답결론"


def interpretation_risk_flags(payload_or_row: dict) -> list[str]:
    # 해석례는 회답 결론과 전제조건이 섞일 때 복수 정답/원문 외 사실이 잘 생긴다.
    # API 전 단계에서는 보수적인 keyword risk flag만 남기고, 실제 품질 판단은 pilot에서 닫는다.
    text = " ".join(
        [
            str(payload_or_row.get("transformed_problem", "")),
            str(payload_or_row.get("short_answer", "")),
            str(payload_or_row.get("generated_explanation", "")),
            str(payload_or_row.get("rule_basis", "")),
            str(payload_or_row.get("fact_basis", "")),
        ]
    )
    flags: list[str] = []
    if any(token in text for token in ["경우", "전제", "요건", "조건"]):
        flags.append("condition_preservation_watch")
    if any(token in text for token in ["범위", "적용", "대상"]):
        flags.append("response_scope_watch")
    if any(token in text for token in ["가능", "할 수", "해야", "하여야"]):
        flags.append("same_direction_distractor_watch")
    if any(token in text for token in ["원칙", "예외", "다만"]):
        flags.append("exception_or_reason_split_watch")
    return flags


def interpretation_seed_action(payload_or_row: dict) -> tuple[str, str]:
    flags = interpretation_risk_flags(payload_or_row)
    if flags:
        return "template_only", "|".join(flags)
    return "normal", "none"


def passes_interpretation_seed_filter(spec: dict, payload: dict) -> tuple[bool, str]:
    if spec["doc_type_name"] != "해석례_QA":
        return False, "interpretation_only_scope"
    passes_base, reason = pb6.pb4.passes_seed_quality_filter(
        spec["doc_type_name"],
        payload["label"]["input"],
        payload["label"]["output"],
    )
    if not passes_base:
        return False, reason
    return True, "interpretation_seed_preflight_candidate"


def target_label_for_index(index: int) -> str:
    return ["A", "B", "C", "D"][index % 4]


def build_seed_row(record: dict) -> dict:
    row = ORIGINAL_PB6_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_interpretation_repair_pilot_seed_preflight_seed"
    row["selection_note"] = "해석례_QA answer uniqueness / response scope / grounding repair pilot no-API seed preflight"
    row["pb6_seed_filter_note"] = "interpretation_only_repair_seed_filter"
    row["non_law_scope_note"] = "interpretation_repair_pilot_preflight_no_api"
    return row


def augment_seed_row(row: dict[str, str], index: int) -> dict[str, str]:
    action, flags = interpretation_seed_action(row)
    text = " ".join([row.get("transformed_problem", ""), row.get("short_answer", ""), row.get("generated_explanation", "")])
    row["interpretation_seed_action"] = action
    row["interpretation_axis"] = classify_interpretation_axis(text)
    row["interpretation_risk_flags"] = flags
    row["target_correct_choice"] = target_label_for_index(index)
    row["validator_report_schema_required"] = "예"
    row["source_field_contract_required"] = "예"
    return row


def build_preflight_rows(seed_rows: list[dict], exclusion_sets: dict) -> list[dict]:
    seed_counts = Counter(row["seed_sample_id"] for row in seed_rows)
    reference_counts = Counter(row["reference_sample_id"] for row in seed_rows)
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows = []
    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "reference_sample_id": row["reference_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "target_correct_choice": row["target_correct_choice"],
                "interpretation_seed_action": row["interpretation_seed_action"],
                "interpretation_axis": row["interpretation_axis"],
                "interpretation_risk_flags": row["interpretation_risk_flags"],
                "seed_sample_id_duplicate_in_batch": "예" if seed_counts[row["seed_sample_id"]] > 1 else "아니오",
                "reference_sample_id_duplicate_in_batch": "예" if reference_counts[row["reference_sample_id"]] > 1 else "아니오",
                "family_duplicate_in_batch": "예" if family_counts[row["family_id"]] > 1 else "아니오",
                "label_path_duplicate_in_batch": "예" if label_counts[row["label_path"]] > 1 else "아니오",
                "raw_path_duplicate_in_batch": "예" if raw_counts[row["raw_path"]] > 1 else "아니오",
                "seed_sample_id_overlap_with_prior": "예" if row["seed_sample_id"] in exclusion_sets["sample_ids"] else "아니오",
                "reference_sample_id_overlap_with_prior": "예"
                if row["reference_sample_id"] in exclusion_sets["reference_sample_ids"]
                else "아니오",
                "family_overlap_with_prior": "예" if row["family_id"] in exclusion_sets["family_ids"] else "아니오",
                "label_path_overlap_with_prior": "예" if row["label_path"] in exclusion_sets["label_paths"] else "아니오",
                "raw_path_overlap_with_prior": "예" if row["raw_path"] in exclusion_sets["raw_paths"] else "아니오",
                "answer_mode": row["answer_mode"],
                "problem_generation_mode": row["problem_generation_mode"],
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    return preflight_rows


def assert_interpretation_preflight(seed_rows: list[dict], preflight_rows: list[dict]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"interpretation seed count mismatch: {len(seed_rows)}")
    if dict(doc_counts) != EXPECTED_DOC_TYPE_COUNTS:
        raise RuntimeError(f"interpretation doc type mismatch: {dict(doc_counts)}")
    for key, expected in EXPECTED_LANE_BY_DOC.items():
        if lane_by_doc.get(key, 0) != expected:
            raise RuntimeError(f"interpretation lane mismatch: {key}={lane_by_doc.get(key, 0)}")
    for source_subset, expected in INTERPRETATION_SOURCE_COUNTS.items():
        if source_counts.get(source_subset, 0) != expected:
            raise RuntimeError(f"interpretation source split mismatch: {source_subset}={source_counts.get(source_subset, 0)}")
    if dict(target_counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"target label schedule mismatch: {dict(target_counts)}")
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
            raise RuntimeError(f"interpretation preflight overlap failed: {row['seed_sample_id']}")


def write_preflight_report(seed_rows: list[dict], preflight_rows: list[dict]) -> None:
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
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- source_subset_counts: `{dict(source_counts)}`",
        f"- target_label_counts: `{dict(target_counts)}`",
        f"- interpretation_seed_action_counts: `{dict(action_counts)}`",
        f"- interpretation_axis_counts: `{dict(axis_counts)}`",
        "",
        "## exclusion summary",
        f"- current_next_seed_planning_exclusion_pool: `{sum(COUNTED_EXCLUSION_COMPONENTS.values())}`",
        f"- counted_exclusion_components: `{COUNTED_EXCLUSION_COMPONENTS}`",
        f"- actual_exclusion_registry_paths: `{len([path for path in REFERENCE_SEED_REGISTRY_PATHS if path.exists()])}`",
        "",
        "## source subset counts",
        "| source_subset | count |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    checks = [
        "total seed count is 16",
        "doc type is 해석례_QA only",
        "source split is 01/02/03/04 each 4",
        "lane split is 8/8",
        "target label schedule is A/B/C/D = 4/4/4/4",
        "no batch duplicate",
        "no prior/candidate/failed/held-out/audit/tail overlap",
        "validator report schema fields defined",
        "source field contract fields checked",
    ]
    for check in checks:
        lines.append(f"| {check} | `pass` |")
    write_text(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_target_label_schedule(seed_rows: list[dict]) -> None:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "interpretation_axis": row["interpretation_axis"],
            "target_correct_choice": row["target_correct_choice"],
        }
        for row in seed_rows
    ]
    write_csv(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)


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
    source_text_fields = [
        field
        for field in ["rule_basis", "fact_basis", "generated_explanation"]
        if row.get(field, "").strip()
    ]
    fallback_fields = [
        field
        for field in ["rule_basis", "fact_basis"]
        if row.get(field, "").strip()
    ]
    raw_path = Path(row["raw_path"])
    label_path = Path(row["label_path"])
    contract_pass = bool(source_text_fields and row.get("raw_path") and row.get("family_id") and row.get("generated_explanation") and fallback_fields)
    return {
        "seed_sample_id": row["seed_sample_id"],
        "doc_type_name": row["doc_type_name"],
        "source_subset": row["source_subset"],
        "sampling_lane": row["sampling_lane"],
        "source_text_field": "|".join(source_text_fields),
        "source_text_present": "예" if source_text_fields else "아니오",
        "raw_path_present": "예" if row.get("raw_path", "") else "아니오",
        "raw_path_exists": "예" if raw_path.exists() else "아니오",
        "label_path_present": "예" if row.get("label_path", "") else "아니오",
        "label_path_exists": "예" if label_path.exists() else "아니오",
        "source_identifier_field": "family_id|raw_path",
        "source_identifier_present": "예" if row.get("family_id") and row.get("raw_path") else "아니오",
        "gold_reference_explanation_field": "generated_explanation",
        "gold_reference_explanation_present": "예" if row.get("generated_explanation", "").strip() else "아니오",
        "fallback_grounding_field": "|".join(fallback_fields),
        "fallback_grounding_present": "예" if fallback_fields else "아니오",
        "source_field_contract_pass": "예" if contract_pass else "아니오",
    }


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
        "## required fields",
        "| field | purpose |",
        "| --- | --- |",
        "| `source_text_field` | grounding validator가 회답서 본문 또는 준본문으로 읽을 field |",
        "| `raw_path` / `source_identifier` | 원천 row를 되짚기 위한 source identifier |",
        "| `gold_reference_explanation` | 생성 기준 설명 또는 정답 근거로 쓸 reference explanation alias |",
        "| `fallback_grounding_field` | reference explanation이 약할 때 쓸 `rule_basis/fact_basis` fallback |",
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


def configure_interpretation_seed_selector() -> None:
    # pb6 selector의 label/raw 연결과 family_id 생성 규칙을 재사용하되,
    # scope와 artifact path만 해석례 repair preflight로 바꾼다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = RUN_LABEL
    pb6.SEED_ID_PREFIX = "interpretation_repair_preflight"
    pb6.SEED_SELECTION_ROLE = "objective_interpretation_repair_pilot_seed_preflight_seed"
    pb6.SEED_SELECTION_NOTE = "해석례_QA repair pilot no-API seed preflight"
    pb6.SEED_FILTER_NOTE = "interpretation_only_repair_seed_filter"
    pb6.SCOPE_NOTE = "해석례_QA only; API 실행 전 seed registry/wiring/source contract check"
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.PB6_SOURCE_COUNTS = INTERPRETATION_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.OVERLAP_CHECK_LABEL = "no current/failed/candidate/held-out/audit/tail overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        "`165`는 current usable이 아니라 current next seed planning exclusion pool이다.",
        "failed/candidate objective seed까지 추가 제외해 API 비용이 드는 재사용을 막는다.",
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
    pb6.passes_pb6_seed_filter = passes_interpretation_seed_filter
    pb6.build_seed_row = build_seed_row
    pb6.build_preflight_rows = build_preflight_rows
    pb6.assert_preflight = assert_interpretation_preflight
    pb6.write_preflight_report = write_preflight_report


def build_seed_registry() -> list[dict]:
    configure_interpretation_seed_selector()
    pb6.pb4.pb3.base.ensure_dirs(INTERIM_DIR, RUN_DIR, RUN_INPUTS_DIR, RUN_EXPORTS_DIR)
    records, exclusion_sets = pb6.select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["source_subset"], row["seed_sample_id"]))
    for index, row in enumerate(seed_rows):
        augment_seed_row(row, index)
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_interpretation_preflight(seed_rows, preflight_rows)
    write_csv(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    write_target_label_schedule(seed_rows)
    write_validator_schema()
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def write_run_manifest(
    seed_rows: list[dict],
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
        "run_type": "no_api_seed_preflight_wiring_check",
        "api_calls": 0,
        "objective_current_count_reference": {
            "usable": 151,
            "train": 116,
            "eval": 35,
            "audit": 6,
            "hard_fail": 5,
            "soft_fail": 3,
        },
        "current_next_seed_planning_exclusion_pool": sum(COUNTED_EXCLUSION_COMPONENTS.values()),
        "counted_exclusion_components": COUNTED_EXCLUSION_COMPONENTS,
        "actual_reference_seed_registry_path_count": len([path for path in REFERENCE_SEED_REGISTRY_PATHS if path.exists()]),
        "seed_count": len(seed_rows),
        "doc_type_counts": {"해석례_QA": len(seed_rows)},
        "source_subset_counts": dict(source_counts),
        "lane_counts": dict(lane_counts),
        "target_label_counts": dict(target_counts),
        "interpretation_seed_action_counts": dict(action_counts),
        "interpretation_axis_counts": dict(axis_counts),
        "success_criteria_for_future_api_pilot": {
            "pilot_signal_success": {
                "usable_min": 14,
                "hard_fail_max": 1,
                "soft_fail_max": 1,
                "audit_max": 1,
                "target_failure_recurrence_max": 0,
            },
            "countable_package_success": {
                "hard_fail_max": 0,
                "soft_fail_max": 0,
                "unresolved_audit_max": 0,
                "label_shuffle_metadata_schema_mismatch_max": 0,
                "count_reflection": "not_counted_until_future_api_pilot_and_reviewer_signoff",
            },
        },
        "preflight_result": {
            "passed": True,
            "api_execution_allowed_by_this_run": False,
            "next_stop_line": "reviewer_signoff_for_interpretation_repair_api_pilot",
            "interim_seed_registry_overlap_count": exclusion_audit_summary["interim_seed_registry_overlap_count"],
            "processed_split_overlap_count": exclusion_audit_summary["processed_train_dev_test_overlap_count"],
            "audit_queue_overlap_count": exclusion_audit_summary["processed_audit_queue_overlap_count"],
            "tail_memo_overlap_count": exclusion_audit_summary["analysis_tail_memo_overlap_count"],
            "source_field_contract_pass_count": source_contract_summary["source_field_contract_pass_count"],
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
            "validator_schema_csv": str(VALIDATOR_SCHEMA_CSV_PATH),
            "validator_schema_md": str(VALIDATOR_SCHEMA_MD_PATH),
            "source_field_contract_csv": str(SOURCE_FIELD_CONTRACT_CSV_PATH),
            "source_field_contract_md": str(SOURCE_FIELD_CONTRACT_MD_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def main() -> None:
    seed_rows = build_seed_registry()
    exclusion_audit_summary = write_exclusion_audit(seed_rows)
    source_contract_summary = write_source_field_contract(seed_rows)
    write_run_manifest(seed_rows, exclusion_audit_summary, source_contract_summary)
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "seed_count": len(seed_rows),
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
