from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `pb9` first API run은 34개 export-ready row와 6개 결손 slot으로 국소화됐으므로,
# 전체 40개 재실행 대신 accepted 34개를 보존하고 fresh replacement 6개로 final package를 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as pb8  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_objective_pb9_decision_only as pb9_smoke  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_objective_pb9_decision_only_api as pb9_api  # noqa: E402


VERSION_TAG = "pb9_accepted34_6slot_replacement_package"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_pb9_tail_6slot_salvage_package"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb8.pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PB9_RUN_NAME = "2026-04-26_031559_pb9_decision_only_controlled_production_with_choice_validator_objective_r2_decision_only_api_execution"
SOURCE_PB9_RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / SOURCE_PB9_RUN_NAME
SOURCE_PB9_MERGED_PATH = (
    SOURCE_PB9_RUN_DIR / "merged" / "merged_problem_scores_pb9_decision_only_controlled_production_with_choice_validator.csv"
)
SOURCE_PB9_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb9_decision_only_controlled_production_with_choice_validator"
    / "seed_registry.csv"
)
REFERENCE_04TL_CALIBRATION_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb9_04tl_decision_weak_distractor_calibration_pilot"
    / "seed_registry.csv"
)
REFERENCE_04TL_CSLOT_REPLACEMENT_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb9_04tl_decision_answer_uniqueness_1slot_replacement"
    / "seed_registry.csv"
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
REPLACEMENT_PLAN_CSV_PATH = RUN_EXPORTS_DIR / f"replacement_plan_{VERSION_TAG}.csv"
REPLACEMENT_PLAN_MD_PATH = RUN_EXPORTS_DIR / f"replacement_plan_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"replacement_merged_before_validator_{VERSION_TAG}.csv"
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
VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

EXPECTED_TOTAL_SEED_COUNT = 6
EXPECTED_PRESERVED_COUNT = 34
EXPECTED_FINAL_PACKAGE_COUNT = 40
SUCCESS_USABLE_MIN = 40
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 0
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
REPLACEMENT_TARGET_LABEL_COUNTS = {"B": 2, "C": 3, "D": 1}
REPLACEMENT_SEED_ID_PREFIX = "pb9_replacement"
EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 6}
EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "expansion_01_02"): 1,
    ("결정례_QA", "generalization_03_04"): 5,
}
PB9_REPLACEMENT_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 1,
    "04_TL_결정례_QA": 5,
}
BATCH_STATUS = "pb9_salvage_candidate_not_counted"
COUNT_REFLECTION_STATUS = "not_counted"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

REPLACEMENT_TARGETS = [
    {
        "failed_seed_sample_id": "pb9_decision_003",
        "source_subset": "01_TL_심결례_QA",
        "sampling_lane": "expansion_01_02",
        "target_correct_choice": "C",
        "tail_class": "simple_recall_weak_distractor",
    },
    {
        "failed_seed_sample_id": "pb9_decision_030",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "B",
        "tail_class": "04tl_weak_distractor",
    },
    {
        "failed_seed_sample_id": "pb9_decision_031",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "C",
        "tail_class": "04tl_weak_distractor",
    },
    {
        "failed_seed_sample_id": "pb9_decision_034",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "B",
        "tail_class": "form_ending_audit",
    },
    {
        "failed_seed_sample_id": "pb9_decision_036",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "D",
        "tail_class": "04tl_weak_distractor",
    },
    {
        "failed_seed_sample_id": "pb9_decision_039",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "C",
        "tail_class": "04tl_weak_distractor",
    },
]

VALIDATOR_SUMMARY: dict[str, object] = {}
REPLACEMENT_PLAN_ROWS: list[dict[str, str]] = []


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def accepted_pb9_rows() -> list[dict[str, str]]:
    rows = selected_rows(read_csv_rows(SOURCE_PB9_MERGED_PATH))
    accepted = [
        dict(row)
        for row in rows
        if row.get("validator_export_disposition") == "export_ready"
        and row.get("train_eligible") == "예"
        and row.get("audit_required") == "아니오"
        and row.get("final_status") == "pass"
    ]
    if len(accepted) != EXPECTED_PRESERVED_COUNT:
        raise RuntimeError(f"pb9 accepted row count must be {EXPECTED_PRESERVED_COUNT}: {len(accepted)}")
    return accepted


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


def collect_excluded_rows() -> list[dict[str, str]]:
    # successful/failed/informative seed를 모두 seen seed로 제외해야 salvage가 fresh replacement package로 남는다.
    rows = pb9_smoke.collect_excluded_rows()
    rows.extend(load_csv_rows_if_exists(SOURCE_PB9_SEED_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(REFERENCE_04TL_CALIBRATION_SEED_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(REFERENCE_04TL_CSLOT_REPLACEMENT_SEED_REGISTRY_PATH))
    for source_row in selected_rows(read_csv_rows(SOURCE_PB9_MERGED_PATH)):
        if source_row.get("seed_sample_id") in {target["failed_seed_sample_id"] for target in REPLACEMENT_TARGETS}:
            rows.append(
                {
                    "seed_sample_id": source_row.get("seed_sample_id", ""),
                    "reference_sample_id": source_row.get("reference_sample_id", ""),
                    "family_id": source_row.get("family_id", ""),
                    "label_path": source_row.get("label_path", ""),
                    "raw_path": source_row.get("raw_path", ""),
                }
            )
    return rows


def add_record_to_exclusion_sets(record: dict[str, str], exclusion_sets: dict[str, set[str]]) -> None:
    # 같은 salvage package 안에서 family/raw/label 중복이 생기면 final package 해석이 흐려진다.
    exclusion_sets["sample_ids"].add(record["sample_id"])
    exclusion_sets["family_ids"].add(record["family_id"])
    exclusion_sets["label_paths"].add(record["label_path"])
    exclusion_sets["raw_paths"].add(record["raw_path"])


def matching_specs(source_subset: str, sampling_lane: str) -> list[dict[str, str]]:
    specs = []
    for spec in pb8.pb6.pb4.pb3.DATASET_SPECS:
        if (
            spec["doc_type_name"] == "결정례_QA"
            and spec["source_subset"] == source_subset
            and spec.get("sampling_lane") == sampling_lane
        ):
            copied = dict(spec)
            copied["sample_count"] = 1
            specs.append(copied)
    return specs


def select_fresh_record_for_target(
    target: dict[str, str],
    replacement_index: int,
    exclusion_sets: dict[str, set[str]],
) -> dict[str, str]:
    for spec in matching_specs(target["source_subset"], target["sampling_lane"]):
        label_paths = pb8.pb6.pb4.pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb8.pb6.pb4.pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb8.pb6.pb4.pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        candidate_indices = selected_indices + [index for index in range(len(label_paths)) if index not in set(selected_indices)]

        for candidate_index in candidate_indices:
            label_path = label_paths[candidate_index]
            payload = pb8.pb6.pb4.pb3.explanation_common.normalize_label_payload(
                label_path,
                pb8.pb6.pb4.pb3.explanation_common.load_json(label_path),
                spec["doc_type_name"],
            )
            passes_filter, _ = pb9_smoke.passes_pb9_seed_filter(spec, payload)
            if not passes_filter:
                continue
            try:
                raw_path = pb8.pb6.pb4.pb3.explanation_common.locate_raw_path(
                    raw_paths,
                    spec["doc_type_name"],
                    payload["info"],
                )
            except FileNotFoundError:
                continue

            family_id = pb8.pb6.pb4.pb3.explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
            if family_id in exclusion_sets["family_ids"]:
                continue
            if str(label_path) in exclusion_sets["label_paths"]:
                continue
            if str(raw_path) in exclusion_sets["raw_paths"]:
                continue

            info = payload["info"]
            label = payload["label"]
            return {
                "sample_id": f"{REPLACEMENT_SEED_ID_PREFIX}_{replacement_index:03d}",
                "sample_order": replacement_index,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", ""),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb8.pb6.pb4.pb3.explanation_common.build_title(
                    {"info": info, "doc_type_name": spec["doc_type_name"]}
                ),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": replacement_index,
                "selected_index": candidate_index,
                "replacement_for_seed_sample_id": target["failed_seed_sample_id"],
                "replacement_tail_class": target["tail_class"],
                "target_correct_choice": target["target_correct_choice"],
                "selection_note": f"pb9 salvage replacement for {target['failed_seed_sample_id']}",
            }
    raise RuntimeError(f"fresh replacement seed not found for {target}")


def build_seed_row(record: dict[str, str]) -> dict[str, str]:
    row = pb8.pb6.pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_pb9_accepted34_6slot_replacement_seed"
    row["selection_note"] = record["selection_note"]
    row["pb9_replacement_for_seed_sample_id"] = record["replacement_for_seed_sample_id"]
    row["pb9_replacement_tail_class"] = record["replacement_tail_class"]
    row["target_correct_choice"] = record["target_correct_choice"]
    row["pb9_salvage_scope_note"] = "accepted34_preserved_plus_6_fresh_replacement_not_counted_until_reviewer_signoff"
    return row


def build_preflight_rows(seed_rows: list[dict[str, str]], exclusion_sets: dict[str, set[str]]) -> list[dict[str, str]]:
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    rows = []
    for row in seed_rows:
        rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "replacement_for_seed_sample_id": row["pb9_replacement_for_seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "target_correct_choice": row["target_correct_choice"],
                "tail_class": row["pb9_replacement_tail_class"],
                "family_id": row["family_id"],
                "family_duplicate_in_batch": "예" if family_counts[row["family_id"]] > 1 else "아니오",
                "label_path_duplicate_in_batch": "예" if label_counts[row["label_path"]] > 1 else "아니오",
                "raw_path_duplicate_in_batch": "예" if raw_counts[row["raw_path"]] > 1 else "아니오",
                "family_overlap_with_prior": "예" if row["family_id"] in exclusion_sets["family_ids"] else "아니오",
                "label_path_overlap_with_prior": "예" if row["label_path"] in exclusion_sets["label_paths"] else "아니오",
                "raw_path_overlap_with_prior": "예" if row["raw_path"] in exclusion_sets["raw_paths"] else "아니오",
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    return rows


def assert_preflight(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"replacement seed count must be {EXPECTED_TOTAL_SEED_COUNT}: {len(seed_rows)}")
    if Counter(row["target_correct_choice"] for row in seed_rows) != Counter(REPLACEMENT_TARGET_LABEL_COUNTS):
        raise RuntimeError("replacement target label counts mismatch")
    for row in preflight_rows:
        overlap_flags = [
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        if "예" in overlap_flags:
            raise RuntimeError(f"pb9 salvage seed preflight failed: {row['seed_sample_id']}")


def write_preflight_report(seed_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]]) -> None:
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    pb8.pb6.pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- replacement_seed_count: `{len(seed_rows)}`",
        f"- source_subset_counts: `{dict(source_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- target_label_counts: `{dict(target_counts)}`",
        f"- accepted_source_rows: `{EXPECTED_PRESERVED_COUNT}`",
        f"- final_package_target: `{EXPECTED_FINAL_PACKAGE_COUNT}`",
        "",
        "## replacement targets",
        "| failed_seed | replacement_seed | source | lane | target | tail_class |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in seed_rows:
        lines.append(
            f"| `{row['pb9_replacement_for_seed_sample_id']}` | `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['target_correct_choice']}` | `{row['pb9_replacement_tail_class']}` |"
        )
    lines.extend(
        [
            "",
            "## checks",
            "| check | result |",
            "| --- | --- |",
            f"| accepted {EXPECTED_PRESERVED_COUNT} source rows preserved | `pass` |",
            f"| replacement seed count is {EXPECTED_TOTAL_SEED_COUNT} | `pass` |",
            f"| replacement target labels are {dict(REPLACEMENT_TARGET_LABEL_COUNTS)} | `pass` |",
            "| no prior/family/label/raw overlap | `pass` |",
            "| no batch family/label/raw duplicate | `pass` |",
            "| current count remains not_counted before reviewer sign-off | `pass` |",
        ]
    )
    pb8.pb6.pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_target_label_schedule(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "replacement_for_seed_sample_id": row["pb9_replacement_for_seed_sample_id"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "target_correct_choice": row["target_correct_choice"],
        }
        for row in seed_rows
    ]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    return rows


def write_replacement_plan(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    global REPLACEMENT_PLAN_ROWS
    REPLACEMENT_PLAN_ROWS = [
        {
            "failed_seed_sample_id": row["pb9_replacement_for_seed_sample_id"],
            "replacement_seed_sample_id": row["seed_sample_id"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "target_correct_choice": row["target_correct_choice"],
            "tail_class": row["pb9_replacement_tail_class"],
            "policy": "accepted34_preserved_then_fresh_replacement",
        }
        for row in seed_rows
    ]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(REPLACEMENT_PLAN_CSV_PATH, REPLACEMENT_PLAN_ROWS, list(REPLACEMENT_PLAN_ROWS[0].keys()))
    lines = [
        f"# replacement plan `{VERSION_TAG}`",
        "",
        "| failed_seed | replacement_seed | source | lane | target | tail_class |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in REPLACEMENT_PLAN_ROWS:
        lines.append(
            f"| `{row['failed_seed_sample_id']}` | `{row['replacement_seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['target_correct_choice']}` | `{row['tail_class']}` |"
        )
    pb8.pb6.pb4.pb3.base.write_text_atomic(REPLACEMENT_PLAN_MD_PATH, "\n".join(lines) + "\n")
    return REPLACEMENT_PLAN_ROWS


def build_seed_registry() -> list[dict[str, str]]:
    pb8.pb6.pb4.pb3.base.ensure_dirs(
        INTERIM_DIR,
        PROCESSED_DIR,
        RUN_DIR,
        RUN_PROMPTS_DIR,
        RUN_INPUTS_DIR,
        RUN_GENERATIONS_DIR,
        RUN_JUDGE_LOGS_DIR,
        RUN_MERGED_DIR,
        RUN_EXPORTS_DIR,
    )
    if len(accepted_pb9_rows()) != EXPECTED_PRESERVED_COUNT:
        raise RuntimeError(f"accepted {EXPECTED_PRESERVED_COUNT} package source check failed")

    exclusion_sets = build_exclusion_sets(collect_excluded_rows())
    records = []
    for index, target in enumerate(REPLACEMENT_TARGETS, start=1):
        record = select_fresh_record_for_target(target, index, exclusion_sets)
        records.append(record)
        add_record_to_exclusion_sets(record, exclusion_sets)

    seed_rows = [build_seed_row(record) for record in records]
    preflight_rows = build_preflight_rows(seed_rows, build_exclusion_sets(collect_excluded_rows()))
    assert_preflight(seed_rows, preflight_rows)
    pb8.pb6.pb4.pb3.base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb8.pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    write_target_label_schedule(seed_rows)
    write_replacement_plan(seed_rows)
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = pb9_api.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## pb9 accepted34 + 6-slot replacement 추가 지시
- 이번 seed는 `pb9` failed/not-counted package의 `{seed.get('pb9_replacement_for_seed_sample_id', '')}` slot을 fresh seed로 대체하기 위한 replacement다.
- 이 replacement의 tail class는 `{seed.get('pb9_replacement_tail_class', '')}`이고, target label은 `{seed.get('target_correct_choice', '')}`이다.
- 정답은 반드시 `gold_short_answer`의 판단 기준과 닫혀야 하며, 오답이 `gold_reference_explanation`의 다른 독립 사유를 그대로 충족하면 안 된다.
- 오답 3개는 정답과 같은 결정 이유, 판단 기준, 적용 사실 중 하나를 공유하되 각각 한 축만 어긋나야 한다.
- stem은 하나의 판단 기준이나 적용 사실만 묻고, ending은 한 번만 닫는다.
- 단순 정의·기관 역할·상식적 반대말 오답은 weak distractor로 본다.
"""
    return messages


def target_schedule_by_seed() -> dict[str, str]:
    return {row["seed_sample_id"]: row["target_correct_choice"] for row in read_csv_rows(TARGET_LABEL_SCHEDULE_CSV_PATH)}


def replacement_seed_metadata_by_seed() -> dict[str, dict[str, str]]:
    # merge 단계가 seed registry의 보조 컬럼을 모두 보존하지 않으므로,
    # validator/report/tail artifact에서 어떤 failed slot을 대체했는지 다시 주입한다.
    return {row["seed_sample_id"]: row for row in read_csv_rows(SEED_REGISTRY_PATH)}


def replacement_export_disposition(action: str, metadata_ok) -> tuple[str, str, str]:
    # `regenerate`/`hard_block`은 애초에 export remap 대상이 아니므로 metadata gate보다
    # validator action 자체를 먼저 반영해야 tail 원인이 metadata failure로 오분류되지 않는다.
    if action == "regenerate":
        return "regenerate_required", "아니오", "아니오"
    if action == "hard_block":
        return "hard_blocked", "아니오", "아니오"
    return pb9_smoke.pb9_export_disposition(action, bool(metadata_ok))


def apply_validator_to_replacements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    target_by_seed = target_schedule_by_seed()
    metadata_by_seed = replacement_seed_metadata_by_seed()
    validated_rows = []
    report_rows = []
    for row in selected_rows(rows):
        seed_metadata = metadata_by_seed.get(row["seed_sample_id"], {})
        row["pb9_replacement_for_seed_sample_id"] = seed_metadata.get("pb9_replacement_for_seed_sample_id", "")
        row["pb9_replacement_tail_class"] = seed_metadata.get("pb9_replacement_tail_class", "")
        target_label = target_by_seed[row["seed_sample_id"]]
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

        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
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
        else:
            metadata_ok = None
            metadata_reasons = []

        disposition, split_allowed, count_allowed = replacement_export_disposition(action, metadata_ok)
        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["metadata_remap_ok"] = "예" if metadata_ok is True else "아니오" if metadata_ok is False else "대상아님"
        row["metadata_remap_reasons"] = "|".join(metadata_reasons)
        row["validator_export_disposition"] = disposition
        row["split_allowed"] = split_allowed
        row["count_allowed"] = "아니오"
        row["export_correct_choice"] = row.get("correct_choice", "")
        row["batch_status"] = BATCH_STATUS
        row["count_reflection_status"] = COUNT_REFLECTION_STATUS
        row["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
        row["replacement_package_role"] = "replacement_candidate"

        if disposition == "export_ready":
            row["final_status"] = "pass"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "예"
        elif disposition == "audit_queue":
            row["final_status"] = "pass"
            row["audit_required"] = "예"
            row["audit_reason"] = "validator_audit"
            row["train_eligible"] = "아니오"
        elif disposition == "regenerate_required":
            row["final_status"] = "soft_fail"
            row["audit_required"] = "아니오"
            row["train_eligible"] = "아니오"
        else:
            row["final_status"] = "hard_fail"
            row["audit_required"] = "아니오"
            row["train_eligible"] = "아니오"

        report_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "replacement_for_seed_sample_id": row.get("pb9_replacement_for_seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "tail_class": row.get("pb9_replacement_tail_class", ""),
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "validator_action": action,
                "validator_status": status,
                "validator_reasons": row["validator_reasons"],
                "validator_export_disposition": disposition,
                "metadata_remap_ok": row["metadata_remap_ok"],
                "final_status": row["final_status"],
                "train_eligible": row["train_eligible"],
                "audit_required": row["audit_required"],
                "export_correct_choice": row["export_correct_choice"],
            }
        )
        validated_rows.append(row)

    if rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    pb8.pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, list(report_rows[0].keys()))
    write_validator_report_md(report_rows)
    return validated_rows


def normalize_preserved_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    normalized["batch_status"] = BATCH_STATUS
    normalized["count_reflection_status"] = COUNT_REFLECTION_STATUS
    normalized["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
    normalized["count_allowed"] = "아니오"
    normalized["split_allowed"] = "예"
    normalized["replacement_package_role"] = "preserved_pb9_accept"
    normalized["export_correct_choice"] = normalized.get("export_correct_choice", normalized.get("correct_choice", ""))
    return normalized


def final_package_rows(replacement_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    package = [normalize_preserved_row(row) for row in accepted_pb9_rows()] + replacement_rows
    package.sort(key=lambda row: row["seed_sample_id"])
    if len(package) != EXPECTED_FINAL_PACKAGE_COUNT:
        raise RuntimeError(f"final package must be {EXPECTED_FINAL_PACKAGE_COUNT} rows: {len(package)}")
    return package


def compute_package_summary(package: list[dict[str, str]]) -> dict[str, object]:
    selected = selected_rows(package)
    status_counter = Counter(row.get("final_status", "") for row in selected)
    train_counter = Counter(row.get("train_eligible", "") for row in selected)
    audit_count = sum(1 for row in selected if row.get("audit_required") == "예")
    selected_label_counts = Counter(row.get("export_correct_choice", row.get("correct_choice", "")) for row in selected)
    export_label_counts = Counter(row.get("correct_choice", "") for row in selected if row.get("validator_export_disposition") == "export_ready")
    action_counts = Counter(row.get("validator_action", "") for row in selected)
    metadata_mismatch = sum(
        1
        for row in selected
        if row.get("validator_export_disposition") == "export_ready" and row.get("metadata_remap_ok") != "예"
    )
    shuffle_mismatch = sum(
        1
        for row in selected
        if row.get("validator_export_disposition") == "export_ready"
        and row.get("validator_recalculated_correct_choice", row.get("correct_choice", ""))
        != row.get("export_correct_choice", row.get("correct_choice", ""))
    )
    success = (
        len(selected) == EXPECTED_FINAL_PACKAGE_COUNT
        and train_counter.get("예", 0) >= SUCCESS_USABLE_MIN
        and status_counter.get("hard_fail", 0) <= SUCCESS_HARD_FAIL_MAX
        and status_counter.get("soft_fail", 0) <= SUCCESS_SOFT_FAIL_MAX
        and audit_count <= SUCCESS_AUDIT_MAX
        and metadata_mismatch == 0
        and shuffle_mismatch == 0
        and all(export_label_counts.get(label, 0) == TARGET_LABEL_COUNTS[label] for label in validator_replay.CHOICE_LABELS)
    )
    return {
        "selected_count": len(selected),
        "selected_pass_count": status_counter.get("pass", 0),
        "selected_hard_fail_count": status_counter.get("hard_fail", 0),
        "selected_soft_fail_count": status_counter.get("soft_fail", 0),
        "selected_train_eligible_count": train_counter.get("예", 0),
        "selected_audit_required_count": audit_count,
        "validator_action_counts": dict(action_counts),
        "selected_label_counts": {label: selected_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_label_counts": {label: export_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "metadata_remap_mismatch_count": metadata_mismatch,
        "shuffle_recalc_mismatch_count": shuffle_mismatch,
        "final_package_success_passed": success,
    }


def write_validator_report_md(report_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## replacement actions",
        "| seed | failed_seed | source | lane | target | action | disposition | final | train | audit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['replacement_for_seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['target_correct_choice']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['audit_required']}` |"
        )
    pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_final_package_artifacts(package: list[dict[str, str]]) -> dict[str, object]:
    summary = compute_package_summary(package)
    fieldnames = []
    for row in package:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    pb8.pb6.pb4.pb3.base.write_csv_atomic(FINAL_PACKAGE_CSV_PATH, package, fieldnames)
    lines = [
        f"# final package `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{summary['selected_count']}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        f"- validator_action_counts: `{summary['validator_action_counts']}`",
        f"- selected_label_counts: `{summary['selected_label_counts']}`",
        f"- export_label_counts: `{summary['export_label_counts']}`",
        f"- final_package_success_passed: `{summary['final_package_success_passed']}`",
        "",
        "## package roles",
        "| role | count |",
        "| --- | ---: |",
    ]
    for role, count in sorted(Counter(row.get("replacement_package_role", "") for row in package).items()):
        lines.append(f"| `{role}` | `{count}` |")
    pb8.pb6.pb4.pb3.base.write_text_atomic(FINAL_PACKAGE_MD_PATH, "\n".join(lines) + "\n")
    return summary


def write_tail_memo(package: list[dict[str, str]]) -> list[dict[str, str]]:
    tail_rows = []
    for row in selected_rows(package):
        is_tail = row.get("train_eligible") != "예" or row.get("audit_required") == "예" or row.get("final_status") != "pass"
        if not is_tail:
            continue
        tail_rows.append(
            {
                "seed_sample_id": row.get("seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "validator_action": row.get("validator_action", ""),
                "validator_export_disposition": row.get("validator_export_disposition", ""),
                "target_correct_choice": row.get("target_correct_choice", ""),
                "export_correct_choice": row.get("export_correct_choice", ""),
                "tail_class": row.get("pb9_replacement_tail_class", ""),
            }
        )
    fieldnames = [
        "seed_sample_id",
        "source_subset",
        "sampling_lane",
        "final_status",
        "audit_required",
        "error_tags",
        "validator_action",
        "validator_export_disposition",
        "target_correct_choice",
        "export_correct_choice",
        "tail_class",
    ]
    # tail이 없을 때는 placeholder row를 만들지 않는다. 자동 row count가 실제 tail 수와 같아야 한다.
    pb8.pb6.pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, fieldnames)
    lines = [f"# tail memo `{VERSION_TAG}`", ""]
    if tail_rows:
        lines.extend(
            [
                "| seed | source | lane | status | audit | action | disposition | target/export | tail_class |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in tail_rows:
            lines.append(
                f"| `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['target_correct_choice']}/{row['export_correct_choice']}` | `{row['tail_class']}` |"
            )
    else:
        lines.append("- tail_count: `0`")
        lines.append("- csv_policy: `header_only_no_placeholder_row`")
    pb8.pb6.pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def split_dataset_with_salvage_package(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    validated_replacements = apply_validator_to_replacements(rows)
    package = final_package_rows(validated_replacements)
    fieldnames = []
    for row in package:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    pb8.pb6.pb4.pb3.base.write_csv_atomic(MERGED_SCORES_PATH, package, fieldnames)
    rows[:] = package
    manifest_rows = pb9_api.BASE_SPLIT_DATASET(package)
    enriched_manifest_rows = []
    selected_by_problem_id = {row["candidate_id"]: row for row in package}
    for manifest_row in manifest_rows:
        source = selected_by_problem_id[manifest_row["problem_id"]]
        enriched = dict(manifest_row)
        enriched.update(
            {
                "batch_status": BATCH_STATUS,
                "count_reflection_status": COUNT_REFLECTION_STATUS,
                "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
                "export_correct_choice": source.get("export_correct_choice", source.get("correct_choice", "")),
                "target_correct_choice": source.get("target_correct_choice", ""),
                "validator_action": source.get("validator_action", ""),
                "validator_export_disposition": source.get("validator_export_disposition", ""),
                "validator_recalculated_correct_choice": source.get("validator_recalculated_correct_choice", ""),
                "metadata_remap_ok": source.get("metadata_remap_ok", ""),
            }
        )
        enriched_manifest_rows.append(enriched)
    if enriched_manifest_rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(
            PROBLEM_DATASET_MANIFEST_PATH,
            enriched_manifest_rows,
            list(enriched_manifest_rows[0].keys()),
        )
    rewrite_split_jsonl_with_status(package)
    write_final_package_artifacts(package)
    write_tail_memo(package)
    write_manifest_header_gate(enriched_manifest_rows)
    return enriched_manifest_rows


def rewrite_split_jsonl_with_status(package: list[dict[str, str]]) -> None:
    selected_by_problem_id = {row["candidate_id"]: row for row in package}
    for path in (PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH, PROBLEM_TEST_PATH):
        if not path.exists():
            continue
        payload_rows = pb8.pb6.pb4.pb3.base.load_jsonl(path)
        enriched_rows = []
        for payload in payload_rows:
            source = selected_by_problem_id.get(payload.get("problem_id", ""), {})
            enriched = dict(payload)
            enriched.update(
                {
                    "batch_status": BATCH_STATUS,
                    "count_reflection_status": COUNT_REFLECTION_STATUS,
                    "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
                    "validator_action": source.get("validator_action", ""),
                    "validator_export_disposition": source.get("validator_export_disposition", ""),
                    "target_correct_choice": source.get("target_correct_choice", ""),
                    "export_correct_choice": source.get("export_correct_choice", ""),
                }
            )
            enriched_rows.append(enriched)
        pb8.pb6.pb4.pb3.base.write_jsonl_atomic(path, enriched_rows)


def write_manifest_header_gate(manifest_rows: list[dict[str, str]]) -> None:
    headers = list(manifest_rows[0].keys()) if manifest_rows else []
    missing = [field for field in pb9_smoke.MANIFEST_REQUIRED_FIELDS if field not in headers]
    if missing:
        raise RuntimeError(f"salvage manifest required fields missing: {missing}")
    lines = [
        f"# manifest header gate `{VERSION_TAG}`",
        "",
        "| check | result | value |",
        "| --- | --- | --- |",
        f"| required header fields | `pass` | `{pb9_smoke.MANIFEST_REQUIRED_FIELDS}` |",
        "| count reflection | `pass` | `not_counted until reviewer sign-off` |",
    ]
    pb8.pb6.pb4.pb3.base.write_text_atomic(MANIFEST_HEADER_GATE_MD_PATH, "\n".join(lines) + "\n")


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # `compute_package_summary()`가 내부에서 selected row만 다시 고르므로,
    # 여기서는 final package 전체를 넘겨 package-level 기준을 한 곳에서만 계산한다.
    summary = compute_package_summary(rows)
    # salvage helper를 재사용하는 하위 package도 reviewer-facing target을 자기 label target으로 쓰게 한다.
    label_target_text = "/".join(str(TARGET_LABEL_COUNTS.get(label, 0)) for label in validator_replay.CHOICE_LABELS)
    summary_rows = [
        {"metric": "final_package_count", "value": str(summary["selected_count"])},
        {"metric": "selected_pass", "value": str(summary["selected_pass_count"])},
        {"metric": "selected_hard_fail", "value": str(summary["selected_hard_fail_count"])},
        {"metric": "selected_soft_fail", "value": str(summary["selected_soft_fail_count"])},
        {"metric": "train_eligible", "value": str(summary["selected_train_eligible_count"])},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "final_package_success_passed", "value": str(summary["final_package_success_passed"])},
    ]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    lane_rows = [
        {"sampling_lane": lane, "count": str(count)}
        for lane, count in sorted(Counter(row.get("sampling_lane", "") for row in rows).items())
    ]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, ["sampling_lane", "count"])
    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| final package | `{EXPECTED_FINAL_PACKAGE_COUNT}` | `{summary['selected_count']}` |",
        f"| usable | `>= {SUCCESS_USABLE_MIN}` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `{SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| export label balance | `A/B/C/D = {label_target_text}` | `{summary['export_label_counts']}` |",
        f"| selected label balance | `diagnostic only` | `{summary['selected_label_counts']}` |",
        f"| shuffle/metadata mismatch | `0` | `shuffle {summary['shuffle_recalc_mismatch_count']} / metadata {summary['metadata_remap_mismatch_count']}` |",
        f"| final_package_success_passed | `True` | `{summary['final_package_success_passed']}` |",
        "- count_reflection: `not_counted_until_reviewer_signoff`",
    ]
    pb8.pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict[str, object]:
    # manifest도 같은 package-level summary를 공유해야 batch summary와 숫자가 어긋나지 않는다.
    summary = compute_package_summary(merged_rows)
    tail_rows = read_csv_rows(TAIL_MEMO_CSV_PATH)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": pb8.pb6.pb4.pb3.base.utc_now_iso(),
        "source_pb9_run_name": SOURCE_PB9_RUN_NAME,
        # final salvage 계열은 원본 pb9 또는 직전 salvage package를 source로 삼을 수 있다.
        # count reflection review가 package 구성을 재구성할 수 있도록 중립 source field와 row 수를 함께 남긴다.
        "source_package_run_name": SOURCE_PB9_RUN_NAME,
        "source_salvage_run_name": SOURCE_PB9_RUN_NAME
        if "salvage_package" in SOURCE_PB9_RUN_NAME or "replacement_package" in SOURCE_PB9_RUN_NAME
        else "",
        "preserved_row_count": EXPECTED_PRESERVED_COUNT,
        "replacement_row_count": len(seed_rows),
        "replacement_seed_count": len(seed_rows),
        "generation_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(NEARMISS_LOG_PATH),
        "final_package_count": summary["selected_count"],
        "selected_pass_count": summary["selected_pass_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": pb8.pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": pb8.pb6.pb4.pb3.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "validator_summary": summary,
        "success_criteria": {
            "final_package_count": EXPECTED_FINAL_PACKAGE_COUNT,
            "usable_min": SUCCESS_USABLE_MIN,
            "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
            "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
            "audit_max": SUCCESS_AUDIT_MAX,
            "final_exact_export_label_balance": TARGET_LABEL_COUNTS,
            "selected_label_counts": "diagnostic_only",
            "shuffle_recalc_mismatch": 0,
            "metadata_remap_mismatch": 0,
            "count_reflection": "reviewer_signoff_required",
        },
        "success_result": {
            "usable": summary["selected_train_eligible_count"],
            "hard_fail": summary["selected_hard_fail_count"],
            "soft_fail": summary["selected_soft_fail_count"],
            "audit": summary["selected_audit_required_count"],
            "export_label_counts": summary["export_label_counts"],
            "selected_label_counts": summary["selected_label_counts"],
            "shuffle_recalc_mismatch_count": summary["shuffle_recalc_mismatch_count"],
            "metadata_remap_mismatch_count": summary["metadata_remap_mismatch_count"],
            "passed": summary["final_package_success_passed"],
        },
        "current_count_decision": "not_counted_until_reviewer_signoff_after_salvage_package_review",
        "tail_memo_count": len(tail_rows),
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight_csv": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
            "replacement_plan_csv": str(REPLACEMENT_PLAN_CSV_PATH),
            "replacement_plan_md": str(REPLACEMENT_PLAN_MD_PATH),
            "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "replacement_merged_before_validator": str(REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "final_package_csv": str(FINAL_PACKAGE_CSV_PATH),
            "final_package_md": str(FINAL_PACKAGE_MD_PATH),
            "tail_memo_csv": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md": str(TAIL_MEMO_MD_PATH),
            "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
        },
    }
    pb8.pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_salvage_globals() -> None:
    pb8.pb6.VERSION_TAG = VERSION_TAG
    pb8.pb6.RUN_DATE = RUN_DATE
    pb8.pb6.RUN_PURPOSE = RUN_PURPOSE
    pb8.pb6.RUN_NAME = RUN_NAME
    pb8.pb6.RUN_LABEL = "pb9 accepted34 + 6-slot replacement salvage package"
    pb8.pb6.SEED_ID_PREFIX = REPLACEMENT_SEED_ID_PREFIX
    pb8.pb6.SEED_SELECTION_ROLE = "objective_pb9_accepted34_6slot_replacement_seed"
    pb8.pb6.SEED_SELECTION_NOTE = "pb9 failed-not-counted tail slot을 fresh seed로 대체하는 salvage replacement seed"
    pb8.pb6.SEED_FILTER_NOTE = "pb9_salvage_seen_seed_pool_excluded"
    pb8.pb6.SCOPE_NOTE = (
        f"결정례_QA only; accepted {EXPECTED_PRESERVED_COUNT} preserved plus "
        f"{EXPECTED_TOTAL_SEED_COUNT} fresh replacement; current count 미합산"
    )
    pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb8.pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb8.pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb8.pb6.PB6_SOURCE_COUNTS = PB9_REPLACEMENT_SOURCE_COUNTS
    pb8.pb6.PB6_DATASET_SPECS = pb8.pb6.build_pb6_dataset_specs()
    pb8.pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pb8.pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pb8.pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pb8.pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pb8.pb6.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_pb9_accepted34_6slot_replacement_package"
    pb8.pb6.SEED_REGISTRY_STRATEGY = "pb9_tail_6slot_fresh_replacement_excluding_seen_seed_pool"
    pb8.pb6.LAW_STATUS_NOTE = "pb9_salvage_candidate_not_counted"
    pb8.pb6.OVERLAP_CHECK_LABEL = "no current/failed/repaired/pb9/calibration-seen seed overlap"
    pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`pb9` accepted `34개`는 보존하고 failed/audit tail `6개`만 fresh replacement로 대체한다.",
        "`pb9` original seed registry, `04TL` calibration seed, `04TL` C-slot replacement seed는 모두 seen seed로 제외한다.",
        "final `40-slot` package가 통과해도 reviewer sign-off 전 current count에는 합산하지 않는다.",
    ]
    pb8.pb6.INTERIM_DIR = INTERIM_DIR
    pb8.pb6.PROCESSED_DIR = PROCESSED_DIR
    pb8.pb6.RUN_DIR = RUN_DIR
    pb8.pb6.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb8.pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb8.pb6.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb8.pb6.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb8.pb6.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb8.pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb8.pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb8.pb6.SEED_READY_PATH = SEED_READY_PATH
    pb8.pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb8.pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb8.pb6.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
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
    pb8.pb6.build_seed_registry = build_seed_registry
    pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb8.pb6.build_batch_summary = build_batch_summary
    pb8.pb6.build_run_manifest = build_run_manifest
    pb8.pb6.pb4.pb3.base.split_dataset = split_dataset_with_salvage_package


def main() -> dict[str, object]:
    configure_salvage_globals()
    return pb8.pb6.main()


if __name__ == "__main__":
    main()
