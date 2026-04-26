from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# 서술형은 기존 `v3` split-descriptive recipe를 유지하되, 마감 운영을 위해
# fresh seed를 먼저 쓰고 부족분만 train-only split-lock 재사용으로 보충한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_descriptive_pb1 as base  # noqa: E402
from scripts.aihub.problem_generation.shared.production_batch_common import write_text_atomic  # noqa: E402


EXPLANATION_DIR = base.PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "explanation_generation"
if str(EXPLANATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLANATION_DIR))

import common as explanation_common  # noqa: E402
from extract_evidence_cards import build_card  # noqa: E402
from generate_explanations import build_local_fallback_explanation, postprocess_generated_explanation  # noqa: E402
from settings import DATASET_SPECS  # noqa: E402
from transform_problems import build_transformed_sample  # noqa: E402


VERSION_TAG = "descriptive_wave_v2_split_lock"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "descriptive_v3_split_lock_target40_candidate56_api_execution"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
ROUTE_NAME = "descriptive_wave_v2_split_lock_api_execution"

PROJECT_ROOT = base.PROJECT_ROOT
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
EXCLUSION_AUDIT_CSV_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.csv"
PACKAGE_SPEC_MD_PATH = RUN_EXPORTS_DIR / f"package_spec_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWERABILITY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answerability_{VERSION_TAG}.jsonl"
TASKFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_taskfit_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
CANDIDATE_POOL_PATH = RUN_DIR / "candidate_pool.csv"
ACCEPTED_POOL_PATH = RUN_DIR / "accepted_pool.csv"
REJECTED_POOL_PATH = RUN_DIR / "rejected_pool.csv"
TAIL_TAXONOMY_PATH = RUN_DIR / "tail_taxonomy.csv"
QUOTA_SURPLUS_POOL_PATH = RUN_DIR / "quota_surplus_pool.csv"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
EVIDENCE_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"evidence_summary_{VERSION_TAG}.md"
COMPILER_MANIFEST_PATH = RUN_DIR / "compiler_manifest.json"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

PACKAGE_ROLE = "count_reflection_candidate_package"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
COUNT_DISPOSITION = "candidate_not_counted"
PROMOTION_CONTRACT_STATUS = "reviewer_signoff_needed"
YES = "예"
NO = "아니오"

EXPLANATION_GENERATION_VARIANT = {
    "name": "without_long_answer",
    "label": "long_answer 제외",
    "include_long_answer": False,
}

PRIMARY_SOURCE_COUNTS = {
    "01_TL_법령_QA": 3,
    "02_TL_법령_QA": 3,
    "03_TL_법령_QA": 4,
    "04_TL_법령_QA": 4,
    "01_TL_유권해석_QA": 3,
    "02_TL_유권해석_QA": 3,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 4,
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 2,
    "02_TL_심결문_QA": 2,
    "03_TL_결정례_QA": 4,
    "04_TL_결정례_QA": 3,
    "01_TL_판결문_QA": 3,
    "02_TL_판결문_QA": 3,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 4,
}
PRIMARY_FINAL_SOURCE_COUNTS = {
    "01_TL_법령_QA": 2,
    "02_TL_법령_QA": 2,
    "03_TL_법령_QA": 3,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 2,
    "02_TL_유권해석_QA": 2,
    "03_TL_해석례_QA": 3,
    "04_TL_해석례_QA": 3,
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 3,
    "01_TL_판결문_QA": 2,
    "02_TL_판결문_QA": 2,
    "03_TL_판결문_QA": 3,
    "04_TL_판결문_QA": 3,
}
FALLBACK_SOURCE_COUNTS = {
    "01_TL_법령_QA": 2,
    "02_TL_법령_QA": 2,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 2,
    "02_TL_유권해석_QA": 2,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 3,
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 2,
    "01_TL_판결문_QA": 2,
    "02_TL_판결문_QA": 2,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 3,
}
FALLBACK_FINAL_SOURCE_COUNTS = {
    "01_TL_법령_QA": 1,
    "02_TL_법령_QA": 1,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 2,
    "01_TL_유권해석_QA": 1,
    "02_TL_유권해석_QA": 1,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 2,
    "01_TL_심결례_QA": 1,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 2,
    "04_TL_결정례_QA": 1,
    "01_TL_판결문_QA": 1,
    "02_TL_판결문_QA": 1,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 2,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_base_paths() -> None:
    # 기존 `run_descriptive_pb1`의 generator/Judge 구현을 재사용하되, 모든 산출물은 wave 전용 path에만 쓴다.
    base.VERSION_TAG = VERSION_TAG
    base.RUN_DATE = RUN_DATE
    base.RUN_PURPOSE = RUN_PURPOSE
    base.RUN_NAME = RUN_NAME
    base.INTERIM_DIR = INTERIM_DIR
    base.PROCESSED_DIR = PROCESSED_DIR
    base.RUN_DIR = RUN_DIR
    base.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    base.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    base.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    base.RUN_MERGED_DIR = RUN_MERGED_DIR
    base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    base.SEED_READY_PATH = SEED_READY_PATH
    base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    base.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    base.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    base.ANSWERABILITY_LOG_PATH = ANSWERABILITY_LOG_PATH
    base.TASKFIT_LOG_PATH = TASKFIT_LOG_PATH
    base.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Answerability": ANSWERABILITY_LOG_PATH,
        "TaskFit": TASKFIT_LOG_PATH,
    }


def row_value(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value:
        return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get(field, "") or "")
    return ""


def collect_exclusion_rows() -> list[dict]:
    # seed reuse와 count leakage를 막기 위해 counted output뿐 아니라 prior candidate registry까지 함께 제외한다.
    rows: list[dict] = []
    for path in (PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation").glob("**/seed_registry.csv"):
        if path == SEED_REGISTRY_PATH:
            continue
        rows.extend(read_csv_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation").glob("**/*.csv"):
        rows.extend(read_csv_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation").glob("**/*.jsonl"):
        rows.extend(read_jsonl_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final").glob("*"):
        if path.suffix == ".csv":
            rows.extend(read_csv_rows(path))
        elif path.suffix == ".jsonl":
            rows.extend(read_jsonl_rows(path))
    return rows


def build_exclusion_sets(rows: list[dict]) -> dict[str, set[str]]:
    exclusion_sets = {
        field: {row_value(row, field) for row in rows if row_value(row, field)}
        for field in ["seed_sample_id", "reference_sample_id", "sample_id", "family_id", "label_path", "raw_path"]
    }
    # split-lock 정책은 source reuse와 eval leakage를 분리한다. counted objective train family만 train split으로 재사용 가능하다.
    for field in [
        "split_lock_train_family_id",
        "split_lock_train_label_path",
        "split_lock_train_raw_path",
        "global_eval_family_id",
        "global_eval_label_path",
        "global_eval_raw_path",
        "quality_tail_family_id",
        "quality_tail_label_path",
        "quality_tail_raw_path",
    ]:
        exclusion_sets[field] = set()
    for row in rows:
        task_type = row_value(row, "problem_task_type")
        split = row_value(row, "split") or row_value(row, "dataset_disposition")
        counted = row_value(row, "count_reflection_status") == "counted" or row_value(row, "count_allowed") == YES
        family_id = row_value(row, "family_id")
        label_path = row_value(row, "label_path")
        raw_path = row_value(row, "raw_path")
        values = {
            "family_id": family_id,
            "label_path": label_path,
            "raw_path": raw_path,
        }
        if task_type == "objective_single_best" and counted and split == "train":
            for field, value in values.items():
                if value:
                    exclusion_sets[f"split_lock_train_{field}"].add(value)
        if counted and split in {"dev", "test"}:
            for field, value in values.items():
                if value:
                    exclusion_sets[f"global_eval_{field}"].add(value)
        if row_value(row, "quality_failure") == YES or row_value(row, "audit_required") == YES or row_value(row, "final_status") in {"hard_fail", "soft_fail"}:
            for field, value in values.items():
                if value:
                    exclusion_sets[f"quality_tail_{field}"].add(value)
    return exclusion_sets


def classify_reuse_policy(family_id: str, label_path: str, raw_path: str, exclusion_sets: dict[str, set[str]]) -> tuple[dict[str, str] | None, str]:
    # P1 계열은 train-only 재사용보다 항상 우선한다.
    if (
        family_id in exclusion_sets["global_eval_family_id"]
        or label_path in exclusion_sets["global_eval_label_path"]
        or raw_path in exclusion_sets["global_eval_raw_path"]
    ):
        return None, "global_eval_holdout_overlap"
    if (
        family_id in exclusion_sets["quality_tail_family_id"]
        or label_path in exclusion_sets["quality_tail_label_path"]
        or raw_path in exclusion_sets["quality_tail_raw_path"]
    ):
        return None, "quality_tail_overlap"

    overlaps_prior = (
        family_id in exclusion_sets["family_id"]
        or label_path in exclusion_sets["label_path"]
        or raw_path in exclusion_sets["raw_path"]
    )
    if not overlaps_prior:
        return {
            "reuse_tier": "Tier 0 fresh-only",
            "source_task": "",
            "source_split": "",
            "locked_split": "",
            "reuse_reason": "fresh_seed_no_prior_overlap",
        }, ""

    train_allowed = (
        family_id in exclusion_sets["split_lock_train_family_id"]
        or label_path in exclusion_sets["split_lock_train_label_path"]
        or raw_path in exclusion_sets["split_lock_train_raw_path"]
    )
    if train_allowed:
        return {
            "reuse_tier": "Tier 2 cross-task split-locked reuse",
            "source_task": "objective_single_best",
            "source_split": "train",
            "locked_split": "train",
            "reuse_reason": "objective_train_family_reused_for_descriptive_train_only",
        }, ""
    return None, "prior_overlap_not_split_lock_allowed"


def schedule_candidate_indices(total: int, required_count: int) -> list[int]:
    if required_count <= 0:
        return []
    if total <= required_count:
        return list(range(total))
    return sorted({min(total - 1, round(index * (total - 1) / max(required_count - 1, 1))) for index in range(required_count)})


def choose_explanation_style(sample: dict) -> str:
    styles = sample.get("candidate_styles", [])
    if "legal_priority" in styles:
        return "legal_priority"
    if styles:
        return styles[0]
    return "single"


def build_seed_from_record(record: dict) -> dict:
    card = build_card(record)
    transformed = build_transformed_sample(card)
    style_name = choose_explanation_style(transformed)
    explanation = build_local_fallback_explanation(transformed, style_name, EXPLANATION_GENERATION_VARIANT)
    explanation = postprocess_generated_explanation(transformed, explanation)
    seed_row = base.build_seed_row(
        {
            "sample_id": record["sample_id"],
            "family_id": record["family_id"],
            "doc_type_name": record["doc_type_name"],
            "source_subset": record["source_subset"],
            "sampling_lane": record["sampling_lane"],
            "answer_mode": transformed.get("answer_mode", "") or "criteria",
            "explanation_target": transformed.get("explanation_target", ""),
            "transformed_problem": transformed["transformed_problem"],
            "short_answer": transformed["short_answer"],
            "generated_explanation": explanation,
            "rule_basis": transformed["evidence_card"].get("rule_basis", ""),
            "fact_basis": transformed["evidence_card"].get("fact_basis", ""),
            "label_path": record["label_path"],
            "raw_path": record["raw_path"],
        }
    )
    seed_row.update(
        {
            "selection_role": "descriptive_production_wave_candidate_seed",
            "selection_note": "fresh AI Hub QA seed를 v3 split-descriptive recipe로 후보 생성하는 production wave seed",
            "candidate_route_name": ROUTE_NAME,
            "candidate_source_schedule": ACTIVE_ROUTE["route_label"],
            "descriptive_explanation_style": style_name,
            "reuse_tier": record.get("reuse_tier", ""),
            "source_task": record.get("source_task", ""),
            "source_split": record.get("source_split", ""),
            "locked_split": record.get("locked_split", ""),
            "reuse_reason": record.get("reuse_reason", ""),
        }
    )
    return seed_row


def select_source_records(source_counts: dict[str, int], exclusion_sets: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    audit_rows: list[dict] = []
    used_family_ids: set[str] = set()
    used_label_paths: set[str] = set()
    used_raw_paths: set[str] = set()
    sample_order = 1

    spec_by_source = {spec["source_subset"]: spec for spec in DATASET_SPECS}
    for source_subset, required_count in source_counts.items():
        spec = spec_by_source[source_subset]
        label_paths = explanation_common.list_label_files(spec["label_glob"])
        raw_paths = explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = schedule_candidate_indices(len(label_paths), required_count)
        source_selected = 0

        for selected_index in selected_indices:
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            skip_reason = "candidate_not_found"
            for candidate_index in candidate_indices:
                label_path = label_paths[candidate_index]
                payload = explanation_common.normalize_label_payload(
                    label_path,
                    explanation_common.load_json(label_path),
                    spec["doc_type_name"],
                )
                try:
                    raw_path = explanation_common.locate_raw_path(raw_paths, spec["doc_type_name"], payload["info"])
                except FileNotFoundError:
                    skip_reason = "raw_path_missing"
                    continue
                family_id = explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
                if family_id in used_family_ids:
                    skip_reason = "family_overlap_in_batch"
                    continue
                if str(label_path) in used_label_paths:
                    skip_reason = "label_path_overlap_in_batch"
                    continue
                if str(raw_path) in used_raw_paths:
                    skip_reason = "raw_path_overlap_in_batch"
                    continue
                reuse_meta, skip_reason = classify_reuse_policy(family_id, str(label_path), str(raw_path), exclusion_sets)
                if reuse_meta is None:
                    continue
                chosen = (candidate_index, label_path, raw_path, family_id, payload, reuse_meta)
                break

            if chosen is None:
                audit_rows.append(
                    {
                        "source_subset": source_subset,
                        "doc_type_name": spec["doc_type_name"],
                        "required_count": str(required_count),
                        "selected_count": str(source_selected),
                        "skip_reason": skip_reason,
                    }
                )
                continue

            candidate_index, label_path, raw_path, family_id, payload, reuse_meta = chosen
            info = payload["info"]
            label = payload["label"]
            sample_id = f"desc_wave_{sample_order:03d}"
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_order": sample_order,
                    "source_subset": source_subset,
                    "domain": spec["domain"],
                    "doc_type_name": spec["doc_type_name"],
                    "sampling_lane": spec.get("sampling_lane", ""),
                    "source_schema": info.get("source_schema", ""),
                    "family_id": family_id,
                    "title": explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                    "info_json": json.dumps(info, ensure_ascii=False),
                    "label_path": str(label_path),
                    "raw_path": str(raw_path),
                    "label_input": label["input"],
                    "label_output": label["output"],
                    "selected_index": candidate_index,
                    "selection_note": "descriptive split-lock production wave candidate seed",
                    **reuse_meta,
                }
            )
            used_family_ids.add(family_id)
            used_label_paths.add(str(label_path))
            used_raw_paths.add(str(raw_path))
            source_selected += 1
            sample_order += 1

    return records, audit_rows


def choose_route(exclusion_sets: dict[str, set[str]]) -> tuple[dict, list[dict], list[dict]]:
    routes = [
        {
            "route_label": "primary_target40_candidate56",
            "target_count": 40,
            "source_counts": PRIMARY_SOURCE_COUNTS,
            "final_source_counts": PRIMARY_FINAL_SOURCE_COUNTS,
        },
        {
            "route_label": "fallback_target24_candidate36",
            "target_count": 24,
            "source_counts": FALLBACK_SOURCE_COUNTS,
            "final_source_counts": FALLBACK_FINAL_SOURCE_COUNTS,
        },
    ]
    for route in routes:
        records, audit_rows = select_source_records(route["source_counts"], exclusion_sets)
        if len(records) == sum(route["source_counts"].values()):
            return route, records, audit_rows
    # fallback route도 candidate quota를 채우지 못하면 API를 태우지 않는다.
    # 이 경우 reviewer가 지정한 runner-level P2로 보고 objective decision package factory fallback으로 전환한다.
    raise RuntimeError(
        "descriptive_seed_availability_blocker: "
        f"required={sum(routes[-1]['source_counts'].values())}, available={len(records)}, "
        f"route={routes[-1]['route_label']}"
    )


def write_preflight(seed_rows: list[dict], exclusion_sets: dict[str, set[str]], availability_audit: list[dict]) -> None:
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows: list[dict] = []
    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "family_duplicate_in_batch": YES if family_counts[row["family_id"]] > 1 else NO,
                "label_path_duplicate_in_batch": YES if label_counts[row["label_path"]] > 1 else NO,
                "raw_path_duplicate_in_batch": YES if raw_counts[row["raw_path"]] > 1 else NO,
                "family_overlap_with_prior": YES if row["family_id"] in exclusion_sets["family_id"] else NO,
                "label_path_overlap_with_prior": YES if row["label_path"] in exclusion_sets["label_path"] else NO,
                "raw_path_overlap_with_prior": YES if row["raw_path"] in exclusion_sets["raw_path"] else NO,
                "reuse_tier": row.get("reuse_tier", ""),
                "source_task": row.get("source_task", ""),
                "source_split": row.get("source_split", ""),
                "locked_split": row.get("locked_split", ""),
                "reuse_reason": row.get("reuse_reason", ""),
                "answer_mode": row["answer_mode"],
                "problem_generation_mode": row["problem_generation_mode"],
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    for row in preflight_rows:
        batch_duplicate = YES in [
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
        ]
        prior_overlap = YES in [
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        split_lock_ok = row["reuse_tier"] == "Tier 2 cross-task split-locked reuse" and row["locked_split"] == "train"
        if batch_duplicate or (prior_overlap and not split_lock_ok):
            raise RuntimeError(f"descriptive preflight overlap failed: {row['seed_sample_id']}")

    source_counts = Counter(row["source_subset"] for row in seed_rows)
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    reuse_counts = Counter(row.get("reuse_tier", "") for row in seed_rows)
    base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))
    base.write_csv_atomic(
        EXCLUSION_AUDIT_CSV_PATH,
        availability_audit,
        list(availability_audit[0].keys()) if availability_audit else ["source_subset", "doc_type_name", "required_count", "selected_count", "skip_reason"],
    )
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- route_label: `{ACTIVE_ROUTE['route_label']}`",
        f"- candidate_count: `{len(seed_rows)}`",
        f"- final_target_count: `{ACTIVE_ROUTE['target_count']}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- reuse_tier_counts: `{dict(reuse_counts)}`",
        "",
        "## source subset counts",
        "| source_subset | count |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(
        [
            "",
            "## checks",
            "| check | result |",
            "| --- | --- |",
            "| candidate source schedule satisfied | `pass` |",
            "| no batch family_id/label_path/raw_path duplicate | `pass` |",
            "| prior overlap policy | `pass` | Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용 |",
            "| global eval and quality tail exclusion | `pass` | dev/test, audit, hard/soft fail family는 재사용 금지 |",
        ]
    )
    write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")


def build_seed_registry() -> list[dict]:
    base.ensure_dirs(
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
    exclusion_rows = collect_exclusion_rows()
    exclusion_sets = build_exclusion_sets(exclusion_rows)
    global ACTIVE_ROUTE
    ACTIVE_ROUTE, records, availability_audit = choose_route(exclusion_sets)
    seed_rows = [build_seed_from_record(record) for record in records]
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["source_subset"], row["seed_sample_id"]))
    write_preflight(seed_rows, exclusion_sets, availability_audit)
    base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    for path in [SEED_REGISTRY_PATH, SEED_READY_PATH, SEED_PREFLIGHT_CSV_PATH, SEED_PREFLIGHT_MD_PATH, EXCLUSION_AUDIT_CSV_PATH]:
        base.copy_file_to_run_inputs(path, RUN_INPUTS_DIR)
    return seed_rows


def reject_metadata(row: dict, reason: str) -> dict:
    if reason == "quota_surplus":
        return {
            **row,
            "pool_class": "quota_surplus",
            "quality_failure": NO,
            "tail_class": "quota_surplus_not_quality_failure",
            "future_candidate_reusable": YES,
            "candidate_reuse_policy": "reuse_allowed_as_surplus_candidate_after_dedup",
            "selection_reason": "strict_pass_not_selected_due_final_source_quota",
            "not_selected_reason": "source_quota_filled",
        }
    return {
        **row,
        "pool_class": "quality_reject",
        "quality_failure": YES,
        "tail_class": reason,
        "future_candidate_reusable": NO,
        "candidate_reuse_policy": "do_not_reuse_without_repair_review",
        "selection_reason": "",
        "not_selected_reason": reason,
    }


def split_for_final_rows(final_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in final_rows:
        grouped.setdefault(row["doc_type_name"], []).append(row)
    with_splits: list[dict] = []
    for doc_type_name, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: (item["source_subset"], item["seed_sample_id"]))
        total = len(rows)
        if total >= 5:
            train_count = total - 2
            dev_count = 1
        elif total >= 3:
            train_count = total - 2
            dev_count = 1
        else:
            train_count = total
            dev_count = 0
        # split-lock reuse row는 반드시 train에 먼저 배치해서 cross-task train/eval leakage를 막는다.
        locked_train_rows = [row for row in rows if row.get("locked_split") == "train"]
        unlocked_rows = [row for row in rows if row.get("locked_split") != "train"]
        ordered_rows = locked_train_rows + unlocked_rows
        for index, row in enumerate(ordered_rows):
            split = "train" if index < max(train_count, len(locked_train_rows)) else "dev" if index < max(train_count, len(locked_train_rows)) + dev_count else "test"
            with_splits.append({**row, "split": split, "dataset_disposition": split})
    return sorted(with_splits, key=lambda item: int(item["selection_rank"]))


def compile_final_package(merged_rows: list[dict]) -> dict[str, list[dict]]:
    selected_rows = [row for row in merged_rows if row.get("selected_for_seed") == YES]
    candidate_pool = [{**row, "pool_class": "candidate_pool", "package_role": PACKAGE_ROLE} for row in selected_rows]
    accepted = []
    rejected = []
    for row in selected_rows:
        if row.get("final_status") != "pass":
            rejected.append(reject_metadata(row, "final_status_failure"))
        elif row.get("train_eligible") != YES or row.get("audit_required") == YES:
            rejected.append(reject_metadata(row, "audit_or_not_train_eligible"))
        else:
            accepted.append({**row, "pool_class": "strict_accepted", "quality_failure": NO})

    quota = dict(ACTIVE_ROUTE["final_source_counts"])
    selected_by_source = {source_subset: 0 for source_subset in quota}
    final_rows = []
    accepted_with_selection = []
    for row in sorted(
        accepted,
        key=lambda item: (
            0 if item.get("reuse_tier") == "Tier 0 fresh-only" else 1,
            -float(item.get("weighted_score", 0)),
            item["source_subset"],
            item["seed_sample_id"],
        ),
    ):
        source_subset = row["source_subset"]
        if selected_by_source.get(source_subset, 0) >= quota.get(source_subset, 0):
            surplus = reject_metadata(row, "quota_surplus")
            accepted_with_selection.append(surplus)
            rejected.append(surplus)
            continue
        selected_by_source[source_subset] = selected_by_source.get(source_subset, 0) + 1
        selected = {
            **row,
            "pool_class": "final_package_selected",
            "selection_rank": str(len(final_rows) + 1),
            "selection_reason": "strict_pass_selected_by_source_quota",
            "not_selected_reason": "",
            "package_role": PACKAGE_ROLE,
            "batch_status": CANDIDATE_BATCH_STATUS,
            "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
            "downstream_consumption_allowed": NO,
            "count_allowed": NO,
            "count_disposition": COUNT_DISPOSITION,
            "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
        }
        accepted_with_selection.append(selected)
        final_rows.append(selected)
    final_rows = split_for_final_rows(final_rows)
    return {
        "candidate_pool": candidate_pool,
        "accepted_pool": accepted_with_selection,
        "rejected_pool": rejected,
        "quality_tail": [row for row in rejected if row.get("quality_failure") == YES],
        "quota_surplus": [row for row in rejected if row.get("pool_class") == "quota_surplus"],
        "final_rows": final_rows,
    }


def union_fields(rows: list[dict]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def final_payload(row: dict) -> dict:
    return {
        "problem_id": row["candidate_id"],
        "seed_sample_id": row["seed_sample_id"],
        "family_id": row["family_id"],
        "doc_type_name": row["doc_type_name"],
        "source_subset": row["source_subset"],
        "sampling_lane": row.get("sampling_lane", ""),
        "problem_task_type": row["problem_task_type"],
        "problem_generation_mode": row["problem_generation_mode"],
        "generated_problem": row["generated_problem"],
        "split_strategy": row.get("split_strategy", ""),
        "focus_issue": row.get("focus_issue", ""),
        "multi_query_signal": row.get("multi_query_signal", ""),
        "split_focus_hint": row.get("split_focus_hint", ""),
        "gold_short_answer": row["gold_short_answer"],
        "gold_reference_explanation": row["gold_reference_explanation"],
        "answer_mode": row.get("answer_mode", ""),
        "explanation_target": row.get("explanation_target", ""),
        "weighted_score": row["weighted_score"],
        "error_tags": row.get("error_tags", ""),
        "generator_model": row.get("generator_model", ""),
        "generation_mode": row.get("generation_mode", ""),
        "version_tag": row.get("version_tag", ""),
        "run_name": row.get("run_name", ""),
        "label_path": row.get("label_path", ""),
        "raw_path": row.get("raw_path", ""),
        "reuse_tier": row.get("reuse_tier", ""),
        "source_task": row.get("source_task", ""),
        "source_split": row.get("source_split", ""),
        "locked_split": row.get("locked_split", ""),
        "reuse_reason": row.get("reuse_reason", ""),
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
        "split": row["split"],
    }


def write_compiled_outputs(compiled: dict[str, list[dict]], merged_rows: list[dict]) -> list[dict]:
    final_rows = compiled["final_rows"]
    manifest_rows = []
    for row in final_rows:
        manifest_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "reuse_tier": row.get("reuse_tier", ""),
                "source_task": row.get("source_task", ""),
                "source_split": row.get("source_split", ""),
                "locked_split": row.get("locked_split", ""),
                "reuse_reason": row.get("reuse_reason", ""),
                "split": row["split"],
                "dataset_disposition": row["split"],
                "train_eligible": row.get("train_eligible", YES),
                "audit_required": row.get("audit_required", NO),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
                "package_role": PACKAGE_ROLE,
                "batch_status": CANDIDATE_BATCH_STATUS,
                "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
                "downstream_consumption_allowed": NO,
                "count_allowed": NO,
                "count_disposition": COUNT_DISPOSITION,
                "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
            }
        )
    for split_name in ["train", "dev", "test"]:
        payloads = [final_payload(row) for row in final_rows if row["split"] == split_name]
        base.write_jsonl_atomic(PROCESSED_DIR / f"{split_name}.jsonl", payloads)
    base.write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["problem_id"])
    base.write_csv_atomic(PROBLEM_AUDIT_QUEUE_PATH, [], ["problem_id", "seed_sample_id", "family_id", "doc_type_name", "source_subset", "audit_reason"])
    for path, rows in [
        (CANDIDATE_POOL_PATH, compiled["candidate_pool"]),
        (ACCEPTED_POOL_PATH, compiled["accepted_pool"]),
        (REJECTED_POOL_PATH, compiled["rejected_pool"]),
        (TAIL_TAXONOMY_PATH, compiled["quality_tail"]),
        (QUOTA_SURPLUS_POOL_PATH, compiled["quota_surplus"]),
        (FINAL_PACKAGE_CSV_PATH, final_rows),
    ]:
        base.write_csv_atomic(path, rows, union_fields(rows) if rows else ["empty"])
    # merged score에는 raw candidate 전체와 final package contract를 같이 남겨 reviewer가 미선택 row를 추적할 수 있게 한다.
    final_ids = {row["candidate_id"] for row in final_rows}
    merged_with_contract = []
    for row in merged_rows:
        selected = row["candidate_id"] in final_ids
        merged_with_contract.append(
            {
                **row,
                "package_role": PACKAGE_ROLE,
                "batch_status": CANDIDATE_BATCH_STATUS,
                "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
                "downstream_consumption_allowed": NO,
                "count_allowed": NO,
                "count_disposition": COUNT_DISPOSITION,
                "promotion_contract_status": PROMOTION_CONTRACT_STATUS if selected else "candidate_pool_not_promoted",
                "final_package_selected": YES if selected else NO,
            }
        )
    base.write_csv_atomic(MERGED_SCORES_PATH, merged_with_contract, union_fields(merged_with_contract))
    return manifest_rows


def render_markdown_outputs(compiled: dict[str, list[dict]], manifest_rows: list[dict]) -> None:
    final_rows = compiled["final_rows"]
    doc_counts = Counter(row["doc_type_name"] for row in final_rows)
    source_counts = Counter(row["source_subset"] for row in final_rows)
    split_counts = Counter(row["split"] for row in final_rows)
    reuse_counts = Counter(row.get("reuse_tier", "") for row in final_rows)
    summary_rows = [
        {
            "metric": "candidate_total",
            "value": str(len(compiled["candidate_pool"])),
        },
        {
            "metric": "accepted_total",
            "value": str(len(compiled["accepted_pool"])),
        },
        {
            "metric": "final_package_total",
            "value": str(len(final_rows)),
        },
        {
            "metric": "quality_tail_total",
            "value": str(len(compiled["quality_tail"])),
        },
        {
            "metric": "quota_surplus_total",
            "value": str(len(compiled["quota_surplus"])),
        },
    ]
    base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    lines = [
        f"# descriptive production wave `{VERSION_TAG}`",
        "",
        "## package summary",
        f"- route_label: `{ACTIVE_ROUTE['route_label']}`",
        f"- candidate_total: `{len(compiled['candidate_pool'])}`",
        f"- accepted_total: `{len(compiled['accepted_pool'])}`",
        f"- final_package_total: `{len(final_rows)}`",
        f"- quality_tail_total: `{len(compiled['quality_tail'])}`",
        f"- quota_surplus_total: `{len(compiled['quota_surplus'])}`",
        f"- split_counts: `{dict(split_counts)}`",
        f"- reuse_tier_counts: `{dict(reuse_counts)}`",
        "",
        "## final doc type counts",
        "| doc_type | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(doc_counts.items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## final source counts", "| source_subset | count |", "| --- | ---: |"])
    for key, value in sorted(source_counts.items()):
        lines.append(f"| `{key}` | `{value}` |")
    write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    write_text_atomic(FINAL_PACKAGE_MD_PATH, "\n".join(lines) + "\n")

    evidence = [
        f"# evidence summary `{VERSION_TAG}`",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| route_name | `{ROUTE_NAME}` |",
        f"| package_role | `{PACKAGE_ROLE}` |",
        f"| seed_policy | `Tier 0 fresh first + Tier 2 train-only split-lock fallback` |",
        f"| batch_status | `{CANDIDATE_BATCH_STATUS}` |",
        f"| count_reflection_status | `{CANDIDATE_REFLECTION_STATUS}` |",
        f"| downstream_consumption_allowed | `{NO}` |",
        f"| count_allowed | `{NO}` |",
        f"| candidate_total | `{len(compiled['candidate_pool'])}` |",
        f"| final_package_total | `{len(final_rows)}` |",
        f"| hard_soft_audit_in_final | `0/0/0` |",
        f"| manifest_count | `{len(manifest_rows)}` |",
        f"| train/dev/test | `{split_counts.get('train', 0)}/{split_counts.get('dev', 0)}/{split_counts.get('test', 0)}` |",
        "| decision | `reviewer_signoff_needed_before_count_reflection` |",
    ]
    write_text_atomic(EVIDENCE_SUMMARY_MD_PATH, "\n".join(evidence) + "\n")
    spec_lines = [
        f"# package spec `{VERSION_TAG}`",
        "",
        f"- primary route: `target 40 / candidate 56`",
        f"- fallback route: `target 24 / candidate 36`",
        f"- active route: `{ACTIVE_ROUTE['route_label']}`",
        "- seed policy: Tier 0 fresh-only 우선, 부족분은 objective train family의 Tier 2 train-only split-lock reuse 허용",
        "- count rule: candidate 전체가 아니라 strict final package만 reviewer sign-off 이후 count 후보",
        "",
    ]
    write_text_atomic(PACKAGE_SPEC_MD_PATH, "\n".join(spec_lines))


def write_manifest(seed_rows: list[dict], merged_rows: list[dict], manifest_rows: list[dict], compiled: dict[str, list[dict]]) -> dict:
    final_rows = compiled["final_rows"]
    split_counts = Counter(row.get("split", "") for row in final_rows)
    reuse_counts = Counter(row.get("reuse_tier", "") for row in final_rows)
    success = len(final_rows) == ACTIVE_ROUTE["target_count"]
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "route_name": ROUTE_NAME,
        "route_label": ACTIVE_ROUTE["route_label"],
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
        "seed_registry_strategy": "tier0_fresh_first_with_tier2_objective_train_split_locked_reuse",
        "seed_reuse_policy": {
            "tier0": "fresh-only source family without prior overlap",
            "tier2": "objective train family may be reused for descriptive train only under split-lock",
            "global_eval_holdout": "counted dev/test family remains blocked across tasks",
            "quality_tail_reuse": "blocked",
        },
        "seed_registry_count": len(seed_rows),
        "candidate_total": len(compiled["candidate_pool"]),
        "accepted_total": len(compiled["accepted_pool"]),
        "final_package_total": len(final_rows),
        "quality_tail_total": len(compiled["quality_tail"]),
        "quota_surplus_total": len(compiled["quota_surplus"]),
        "generation_count": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_answerability_count": base.load_jsonl_count(ANSWERABILITY_LOG_PATH),
        "judge_taskfit_count": base.load_jsonl_count(TASKFIT_LOG_PATH),
        "merged_count": base.load_csv_count(MERGED_SCORES_PATH),
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "split_counts": dict(split_counts),
        "reuse_tier_counts": dict(reuse_counts),
        "success_criteria": {
            "final_package_equals_target": ACTIVE_ROUTE["target_count"],
            "hard_soft_audit_in_final": "0/0/0",
            "count_reflection_requires_reviewer_signoff": True,
        },
        "success_result": {
            "passed": success,
            "reason": "final package target met" if success else "final package target not met",
        },
        "api_call_summary": {
            "openai_api": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
            "gemini_api": base.load_jsonl_count(GROUNDING_LOG_PATH)
            + base.load_jsonl_count(ANSWERABILITY_LOG_PATH)
            + base.load_jsonl_count(TASKFIT_LOG_PATH),
            "total_api_calls": base.load_jsonl_count(GENERATED_PROBLEMS_PATH)
            + base.load_jsonl_count(GROUNDING_LOG_PATH)
            + base.load_jsonl_count(ANSWERABILITY_LOG_PATH)
            + base.load_jsonl_count(TASKFIT_LOG_PATH),
        },
        "artifact_paths": {
            "seed_registry": repo_rel(SEED_REGISTRY_PATH),
            "seed_ready": repo_rel(SEED_READY_PATH),
            "seed_preflight": repo_rel(SEED_PREFLIGHT_MD_PATH),
            "generated_problems": repo_rel(GENERATED_PROBLEMS_PATH),
            "merged_scores": repo_rel(MERGED_SCORES_PATH),
            "final_package": repo_rel(FINAL_PACKAGE_CSV_PATH),
            "candidate_pool": repo_rel(CANDIDATE_POOL_PATH),
            "accepted_pool": repo_rel(ACCEPTED_POOL_PATH),
            "rejected_pool": repo_rel(REJECTED_POOL_PATH),
            "tail_taxonomy": repo_rel(TAIL_TAXONOMY_PATH),
            "quota_surplus_pool": repo_rel(QUOTA_SURPLUS_POOL_PATH),
            "problem_dataset_manifest": repo_rel(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_train": repo_rel(PROBLEM_TRAIN_PATH),
            "problem_dev": repo_rel(PROBLEM_DEV_PATH),
            "problem_test": repo_rel(PROBLEM_TEST_PATH),
            "evidence_summary": repo_rel(EVIDENCE_SUMMARY_MD_PATH),
        },
    }
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    base.write_json_atomic(
        COMPILER_MANIFEST_PATH,
        {
            "compiler_manifest_version": "descriptive_production_wave_v1",
            "package_role": PACKAGE_ROLE,
            "route_label": ACTIVE_ROUTE["route_label"],
            "selection_policy": "pass/no-audit accepted rows selected by final source quota and weighted score",
            "row_counts": {
                "candidate_total": len(compiled["candidate_pool"]),
                "accepted_total": len(compiled["accepted_pool"]),
                "final_package_total": len(final_rows),
                "quality_tail_total": len(compiled["quality_tail"]),
                "quota_surplus_total": len(compiled["quota_surplus"]),
            },
            "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
            "count_allowed": NO,
        },
    )
    return manifest


def main() -> dict:
    configure_base_paths()
    seed_rows = build_seed_registry()
    print(f"[descriptive wave] route={ACTIVE_ROUTE['route_label']} candidates={len(seed_rows)}", flush=True)
    base.run_generation(mode="main")
    base.run_generation(mode="strict_finalize")
    base.run_judges(mode="main")
    base.run_judges(mode="strict_finalize")
    merged_rows = base.merge_scores()
    compiled = compile_final_package(merged_rows)
    manifest_rows = write_compiled_outputs(compiled, merged_rows)
    render_markdown_outputs(compiled, manifest_rows)
    manifest = write_manifest(seed_rows, merged_rows, manifest_rows, compiled)
    print(f"[descriptive wave] final_package={manifest['final_package_total']} success={manifest['success_result']['passed']}", flush=True)
    return manifest


ACTIVE_ROUTE: dict = {}


if __name__ == "__main__":
    main()
