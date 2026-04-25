import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `pb3`는 current objective recipe(`r2`)를 fresh AI Hub QA seed에 태우는 첫 production batch다.
# 기존 `pb2`처럼 runner를 재사용하되, seed registry만 새로 만들고 preflight를 강제한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.v2_objective_difficulty_patch_r2 import run_difficulty_patch as r2


base = r2.base

# explanation_generation 모듈들은 과거 실행 습관상 top-level `common`, `settings` import를 쓰므로,
# 구조 개편 후 경로와 legacy 경로를 순서대로 확인해 fresh seed 변환 로직을 그대로 재사용한다.
EXPLANATION_DIR_CANDIDATES = [
    base.PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "explanation_generation",
    base.PROJECT_ROOT / "scripts" / "aihub" / "explanation_generation",
]
for explanation_dir in reversed(EXPLANATION_DIR_CANDIDATES):
    if explanation_dir.exists() and str(explanation_dir) not in sys.path:
        sys.path.insert(0, str(explanation_dir))

import common as explanation_common
from extract_evidence_cards import build_card
from generate_explanations import build_local_fallback_explanation, postprocess_generated_explanation
from settings import DATASET_SPECS
from transform_problems import build_transformed_sample


VERSION_TAG = "pb3_objective_current_r2"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-24_190808"
RUN_PURPOSE = "objective_r2_current_production_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROMPT_DIR = SCRIPT_DIR.parent / "v2_objective_difficulty_patch_r2" / "prompts"
INTERIM_DIR = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"

V7_PROCESSED_DIR = base.PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final"
V7_SAMPLE_REGISTRY_PATH = (
    base.PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "explanation_generation"
    / "llm_runs"
    / "2026-04-14_103340_v7_tail_stabilization_full_01_04"
    / "inputs"
    / "sample_registry_v7.csv"
)
REFERENCE_R2_SEED_REGISTRY_PATH = (
    base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "v2_difficulty_patch_r2" / "seed_registry.csv"
)
REFERENCE_PB2_SEED_REGISTRY_PATH = (
    base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb2_objective_candidate"
    / "seed_registry.csv"
)

EXPECTED_DOC_TYPE_COUNTS = {
    "법령_QA": 10,
    "해석례_QA": 10,
    "결정례_QA": 10,
    "판결문_QA": 10,
}

ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "single_best_rule",
    "definition": "single_best_rule",
    "requirement": "single_best_rule",
    "application": "single_best_application",
    "scope": "single_best_scope",
}

EXPLANATION_GENERATION_VARIANT = {
    "name": "without_long_answer",
    "label": "long_answer 제외",
    "include_long_answer": False,
}


def configure_pb3():
    # `r2` runner와 그 base 모듈의 모든 산출물 경로를 `pb3` 전용 폴더로 재배선한다.
    r2.VERSION_TAG = VERSION_TAG
    r2.RUN_DATE = RUN_DATE
    r2.RUN_PURPOSE = RUN_PURPOSE
    r2.RUN_NAME = RUN_NAME
    r2.PROMPT_DIR = PROMPT_DIR
    r2.INTERIM_DIR = INTERIM_DIR
    r2.PROCESSED_DIR = PROCESSED_DIR
    r2.RUN_DIR = RUN_DIR
    r2.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    r2.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    r2.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    r2.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    r2.RUN_MERGED_DIR = RUN_MERGED_DIR
    r2.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    r2.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    r2.SEED_READY_PATH = SEED_READY_PATH
    r2.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    r2.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    r2.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    r2.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    r2.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    r2.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    r2.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    r2.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    r2.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    r2.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    r2.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    r2.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    r2.SIDE_BY_SIDE_MD_PATH = BATCH_SUMMARY_MD_PATH
    r2.SIDE_BY_SIDE_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    r2.REFERENCE_PATCH_MERGED_PATH = MERGED_SCORES_PATH
    r2.REFERENCE_PATCH_ROWS_CACHE = None
    r2.configure_base()

    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Keyedness": KEYEDNESS_LOG_PATH,
        "DistractorFit": DISTRACTORFIT_LOG_PATH,
        "NearMiss": NEARMISS_LOG_PATH,
    }


def load_csv_rows_if_exists(path):
    if not path.exists():
        return []
    return base.load_csv_rows(path)


def load_jsonl_rows_if_exists(path):
    if not path.exists():
        return []
    return base.load_jsonl(path)


def collect_excluded_rows():
    # fresh pool은 current objective line뿐 아니라 `v7` held-out/audit와 과거 selected registry도 보호한다.
    rows = []
    for filename in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        rows.extend(load_jsonl_rows_if_exists(V7_PROCESSED_DIR / filename))
    rows.extend(load_csv_rows_if_exists(V7_PROCESSED_DIR / "audit_queue.csv"))
    rows.extend(load_csv_rows_if_exists(V7_SAMPLE_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(REFERENCE_R2_SEED_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(REFERENCE_PB2_SEED_REGISTRY_PATH))
    return rows


def build_exclusion_sets(rows):
    return {
        "sample_ids": {row.get("sample_id", "") or row.get("seed_sample_id", "") for row in rows if row.get("sample_id", "") or row.get("seed_sample_id", "")},
        "family_ids": {row.get("family_id", "") for row in rows if row.get("family_id", "")},
        "label_paths": {row.get("label_path", "") for row in rows if row.get("label_path", "")},
        "raw_paths": {row.get("raw_path", "") for row in rows if row.get("raw_path", "")},
    }


def select_fresh_registry_records():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)
    records = []
    sample_order = 1
    batch_family_ids = set()
    batch_label_paths = set()
    batch_raw_paths = set()

    for spec in DATASET_SPECS:
        label_paths = explanation_common.list_label_files(spec["label_glob"])
        raw_paths = explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = explanation_common.normalize_label_payload(
                    candidate_label_path,
                    explanation_common.load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                try:
                    candidate_raw_path = explanation_common.locate_raw_path(
                        raw_paths,
                        spec["doc_type_name"],
                        candidate_payload["info"],
                    )
                except FileNotFoundError:
                    continue
                candidate_family_id = explanation_common.make_family_id(spec["doc_type_name"], candidate_payload["info"])
                candidate_label_path_text = str(candidate_label_path)
                candidate_raw_path_text = str(candidate_raw_path)

                if candidate_family_id in exclusion_sets["family_ids"] or candidate_family_id in batch_family_ids:
                    continue
                if candidate_label_path_text in exclusion_sets["label_paths"] or candidate_label_path_text in batch_label_paths:
                    continue
                if candidate_raw_path_text in exclusion_sets["raw_paths"] or candidate_raw_path_text in batch_raw_paths:
                    continue
                chosen = (
                    candidate_index,
                    candidate_label_path,
                    candidate_payload,
                    candidate_raw_path,
                    candidate_family_id,
                )
                break

            if chosen is None:
                raise RuntimeError(f"{spec['source_subset']}에서 fresh pb3 seed를 충분히 찾지 못했습니다.")

            used_indices.add(chosen[0])
            label_path, payload, raw_path, family_id = chosen[1], chosen[2], chosen[3], chosen[4]
            info = payload["info"]
            label = payload["label"]
            sample_id = f"pb3_{sample_order:03d}"

            record = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": local_order,
                "selected_index": chosen[0],
                "selection_note": "pb3 fresh objective seed: v7/current objective used family_id/label_path/raw_path 제외 후 신규 선택",
            }
            records.append(record)
            batch_family_ids.add(family_id)
            batch_label_paths.add(str(label_path))
            batch_raw_paths.add(str(raw_path))
            sample_order += 1

    return records, exclusion_sets


def choose_explanation_style(sample):
    # 객관식 seed 설명은 정답 anchor가 선명해야 하므로 다중 style에서는 법리 우선형을 우선한다.
    styles = sample.get("candidate_styles", [])
    if "legal_priority" in styles:
        return "legal_priority"
    if styles:
        return styles[0]
    return "single"


def build_seed_row(record):
    card = build_card(record)
    transformed = build_transformed_sample(card)
    style_name = choose_explanation_style(transformed)
    generated_explanation = build_local_fallback_explanation(
        transformed,
        style_name,
        EXPLANATION_GENERATION_VARIANT,
    )
    generated_explanation = postprocess_generated_explanation(transformed, generated_explanation)
    answer_mode = transformed.get("answer_mode", "") or "criteria"

    return {
        "seed_sample_id": record["sample_id"],
        "reference_sample_id": record["sample_id"],
        "family_id": record["family_id"],
        "doc_type_name": record["doc_type_name"],
        "source_subset": record["source_subset"],
        "sampling_lane": record["sampling_lane"],
        "answer_mode": answer_mode,
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "single_best_rule"),
        "explanation_target": transformed.get("explanation_target", ""),
        "selection_role": "objective_pb3_current_r2_seed",
        "selection_note": "fresh AI Hub QA seed를 current objective recipe r2로 생산하기 위한 pb3 seed",
        "transformed_problem": transformed["transformed_problem"],
        "short_answer": transformed["short_answer"],
        "generated_explanation": generated_explanation,
        "rule_basis": transformed["evidence_card"].get("rule_basis", ""),
        "fact_basis": transformed["evidence_card"].get("fact_basis", ""),
        "label_path": record["label_path"],
        "raw_path": record["raw_path"],
        "selected_at_utc": base.utc_now_iso(),
        "pb3_seed_generation_mode": "deterministic_evidence_card_template",
        "pb3_explanation_style": style_name,
    }


def build_preflight_rows(seed_rows, exclusion_sets):
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows = []

    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "family_duplicate_in_batch": "예" if family_counts[row["family_id"]] > 1 else "아니오",
                "label_path_duplicate_in_batch": "예" if label_counts[row["label_path"]] > 1 else "아니오",
                "raw_path_duplicate_in_batch": "예" if raw_counts[row["raw_path"]] > 1 else "아니오",
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


def assert_preflight(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    if len(seed_rows) != 40:
        raise RuntimeError(f"pb3 seed 수가 40개가 아닙니다: {len(seed_rows)}")
    for doc_type_name, expected_count in EXPECTED_DOC_TYPE_COUNTS.items():
        actual_count = doc_type_counts.get(doc_type_name, 0)
        if actual_count != expected_count:
            raise RuntimeError(f"{doc_type_name} seed 수가 {expected_count}개가 아닙니다: {actual_count}")
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
            raise RuntimeError(f"pb3 seed preflight 중복/누수 실패: {row['seed_sample_id']}")
    if lane_counts.get("generalization_03_04", 0) == 0 or lane_counts.get("expansion_01_02", 0) == 0:
        raise RuntimeError(f"pb3 source lane 분포가 비어 있습니다: {dict(lane_counts)}")


def write_preflight_report(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)

    base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_type_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        "",
        "## source subset counts",
        "| source_subset | count |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    lines.extend(
        [
            "| total seed count is 40 | `pass` |",
            "| doc type count is 10 each | `pass` |",
            "| no batch family_id duplicate | `pass` |",
            "| no prior family_id/label_path/raw_path overlap | `pass` |",
            "| both generalization and expansion lanes represented | `pass` |",
        ]
    )
    base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry():
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
    records, exclusion_sets = select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["sampling_lane"], row["seed_sample_id"]))
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_preflight(seed_rows, preflight_rows)

    base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def load_empty_reference_rows():
    # fresh seed에는 같은 seed의 historical objective row가 없으므로 reference prompt 영역은 빈 값으로 둔다.
    return {}


def summarize_rows(rows):
    selected_rows = [row for row in rows if row["selected_for_seed"] == "예"]
    return {
        "selected_pass_count": sum(1 for row in selected_rows if row["final_status"] == "pass"),
        "selected_hard_fail_count": sum(1 for row in selected_rows if row["final_status"] == "hard_fail"),
        "selected_soft_fail_count": sum(1 for row in selected_rows if row["final_status"] == "soft_fail"),
        "selected_train_eligible_count": sum(1 for row in selected_rows if row.get("train_eligible") == "예"),
        "selected_audit_required_count": sum(1 for row in selected_rows if row.get("audit_required") == "예"),
        # `pb3`는 fresh pool 검증이므로 doc type뿐 아니라 source lane yield도 review 대상이다.
        "doc_type_planned_counter": Counter(row["doc_type_name"] for row in selected_rows),
        "lane_planned_counter": Counter(row.get("sampling_lane", "") for row in selected_rows),
        "source_subset_planned_counter": Counter(row.get("source_subset", "") for row in selected_rows),
        "doc_type_train_counter": Counter(
            row["doc_type_name"]
            for row in selected_rows
            if row["final_status"] == "pass" and row.get("train_eligible") == "예"
        ),
        "doc_type_audit_counter": Counter(
            row["doc_type_name"]
            for row in selected_rows
            if row.get("audit_required") == "예"
        ),
        "doc_type_hard_fail_counter": Counter(
            row["doc_type_name"]
            for row in selected_rows
            if row["final_status"] == "hard_fail"
        ),
        "doc_type_soft_fail_counter": Counter(
            row["doc_type_name"]
            for row in selected_rows
            if row["final_status"] == "soft_fail"
        ),
        "lane_train_counter": Counter(
            row.get("sampling_lane", "")
            for row in selected_rows
            if row["final_status"] == "pass" and row.get("train_eligible") == "예"
        ),
        "lane_audit_counter": Counter(
            row.get("sampling_lane", "")
            for row in selected_rows
            if row.get("audit_required") == "예"
        ),
        "lane_hard_fail_counter": Counter(
            row.get("sampling_lane", "")
            for row in selected_rows
            if row["final_status"] == "hard_fail"
        ),
        "lane_soft_fail_counter": Counter(
            row.get("sampling_lane", "")
            for row in selected_rows
            if row["final_status"] == "soft_fail"
        ),
    }


def build_batch_summary(pb3_rows):
    selected_rows = [row for row in pb3_rows if row["selected_for_seed"] == "예"]
    selected_rows.sort(key=lambda row: (row["doc_type_name"], row["seed_sample_id"]))
    summary = summarize_rows(pb3_rows)
    doc_types = sorted(EXPECTED_DOC_TYPE_COUNTS)

    summary_rows = []
    for doc_type_name in doc_types:
        summary_rows.append(
            {
                "doc_type_name": doc_type_name,
                "planned_seed_count": str(EXPECTED_DOC_TYPE_COUNTS[doc_type_name]),
                "train_eligible_count": str(summary["doc_type_train_counter"].get(doc_type_name, 0)),
                "audit_required_count": str(summary["doc_type_audit_counter"].get(doc_type_name, 0)),
                "hard_fail_count": str(summary["doc_type_hard_fail_counter"].get(doc_type_name, 0)),
                "soft_fail_count": str(summary["doc_type_soft_fail_counter"].get(doc_type_name, 0)),
            }
        )
    lane_rows = []
    for sampling_lane, planned_count in sorted(summary["lane_planned_counter"].items()):
        lane_rows.append(
            {
                "sampling_lane": sampling_lane,
                "planned_seed_count": str(planned_count),
                "train_eligible_count": str(summary["lane_train_counter"].get(sampling_lane, 0)),
                "audit_required_count": str(summary["lane_audit_counter"].get(sampling_lane, 0)),
                "hard_fail_count": str(summary["lane_hard_fail_counter"].get(sampling_lane, 0)),
                "soft_fail_count": str(summary["lane_soft_fail_counter"].get(sampling_lane, 0)),
            }
        )

    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected_rows)}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        "",
        "## doc type yield",
        "| doc_type | planned | train_eligible | audit | hard_fail | soft_fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['doc_type_name']}` | `{row['planned_seed_count']}` | `{row['train_eligible_count']}` | `{row['audit_required_count']}` | `{row['hard_fail_count']}` | `{row['soft_fail_count']}` |"
        )

    lines.extend(
        [
            "",
            "## source lane yield",
            "| sampling_lane | planned | train_eligible | audit | hard_fail | soft_fail |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lane_rows:
        lines.append(
            f"| `{row['sampling_lane']}` | `{row['planned_seed_count']}` | `{row['train_eligible_count']}` | `{row['audit_required_count']}` | `{row['hard_fail_count']}` | `{row['soft_fail_count']}` |"
        )

    lines.extend(
        [
            "",
            "## source subset counts",
            "| source_subset | planned |",
            "| --- | ---: |",
        ]
    )
    for source_subset, planned_count in sorted(summary["source_subset_planned_counter"].items()):
        lines.append(f"| `{source_subset}` | `{planned_count}` |")

    lines.extend(["", "## row status", "| seed_sample_id | doc_type | source_lane | final_status | train_eligible | audit_required | error_tags |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in selected_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['doc_type_name']}` | `{row.get('sampling_lane', '')}` | `{row['final_status']}` | `{row.get('train_eligible', '')}` | `{row.get('audit_required', '')}` | `{row.get('error_tags', '')}` |"
        )

    base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, list(summary_rows[0].keys()))
    # Lane summary is kept as a separate CSV so the doc-type table stays backwards compatible.
    base.write_csv_atomic(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, list(lane_rows[0].keys()))
    base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    summary = summarize_rows(merged_rows)
    # Manifest에는 reviewer가 재계산하지 않아도 되는 최소 집계축을 함께 남긴다.
    split_counts = Counter(row.get("split", "") for row in manifest_rows)
    manifest_doc_type_counts = Counter(row.get("doc_type_name", "") for row in manifest_rows)
    manifest_lane_counts = Counter(row.get("sampling_lane", "") for row in manifest_rows)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "candidate_recipe_source": "v2_difficulty_patch_r2",
        "seed_registry_strategy": "fresh_aihub_qa_training_pool_excluding_v7_current_objective_and_heldout_rows",
        "seed_registry_count": len(seed_rows),
        "seed_doc_type_counts": dict(Counter(row.get("doc_type_name", "") for row in seed_rows)),
        "seed_lane_counts": dict(Counter(row.get("sampling_lane", "") for row in seed_rows)),
        "seed_source_subset_counts": dict(Counter(row.get("source_subset", "") for row in seed_rows)),
        "seed_preflight_csv_path": str(SEED_PREFLIGHT_CSV_PATH),
        "generation_count": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": base.load_jsonl_count(NEARMISS_LOG_PATH),
        "merged_count": base.load_csv_count(MERGED_SCORES_PATH),
        "selected_pass_count": summary["selected_pass_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "dataset_split_counts": dict(split_counts),
        "dataset_doc_type_counts": dict(manifest_doc_type_counts),
        "dataset_lane_counts": dict(manifest_lane_counts),
        "selected_doc_type_train_counts": dict(summary["doc_type_train_counter"]),
        "selected_doc_type_audit_counts": dict(summary["doc_type_audit_counter"]),
        "selected_doc_type_hard_fail_counts": dict(summary["doc_type_hard_fail_counter"]),
        "selected_doc_type_soft_fail_counts": dict(summary["doc_type_soft_fail_counter"]),
        "selected_lane_train_counts": dict(summary["lane_train_counter"]),
        "selected_lane_audit_counts": dict(summary["lane_audit_counter"]),
        "selected_lane_hard_fail_counts": dict(summary["lane_hard_fail_counter"]),
        "selected_lane_soft_fail_counts": dict(summary["lane_soft_fail_counter"]),
        "batch_summary_count": len(summary_rows),
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight": str(SEED_PREFLIGHT_CSV_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "batch_summary_md": str(BATCH_SUMMARY_MD_PATH),
            "batch_summary_csv": str(BATCH_SUMMARY_CSV_PATH),
            "batch_lane_summary_csv": str(BATCH_LANE_SUMMARY_CSV_PATH),
            "problem_train": str(PROBLEM_TRAIN_PATH),
            "problem_dev": str(PROBLEM_DEV_PATH),
            "problem_test": str(PROBLEM_TEST_PATH),
            "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
        },
    }
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def main():
    configure_pb3()
    base.build_seed_registry = build_seed_registry
    base.load_reference_v2_rows = load_empty_reference_rows
    base.build_local_fallback_problem = r2.build_local_fallback_problem
    base.postprocess_problem = r2.postprocess_problem
    base.build_generation_messages = r2.build_generation_messages
    base.build_side_by_side_examples = build_batch_summary
    base.build_run_manifest = build_run_manifest
    return base.main()


if __name__ == "__main__":
    main()
