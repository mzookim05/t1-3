import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Reviewer 회신에 따라 metadata remap hotfix를 먼저 적용한 뒤,
# 실패한 `decision_micro_005` A-slot만 fresh replacement seed로 복구한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_micro_pilot as micro,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_micro_retry as micro_retry,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as pb8  # noqa: E402
from scripts.aihub.problem_generation.v2_objective_difficulty_patch import (  # noqa: E402
    run_difficulty_patch as base,
)
from scripts.aihub.problem_generation.v2_objective_difficulty_patch_r2 import (  # noqa: E402
    run_difficulty_patch as r2,
)


VERSION_TAG = "decision_choice_validator_a_slot_replacement"
# llm_runs 폴더 정렬과 리뷰 추적을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-25_224627"
RUN_PURPOSE = "objective_r2_a_slot_replacement_micro_package"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = base.PROJECT_ROOT
SOURCE_RUN_DIR = micro_retry.RUN_DIR
SOURCE_PACKAGE_PATH = micro_retry.MERGED_SCORES_PATH
FAILED_SLOT_SEED_ID = "decision_micro_005"
FAILED_SLOT_FAMILY_ID = "결정례_QA::가정법_결정례_15417"
REPLACEMENT_SEED_ID = "decision_replacement_001"
REPLACEMENT_TARGET_LABEL = "A"

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
VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH = RUN_EXPORTS_DIR / f"validator_replacement_actions_{VERSION_TAG}.csv"
VALIDATOR_FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"validator_final_package_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
METADATA_REMAP_AUDIT_CSV_PATH = RUN_EXPORTS_DIR / f"metadata_remap_audit_{VERSION_TAG}.csv"
METADATA_REMAP_AUDIT_MD_PATH = RUN_EXPORTS_DIR / f"metadata_remap_audit_{VERSION_TAG}.md"

REFERENCE_PB8_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb8_decision_only_objective_current_r2"
    / "seed_registry.csv"
)
REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_weak_distractor_guardrail_pilot"
    / "seed_registry.csv"
)
REFERENCE_MICRO_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_choice_validator_micro_pilot"
    / "seed_registry.csv"
)
REFERENCE_MICRO_RETRY_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_choice_validator_micro_retry"
    / "seed_registry.csv"
)

SUCCESS_USABLE_MIN = 8
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}
VALIDATOR_SUMMARY = {}


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_csv_rows_if_exists(path):
    if not Path(path).exists():
        return []
    return read_csv_rows(path)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["seed_sample_id"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def all_fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def collect_excluded_rows():
    # current counted seed와 모든 failed/informative decision run seed를 fresh replacement 후보에서 제외한다.
    rows = pb8.collect_excluded_rows()
    for path in (
        REFERENCE_PB8_SEED_REGISTRY_PATH,
        REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH,
        REFERENCE_MICRO_SEED_REGISTRY_PATH,
        REFERENCE_MICRO_RETRY_SEED_REGISTRY_PATH,
    ):
        rows.extend(load_csv_rows_if_exists(path))
    rows.append({"family_id": FAILED_SLOT_FAMILY_ID, "seed_sample_id": FAILED_SLOT_SEED_ID})
    return rows


def build_exclusion_sets(rows):
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


def decision_generalization_specs():
    specs = []
    for spec in pb8.pb6.pb4.pb3.DATASET_SPECS:
        if spec["doc_type_name"] == "결정례_QA" and spec.get("sampling_lane") == "generalization_03_04":
            copied = dict(spec)
            copied["sample_count"] = 1
            specs.append(copied)
    return specs


def select_replacement_record():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)

    for spec in decision_generalization_specs():
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
            passes_filter, _ = pb8.passes_pb8_seed_filter(spec, payload)
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
                "sample_id": REPLACEMENT_SEED_ID,
                "sample_order": 1,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb8.pb6.pb4.pb3.explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": 1,
                "selected_index": candidate_index,
                "selection_note": "decision_micro_005 A-slot replacement seed: prior failed/timepoint weak-distractor seed excluded",
            }, exclusion_sets
    raise RuntimeError("A-slot replacement용 fresh 결정례_QA generalization seed를 찾지 못했습니다.")


def build_seed_row(record):
    row = pb8.pb6.pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_decision_choice_validator_a_slot_replacement_seed"
    row["selection_note"] = "decision_micro_005 실패 slot을 fresh 결정례_QA generalization seed로 대체하는 A-slot replacement seed"
    row["decision_validator_replacement_note"] = "failed_seed_decision_micro_005_family_15417_excluded"
    return row


def build_preflight_rows(seed_rows, exclusion_sets):
    rows = []
    for row in seed_rows:
        rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "reference_sample_id": row["reference_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "seed_sample_id_overlap_with_prior": "예" if row["seed_sample_id"] in exclusion_sets["sample_ids"] else "아니오",
                "reference_sample_id_overlap_with_prior": "예" if row["reference_sample_id"] in exclusion_sets["reference_sample_ids"] else "아니오",
                "family_overlap_with_prior": "예" if row["family_id"] in exclusion_sets["family_ids"] else "아니오",
                "label_path_overlap_with_prior": "예" if row["label_path"] in exclusion_sets["label_paths"] else "아니오",
                "raw_path_overlap_with_prior": "예" if row["raw_path"] in exclusion_sets["raw_paths"] else "아니오",
                "failed_slot_seed_excluded": "예",
                "target_correct_choice": REPLACEMENT_TARGET_LABEL,
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    return rows


def assert_preflight(seed_rows, preflight_rows):
    if len(seed_rows) != 1:
        raise RuntimeError(f"replacement seed는 1개여야 합니다: {len(seed_rows)}")
    row = seed_rows[0]
    if row["doc_type_name"] != "결정례_QA":
        raise RuntimeError("replacement seed는 결정례_QA여야 합니다.")
    if row["sampling_lane"] != "generalization_03_04":
        raise RuntimeError("replacement seed는 generalization_03_04 lane이어야 합니다.")
    for preflight in preflight_rows:
        overlap_values = [
            preflight["seed_sample_id_overlap_with_prior"],
            preflight["reference_sample_id_overlap_with_prior"],
            preflight["family_overlap_with_prior"],
            preflight["label_path_overlap_with_prior"],
            preflight["raw_path_overlap_with_prior"],
        ]
        if "예" in overlap_values:
            raise RuntimeError(f"replacement preflight overlap failed: {preflight['seed_sample_id']}")


def write_seed_inputs():
    record, exclusion_sets = select_replacement_record()
    seed_rows = [build_seed_row(record)]
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_preflight(seed_rows, preflight_rows)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RUN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_csv(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        "- seed_count: `1`",
        "- doc_type_counts: `{'결정례_QA': 1}`",
        "- lane_counts: `{'generalization_03_04': 1}`",
        f"- failed_slot_seed_excluded: `{FAILED_SLOT_SEED_ID}` / `{FAILED_SLOT_FAMILY_ID}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        "| replacement seed count is 1 | `pass` |",
        "| doc type is decision only | `pass` |",
        "| lane is generalization_03_04 | `pass` |",
        "| target label is A | `pass` |",
        "| no current/failed/micro seed overlap | `pass` |",
    ]
    write_text(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)
    return seed_rows


def write_replacement_plan(seed_rows):
    rows = [
        {
            "failed_seed_sample_id": FAILED_SLOT_SEED_ID,
            "replacement_seed_sample_id": seed_rows[0]["seed_sample_id"],
            "replacement_family_id": seed_rows[0]["family_id"],
            "doc_type_name": seed_rows[0]["doc_type_name"],
            "sampling_lane": seed_rows[0]["sampling_lane"],
            "target_correct_choice": REPLACEMENT_TARGET_LABEL,
            "policy": "seed_specific_exclusion_then_one_fresh_replacement",
        }
    ]
    write_csv(REPLACEMENT_PLAN_CSV_PATH, rows, list(rows[0].keys()))
    lines = [
        f"# replacement plan `{VERSION_TAG}`",
        "",
        "| failed_seed | replacement_seed | lane | target | policy |",
        "| --- | --- | --- | --- | --- |",
        f"| `{FAILED_SLOT_SEED_ID}` | `{seed_rows[0]['seed_sample_id']}` | `{seed_rows[0]['sampling_lane']}` | `{REPLACEMENT_TARGET_LABEL}` | `seed_specific_exclusion_then_one_fresh_replacement` |",
    ]
    write_text(REPLACEMENT_PLAN_MD_PATH, "\n".join(lines) + "\n")
    return rows


def configure_base_for_replacement():
    # r2의 generator/Judge 본체는 유지하고, run identity와 path만 replacement package로 재배선한다.
    r2.configure_base()
    base.VERSION_TAG = VERSION_TAG
    base.RUN_DATE = RUN_DATE
    base.RUN_PURPOSE = RUN_PURPOSE
    base.RUN_NAME = RUN_NAME
    base.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    base.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    base.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    base.RUN_MERGED_DIR = RUN_MERGED_DIR
    base.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    base.SEED_READY_PATH = SEED_READY_PATH
    base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    base.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    base.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    base.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    base.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    base.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    base.MERGED_SCORES_PATH = REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH
    base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Keyedness": KEYEDNESS_LOG_PATH,
        "DistractorFit": DISTRACTORFIT_LOG_PATH,
        "NearMiss": NEARMISS_LOG_PATH,
    }
    base.build_generation_messages = build_replacement_generation_messages
    base.postprocess_problem = r2.postprocess_problem
    base.build_local_fallback_problem = r2.build_local_fallback_problem
    base.load_reference_v2_rows = lambda: {}


def build_replacement_generation_messages(seed, reference_v2):
    messages = micro.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## A-slot replacement 추가 지시
- 이 seed는 `decision_micro_005`가 seed당 최대 1회 retry 후에도 weak distractor로 남아 생긴 `A` label 결손을 복구하기 위한 replacement seed다.
- 기존 실패 seed의 `family_id`는 `{FAILED_SLOT_FAMILY_ID}`이며, 이번 seed는 동일 family를 쓰면 안 된다.
- 생성 단계의 choice label은 후처리 validator가 다시 맞출 수 있으므로, 의미상 정답 유일성과 오답 3개의 균등한 near-miss plausibility를 최우선으로 한다.
- 시간·시점형 표현만으로 오답을 구성하지 말고, 판단 기준, 적용 사실, 결론 범위 중 서로 다른 한 축을 비틀어 오답을 분리한다.
- 최종 package에서는 target correct_choice `{REPLACEMENT_TARGET_LABEL}` slot으로 검산될 예정이다.
"""
    return messages


def run_replacement_generation_and_judge():
    configure_base_for_replacement()
    base.ensure_run_dirs()
    base.run_generation(mode="main")
    base.run_generation(mode="strict_finalize")
    base.run_judges(mode="main")
    base.run_judges(mode="strict_finalize")
    return base.merge_scores()


def append_tag(existing_tags, tag):
    tags = [value for value in (existing_tags or "").split("|") if value]
    if tag not in tags:
        tags.append(tag)
    return "|".join(tags)


def selected_rows(rows):
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def preserved_export_rows():
    rows = read_csv_rows(SOURCE_PACKAGE_PATH)
    preserved = []
    for row in rows:
        if row.get("seed_sample_id") == FAILED_SLOT_SEED_ID:
            continue
        if row.get("selected_for_seed") == "예" and row.get("validator_export_disposition") == "export_ready":
            copied = dict(row)
            validator_replay.remap_existing_metadata_to_correct_choice(copied)
            metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(copied)
            copied["package_run_name"] = RUN_NAME
            copied["replacement_package_role"] = "preserved_export_ready"
            copied["source_run_name"] = SOURCE_RUN_DIR.name
            copied["validator_metadata_remap_ok"] = "예" if metadata_ok else "아니오"
            copied["validator_metadata_remap_reasons"] = "|".join(metadata_reasons)
            preserved.append(copied)
    return preserved


def apply_validator_to_replacement_rows(rows):
    report_rows = []
    for row in rows:
        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
        row["upstream_train_eligible"] = row.get("train_eligible", "")
        row["validator_action"] = ""
        row["validator_status"] = ""
        row["validator_reasons"] = ""
        row["validator_target_correct_choice"] = ""
        row["validator_recalculated_correct_choice"] = ""
        row["validator_correct_choice_match_count"] = ""
        row["validator_shuffle_recalc_ok"] = ""
        row["validator_metadata_remap_ok"] = ""
        row["validator_metadata_remap_reasons"] = ""
        row["validator_export_disposition"] = ""

    for row in selected_rows(rows):
        target_label = REPLACEMENT_TARGET_LABEL
        choices = validator_replay.choice_map(row)
        original_correct_choice = row.get("correct_choice", "")
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices, original_correct_choice, target_label
        )
        action, status, reasons = validator_replay.choose_validator_action(row)
        shuffle_ok = recalculated_label == target_label

        if not shuffle_ok:
            action = "hard_block"
            status = "correct_choice_recalc_block"
            reasons = [*reasons, "correct_choice_recalc_mismatch"]

        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["validator_target_correct_choice"] = target_label
        row["validator_recalculated_correct_choice"] = recalculated_label or ""
        row["validator_correct_choice_match_count"] = str(match_count)
        row["validator_shuffle_recalc_ok"] = "예" if shuffle_ok else "아니오"
        row["package_run_name"] = RUN_NAME
        row["replacement_package_role"] = "a_slot_replacement_candidate"
        row["source_run_name"] = ""

        if action in {"accept", "audit"}:
            # replacement row도 choice와 label-keyed metadata를 한 번에 이동해야 reviewer artifact가 맞는다.
            row["choice_a"] = shuffled_choices["A"]
            row["choice_b"] = shuffled_choices["B"]
            row["choice_c"] = shuffled_choices["C"]
            row["choice_d"] = shuffled_choices["D"]
            validator_replay.remap_label_keyed_metadata(row, original_correct_choice, target_label)
            row["correct_choice"] = recalculated_label or row.get("correct_choice", "")
            metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(row)
            row["validator_metadata_remap_ok"] = "예" if metadata_ok else "아니오"
            row["validator_metadata_remap_reasons"] = "|".join(metadata_reasons)
            if not metadata_ok:
                action = "hard_block"
                status = "metadata_remap_block"
                reasons = [*reasons, *metadata_reasons]
                row["validator_action"] = action
                row["validator_status"] = status
                row["validator_reasons"] = "|".join(reasons)
        else:
            row["validator_metadata_remap_ok"] = "대상아님"
            row["validator_metadata_remap_reasons"] = ""

        if action == "accept":
            row["final_status"] = "pass"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "예"
            row["validator_export_disposition"] = "export_ready"
        elif action == "audit":
            row["final_status"] = "pass"
            row["audit_required"] = "예"
            row["audit_reason"] = append_tag(row.get("audit_reason", ""), "validator_audit")
            row["train_eligible"] = "아니오"
            row["validator_export_disposition"] = "audit"
        elif action == "regenerate":
            row["final_status"] = "soft_fail"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "아니오"
            row["error_tags"] = append_tag(row.get("error_tags", ""), "validator_regenerate")
            row["validator_export_disposition"] = "regenerate_excluded"
        else:
            row["final_status"] = "hard_fail"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "아니오"
            row["error_tags"] = append_tag(row.get("error_tags", ""), "validator_hard_block")
            row["validator_export_disposition"] = "hard_block_excluded"

        report_rows.append(
            {
                "report_section": "replacement_action",
                "seed_sample_id": row["seed_sample_id"],
                "upstream_final_status": row["upstream_final_status"],
                "upstream_audit_required": row["upstream_audit_required"],
                "upstream_train_eligible": row["upstream_train_eligible"],
                "validator_action": row["validator_action"],
                "validator_status": row["validator_status"],
                "validator_reasons": row["validator_reasons"],
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "shuffle_recalc_ok": row["validator_shuffle_recalc_ok"],
                "metadata_remap_ok": row["validator_metadata_remap_ok"],
                "metadata_remap_reasons": row["validator_metadata_remap_reasons"],
                "final_status": row["final_status"],
                "audit_required": row["audit_required"],
                "train_eligible": row["train_eligible"],
                "export_disposition": row["validator_export_disposition"],
            }
        )
    return rows, report_rows


def build_package(validated_replacement_rows):
    rows = preserved_export_rows() + selected_rows(validated_replacement_rows)
    rows.sort(key=lambda row: row["seed_sample_id"])
    for row in rows:
        row["selected_for_seed"] = "예"
    return rows


def compute_validator_summary(package):
    global VALIDATOR_SUMMARY
    target_counts = Counter(row.get("validator_target_correct_choice", "") for row in package)
    export_rows = [row for row in package if row.get("validator_export_disposition") in {"export_ready", "audit"}]
    export_label_counts = Counter(row.get("correct_choice", "") for row in export_rows)
    action_counts = Counter(row.get("validator_action", "") for row in package)
    metadata_remap_mismatch_count = sum(
        1
        for row in export_rows
        if row.get("validator_metadata_remap_ok") != "예"
    )
    VALIDATOR_SUMMARY = {
        "selected_count": len(package),
        "validator_action_counts": dict(action_counts),
        "target_label_counts": {label: target_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_label_counts": {label: export_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "shuffle_recalc_mismatch_count": sum(1 for row in package if row.get("validator_shuffle_recalc_ok") != "예"),
        "metadata_remap_mismatch_count": metadata_remap_mismatch_count,
        "selected_train_eligible_count": sum(1 for row in package if row.get("train_eligible") == "예"),
        "selected_hard_fail_count": sum(1 for row in package if row.get("final_status") == "hard_fail"),
        "selected_soft_fail_count": sum(1 for row in package if row.get("final_status") == "soft_fail"),
        "selected_audit_required_count": sum(1 for row in package if row.get("audit_required") == "예"),
        "law_row_count": sum(1 for row in package if row.get("doc_type_name") == "법령_QA"),
    }
    VALIDATOR_SUMMARY["export_ready_label_balance_passed"] = all(
        VALIDATOR_SUMMARY["export_label_counts"][label] == TARGET_LABEL_COUNTS[label]
        for label in validator_replay.CHOICE_LABELS
    )
    VALIDATOR_SUMMARY["replacement_package_success_passed"] = (
        VALIDATOR_SUMMARY["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and VALIDATOR_SUMMARY["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and VALIDATOR_SUMMARY["shuffle_recalc_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["metadata_remap_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["export_ready_label_balance_passed"]
        and VALIDATOR_SUMMARY["law_row_count"] == SUCCESS_LAW_ROW_COUNT
    )
    return VALIDATOR_SUMMARY


def write_metadata_remap_audit(package):
    rows = []
    for row in package:
        if row.get("validator_export_disposition") not in {"export_ready", "audit"}:
            continue
        metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(row)
        rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "correct_choice": row.get("correct_choice", ""),
                "distractor_type_map_correct": validator_replay.parse_label_json_map(row.get("distractor_type_map", "")).get(row.get("correct_choice", ""), ""),
                "near_miss_has_correct_choice": "예"
                if row.get("correct_choice", "") in validator_replay.parse_label_json_map(row.get("near_miss_notes", ""))
                else "아니오",
                "metadata_remap_ok": "예" if metadata_ok else "아니오",
                "metadata_remap_reasons": "|".join(metadata_reasons),
            }
        )
    write_csv(METADATA_REMAP_AUDIT_CSV_PATH, rows, list(rows[0].keys()) if rows else None)
    lines = [
        f"# metadata remap audit `{VERSION_TAG}`",
        "",
        "| seed | correct | map_correct | note_has_correct | metadata_ok | reasons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['correct_choice']}` | `{row['distractor_type_map_correct']}` | `{row['near_miss_has_correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['metadata_remap_reasons']}` |"
        )
    write_text(METADATA_REMAP_AUDIT_MD_PATH, "\n".join(lines) + "\n")
    return rows


def write_validator_report(replacement_report_rows, package):
    summary = compute_validator_summary(package)
    package_rows_for_report = []
    for row in package:
        package_rows_for_report.append(
            {
                "report_section": "final_package",
                "seed_sample_id": row["seed_sample_id"],
                "replacement_package_role": row.get("replacement_package_role", ""),
                "validator_action": row.get("validator_action", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "train_eligible": row.get("train_eligible", ""),
                "target_correct_choice": row.get("validator_target_correct_choice", ""),
                "correct_choice": row.get("correct_choice", ""),
                "metadata_remap_ok": row.get("validator_metadata_remap_ok", ""),
                "metadata_remap_reasons": row.get("validator_metadata_remap_reasons", ""),
                "export_disposition": row.get("validator_export_disposition", ""),
            }
        )
    write_csv(VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH, replacement_report_rows)
    write_csv(VALIDATOR_FINAL_PACKAGE_CSV_PATH, package_rows_for_report)
    write_csv(VALIDATOR_REPORT_CSV_PATH, replacement_report_rows + package_rows_for_report, all_fieldnames(replacement_report_rows + package_rows_for_report))

    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{summary['selected_count']}`",
        f"- validator_action_counts: `{summary['validator_action_counts']}`",
        f"- target_label_counts: `{summary['target_label_counts']}`",
        f"- export_label_counts: `{summary['export_label_counts']}`",
        f"- shuffle_recalc_mismatch_count: `{summary['shuffle_recalc_mismatch_count']}`",
        f"- metadata_remap_mismatch_count: `{summary['metadata_remap_mismatch_count']}`",
        f"- replacement_package_success_passed: `{summary['replacement_package_success_passed']}`",
        "",
        "## replacement action",
        "| seed | upstream_status | action | final_status | train_eligible | target | recalculated | metadata | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in replacement_report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['upstream_final_status']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['recalculated_correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    lines.extend(
        [
            "",
            "## final 8-slot package",
            "| seed | role | action | final_status | train_eligible | target | correct | metadata | disposition |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in package_rows_for_report:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['replacement_package_role']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    write_text(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_wiring_check():
    lines = [
        f"# A-slot replacement wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| metadata remap hotfix applied | `pass` | 보존 row와 replacement row 모두 post-shuffle metadata gate 적용 |",
        "| failed seed excluded | `pass` | `decision_micro_005`, `family_id 15417` 제외 |",
        "| replacement seed scope | `pass` | `결정례_QA`, `generalization_03_04`, fresh seed `1개` |",
        "| replacement target label | `pass` | `A` slot 복구 목적 |",
        "| validator report split | `pass` | replacement action CSV와 final package CSV 분리 |",
        "| count reflection | `pass` | reviewer sign-off 전 current count 미합산 |",
    ]
    write_text(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def write_batch_summary(package):
    summary = compute_validator_summary(package)
    doc_counter = Counter(row["doc_type_name"] for row in package)
    lane_counter = Counter(row["sampling_lane"] for row in package)
    status_counter = Counter(row["final_status"] for row in package)
    train_counter = Counter(row["train_eligible"] for row in package)

    summary_rows = [
        {"metric": "seed_count", "value": str(len(package))},
        {"metric": "selected_pass", "value": str(status_counter.get("pass", 0))},
        {"metric": "selected_hard_fail", "value": str(status_counter.get("hard_fail", 0))},
        {"metric": "selected_soft_fail", "value": str(status_counter.get("soft_fail", 0))},
        {"metric": "train_eligible", "value": str(train_counter.get("예", 0))},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "success_passed", "value": str(summary["replacement_package_success_passed"])},
    ]
    write_csv(BATCH_SUMMARY_CSV_PATH, summary_rows)
    write_csv(BATCH_LANE_SUMMARY_CSV_PATH, [{"sampling_lane": lane, "count": count} for lane, count in sorted(lane_counter.items())], ["sampling_lane", "count"])

    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(package)}`",
        f"- doc_type_counts: `{dict(doc_counter)}`",
        f"- lane_counts: `{dict(lane_counter)}`",
        f"- selected: `{status_counter.get('pass', 0)} pass / {status_counter.get('hard_fail', 0)} hard_fail / {status_counter.get('soft_fail', 0)} soft_fail`",
        f"- train/audit: `train_eligible {train_counter.get('예', 0)} / audit_required {summary['selected_audit_required_count']}`",
        "",
        "## success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| usable | `>= {SUCCESS_USABLE_MIN} / 8` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| metadata remap mismatch | `0` | `{summary['metadata_remap_mismatch_count']}` |",
        f"| export label balance | `A/B/C/D = 2/2/2/2` | `{summary['export_label_counts']}` |",
    ]
    write_text(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def classify_tail(row):
    tags = row.get("error_tags", "")
    if "정답 비유일" in tags or "오답이 정답 가능" in tags:
        return "decision answer uniqueness failure"
    if "validator_regenerate" in tags or "오답약함" in tags or "near_miss_부족" in tags:
        return "decision weak distractor"
    if row.get("final_status") == "hard_fail":
        return "decision hard fail"
    if row.get("final_status") == "soft_fail":
        return "decision soft fail"
    if row.get("audit_required") == "예":
        return "decision audit tail"
    return "tail 없음"


def write_tail_memo(package):
    tail_rows = []
    for row in package:
        if row.get("train_eligible") == "예":
            continue
        tail_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row.get("doc_type_name", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "validator_action": row.get("validator_action", ""),
                "tail_class": classify_tail(row),
            }
        )
    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "doc_type_name": "",
                "sampling_lane": "",
                "final_status": "",
                "audit_required": "",
                "error_tags": "",
                "validator_action": "",
                "tail_class": "tail 없음",
            }
        ]
    write_csv(TAIL_MEMO_CSV_PATH, tail_rows)
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | doc_type | lane | status | audit | validator_action | error_tags | tail_class |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['doc_type_name']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['validator_action']}` | `{row['error_tags']}` | `{row['tail_class']}` |"
        )
    write_text(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def split_package(package):
    old_paths = (
        base.PROBLEM_TRAIN_PATH,
        base.PROBLEM_DEV_PATH,
        base.PROBLEM_TEST_PATH,
        base.PROBLEM_DATASET_MANIFEST_PATH,
        base.PROBLEM_AUDIT_QUEUE_PATH,
    )
    base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    try:
        return base.split_dataset(package)
    finally:
        (
            base.PROBLEM_TRAIN_PATH,
            base.PROBLEM_DEV_PATH,
            base.PROBLEM_TEST_PATH,
            base.PROBLEM_DATASET_MANIFEST_PATH,
            base.PROBLEM_AUDIT_QUEUE_PATH,
        ) = old_paths


def write_manifest(seed_rows, package, manifest_rows, summary_rows, tail_rows, replacement_plan_rows, metadata_audit_rows):
    summary = compute_validator_summary(package)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "source_micro_retry_run_name": SOURCE_RUN_DIR.name,
        "replacement_policy": {
            "failed_seed_sample_id": FAILED_SLOT_SEED_ID,
            "failed_family_id": FAILED_SLOT_FAMILY_ID,
            "replacement_seed_ids": [row["seed_sample_id"] for row in seed_rows],
            "replacement_target_label": REPLACEMENT_TARGET_LABEL,
            "preserve_export_ready_rows": True,
            "metadata_remap_hotfix": True,
            "semantic_judge": "replacement row only",
            "count_reflection": "not_counted_until_reviewer_signoff",
        },
        "seed_registry_count": len(seed_rows),
        "generation_count": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": base.load_jsonl_count(NEARMISS_LOG_PATH),
        "merged_count": len(package),
        "selected_pass_count": summary["selected_count"] - summary["selected_hard_fail_count"] - summary["selected_soft_fail_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "validator_summary": summary,
        "success_criteria": {
            "usable_min": SUCCESS_USABLE_MIN,
            "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
            "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
            "audit_max": SUCCESS_AUDIT_MAX,
            "export_label_balance": TARGET_LABEL_COUNTS,
            "law_row_count": SUCCESS_LAW_ROW_COUNT,
            "metadata_remap_mismatch_count": 0,
        },
        "success_result": {
            "usable": summary["selected_train_eligible_count"],
            "hard_fail": summary["selected_hard_fail_count"],
            "soft_fail": summary["selected_soft_fail_count"],
            "audit": summary["selected_audit_required_count"],
            "export_label_counts": summary["export_label_counts"],
            "law_row_count": summary["law_row_count"],
            "metadata_remap_mismatch_count": summary["metadata_remap_mismatch_count"],
            "passed": summary["replacement_package_success_passed"],
        },
        "current_count_decision": "not_counted_until_reviewer_signoff",
        "replacement_plan_count": len(replacement_plan_rows),
        "metadata_audit_count": len(metadata_audit_rows),
        "tail_memo_count": len([row for row in tail_rows if row.get("seed_sample_id")]),
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight_csv": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
            "replacement_plan_csv": str(REPLACEMENT_PLAN_CSV_PATH),
            "replacement_plan_md": str(REPLACEMENT_PLAN_MD_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "replacement_merged_before_validator": str(REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_replacement_actions_csv": str(VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH),
            "validator_final_package_csv": str(VALIDATOR_FINAL_PACKAGE_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
            "metadata_remap_audit_csv": str(METADATA_REMAP_AUDIT_CSV_PATH),
            "metadata_remap_audit_md": str(METADATA_REMAP_AUDIT_MD_PATH),
            "batch_summary_md": str(BATCH_SUMMARY_MD_PATH),
            "batch_summary_csv": str(BATCH_SUMMARY_CSV_PATH),
            "tail_memo_csv": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md": str(TAIL_MEMO_MD_PATH),
            "problem_train": str(PROBLEM_TRAIN_PATH),
            "problem_dev": str(PROBLEM_DEV_PATH),
            "problem_test": str(PROBLEM_TEST_PATH),
            "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)
    return manifest


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows = write_seed_inputs()
    replacement_plan_rows = write_replacement_plan(seed_rows)
    write_wiring_check()
    replacement_merged = run_replacement_generation_and_judge()
    validated_replacement, replacement_report_rows = apply_validator_to_replacement_rows(replacement_merged)
    package = build_package(validated_replacement)
    write_csv(MERGED_SCORES_PATH, package, all_fieldnames(package))
    write_validator_report(replacement_report_rows, package)
    metadata_audit_rows = write_metadata_remap_audit(package)
    summary_rows = write_batch_summary(package)
    tail_rows = write_tail_memo(package)
    manifest_rows = split_package(package)
    return write_manifest(seed_rows, package, manifest_rows, summary_rows, tail_rows, replacement_plan_rows, metadata_audit_rows)


if __name__ == "__main__":
    main()
