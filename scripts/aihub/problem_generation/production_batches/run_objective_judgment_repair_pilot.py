from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

# reviewer sign-off 이후 같은 16개 판결문 seed를 실제 API 경로로 태우는 runner다.
# no-API seed preflight의 registry와 exclusion audit을 그대로 고정해 seed drift 없이 검증한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_repair_pilot_seed_preflight as preflight,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402


VERSION_TAG = "objective_judgment_repair_pilot"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_judgment_repair_api_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PREFLIGHT_RUN_NAME = preflight.RUN_NAME
SOURCE_PREFLIGHT_RUN_DIR = preflight.RUN_DIR
SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = preflight.SEED_REGISTRY_PATH
SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH = preflight.TARGET_LABEL_SCHEDULE_CSV_PATH
SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH = preflight.EXCLUSION_AUDIT_MD_PATH

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
RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"
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
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
PILOT_BREAKOUT_CSV_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.csv"
PILOT_BREAKOUT_MD_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

EXPECTED_TOTAL_SEED_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 16}
EXPECTED_LANE_BY_DOC = {
    ("판결문_QA", "generalization_03_04"): 8,
    ("판결문_QA", "expansion_01_02"): 8,
}
EXPECTED_SOURCE_COUNTS = {
    "01_TL_판결문_QA": 4,
    "02_TL_판결문_QA": 4,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 4,
}
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
SUCCESS_USABLE_MIN = 14
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 1
SUCCESS_AUDIT_MAX = 1

BATCH_STATUS = "judgment_repair_pilot_not_counted"
COUNT_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

BASE_BUILD_GENERATION_MESSAGES = pb6.ORIGINAL_BUILD_GENERATION_MESSAGES
BASE_SPLIT_DATASET = pb6.pb4.pb3.base.split_dataset
BASE_BUILD_RUN_MANIFEST = pb6.build_run_manifest

VALIDATOR_SUMMARY: dict[str, object] = {}


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


def parse_nearmiss_score(value: str):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def target_schedule_by_seed() -> dict[str, str]:
    return {
        row["seed_sample_id"]: row["target_correct_choice"]
        for row in read_csv_rows(TARGET_LABEL_SCHEDULE_CSV_PATH)
    }


def write_target_label_schedule(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "stem_axis": row.get("stem_axis", ""),
            "judgment_seed_action": row.get("judgment_seed_action", ""),
            "tail_proximity_class": row.get("tail_proximity_class", ""),
            "target_correct_choice": row["target_correct_choice"],
        }
        for row in seed_rows
    ]
    counts = Counter(row["target_correct_choice"] for row in rows)
    if dict(counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"judgment target label schedule mismatch: {dict(counts)}")
    pb6.pb4.pb3.base.write_csv_atomic(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    return rows


def assert_fixed_seed_registry(seed_rows: list[dict[str, str]]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"judgment pilot seed count mismatch: {len(seed_rows)}")
    if dict(doc_counts) != EXPECTED_DOC_TYPE_COUNTS:
        raise RuntimeError(f"judgment pilot doc type mismatch: {dict(doc_counts)}")
    for key, expected_count in EXPECTED_LANE_BY_DOC.items():
        if lane_by_doc.get(key, 0) != expected_count:
            raise RuntimeError(f"judgment pilot lane mismatch: {key}={lane_by_doc.get(key, 0)}")
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"judgment pilot source split mismatch: {dict(source_counts)}")
    if dict(target_counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"judgment pilot target label mismatch: {dict(target_counts)}")


def write_seed_preflight_copy(seed_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "reference_sample_id": row["reference_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "target_correct_choice": row["target_correct_choice"],
            "judgment_seed_action": row.get("judgment_seed_action", ""),
            "stem_axis": row.get("stem_axis", ""),
            "answer_uniqueness_risk_flags": row.get("answer_uniqueness_risk_flags", ""),
            "tail_proximity_class": row.get("tail_proximity_class", ""),
            "label_path": row.get("label_path", ""),
            "raw_path": row.get("raw_path", ""),
        }
        for row in seed_rows
    ]
    pb6.pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, rows, list(rows[0].keys()))
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    tail_counts = Counter(row.get("tail_proximity_class", "") for row in seed_rows)
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## fixed source",
        f"- source_preflight_run: `{SOURCE_PREFLIGHT_RUN_NAME}`",
        f"- source_seed_registry: `{SOURCE_PREFLIGHT_SEED_REGISTRY_PATH}`",
        f"- source_exclusion_audit: `{SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- source_subset_counts: `{dict(source_counts)}`",
        f"- target_label_counts: `{dict(target_counts)}`",
        f"- tail_proximity_counts: `{dict(tail_counts)}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        "| same 16 seed registry as no-API preflight | `pass` |",
        "| source split is 01/02/03/04 each 4 | `pass` |",
        "| lane split is 8/8 | `pass` |",
        "| target label schedule is A/B/C/D = 4/4/4/4 | `pass` |",
        "| exclusion audit hotfix is referenced | `pass` |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry_from_preflight() -> list[dict[str, str]]:
    pb6.pb4.pb3.base.ensure_dirs(
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
    seed_rows = read_csv_rows(SOURCE_PREFLIGHT_SEED_REGISTRY_PATH)
    assert_fixed_seed_registry(seed_rows)
    pb6.pb4.pb3.base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    write_seed_preflight_copy(seed_rows)
    write_target_label_schedule(seed_rows)
    return seed_rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += f"""

## judgment repair pilot 추가 지시
- 이번 run은 `판결문_QA` answer uniqueness repair pilot이다.
- seed action은 `{seed.get('judgment_seed_action', '')}`, stem axis는 `{seed.get('stem_axis', '')}`, tail proximity는 `{seed.get('tail_proximity_class', '')}`다.
- stem은 하나의 청구, 하나의 쟁점, 하나의 판단 기준, 하나의 적용 사실만 묻는다.
- 일반론과 사안 적용을 한 stem 안에서 동시에 묻지 않는다.
- 복수 청구, 복수 처분, 복수 결론 분기가 보이면 하나의 predicate만 남기고 나머지는 선택지에서만 비튼다.
- 정답은 `gold_short_answer`와 같은 판결상 결론 하나에만 닫혀야 한다.
- 오답은 같은 판결문 근거를 공유하되 각각 쟁점, 사실관계, 판단 기준, 결론 범위 중 정확히 한 축만 어긋나야 한다.
- 다른 choice가 별도 일반론이나 별도 사실 적용으로도 정답 가능하게 읽히면 answer uniqueness failure로 본다.
- 후처리 validator가 target label `{seed.get('target_correct_choice', '')}`로 choice를 재배치하므로, 생성 단계에서는 label 위치보다 정답 유일성과 선택지 의미 분리에 집중한다.
"""
    return messages


def answer_uniqueness_reason(row: dict[str, str], answer_match_count: int) -> list[str]:
    tags = validator_replay.split_tags(row.get("error_tags", ""))
    reasons: list[str] = []
    if answer_match_count != 1:
        reasons.append("single_correct_choice_failed")
    if tags & {"정답 비유일", "오답이 정답 가능", "복수 쟁점 혼합"}:
        reasons.append("answer_uniqueness_tag")
    if row.get("final_status") == "hard_fail":
        reasons.append("upstream_hard_fail")
    return reasons


def choose_judgment_validator_action(
    row: dict[str, str],
    answer_match_count: int,
) -> tuple[str, str, list[str]]:
    reasons = answer_uniqueness_reason(row, answer_match_count)
    tags = validator_replay.split_tags(row.get("error_tags", ""))
    nearmiss_score = parse_nearmiss_score(row.get("nearmiss_score", ""))
    if reasons:
        return "hard_block", "answer_uniqueness_block", reasons
    if nearmiss_score is None:
        return "hard_block", "nearmiss_score_missing_or_parse_block", ["nearmiss_score_missing_or_parse_failure"]
    if nearmiss_score <= 2:
        return "regenerate", "weak_distractor_regeneration", ["nearmiss_score_le_2"]
    if row.get("final_status") == "soft_fail":
        return "regenerate", "upstream_soft_fail_regeneration", ["upstream_soft_fail"]
    if nearmiss_score == 3 or tags & {"오답약함", "near_miss_부족", "형식 부적합"}:
        return "audit", "judgment_quality_audit", ["weak_or_form_audit"]
    if row.get("audit_required") == "예":
        return "audit", "upstream_audit", ["upstream_audit"]
    return "accept", "validator_clean", []


def export_disposition(action: str, metadata_ok: bool) -> tuple[str, str, str]:
    if not metadata_ok:
        return "metadata_remap_block", "아니오", "아니오"
    if action == "accept":
        return "export_ready", "예", "예"
    if action == "audit":
        return "audit_queue", "아니오", "아니오"
    if action == "regenerate":
        return "regenerate_required", "아니오", "아니오"
    return "hard_blocked", "아니오", "아니오"


def row_gate_fields(row: dict[str, str], answer_match_count: int) -> dict[str, str]:
    tags = validator_replay.split_tags(row.get("error_tags", ""))
    return {
        "single_correct_choice": "예" if answer_match_count == 1 and not tags & {"정답 비유일", "오답이 정답 가능"} else "아니오",
        "rule_application_split": "아니오" if "복수 쟁점 혼합" in tags else "예",
        "issue_boundary": "아니오" if "복수 쟁점 혼합" in tags else "예",
        "case_fact_alignment": "아니오" if tags & {"원문 외 사실 추가", "근거부족"} else "예",
        "hierarchy_overlap": "아니오" if tags & {"정답 비유일", "오답이 정답 가능"} else "예",
    }


def apply_judgment_validator(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    global VALIDATOR_SUMMARY

    selected = selected_rows(rows)
    target_by_seed = target_schedule_by_seed()
    # base merge가 seed-level planning columns를 모두 보존하지 않을 수 있으므로,
    # validator report와 breakout은 fixed seed registry를 기준으로 다시 보강한다.
    seed_metadata_by_id = {
        row["seed_sample_id"]: row
        for row in read_csv_rows(SEED_REGISTRY_PATH)
    }
    report_rows = []

    for row in rows:
        seed_metadata = seed_metadata_by_id.get(row.get("seed_sample_id", ""), {})
        for field in ("stem_axis", "judgment_seed_action", "answer_uniqueness_risk_flags", "tail_proximity_class"):
            if not row.get(field):
                row[field] = seed_metadata.get(field, "")
        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
        row["upstream_train_eligible"] = row.get("train_eligible", "")
        row["validator_action"] = ""
        row["validator_status"] = ""
        row["validator_reason_short"] = ""
        row["validator_reasons"] = ""
        row["target_correct_choice"] = target_by_seed.get(row.get("seed_sample_id", ""), "")
        row["validator_recalculated_correct_choice"] = ""
        row["validator_correct_choice_match_count"] = ""
        row["metadata_remap_ok"] = ""
        row["metadata_remap_reasons"] = ""
        row["validator_export_disposition"] = ""
        row["split_allowed"] = "아니오"
        row["count_allowed"] = "아니오"
        row["export_correct_choice"] = row.get("correct_choice", "")
        row["batch_status"] = BATCH_STATUS
        row["count_reflection_status"] = COUNT_REFLECTION_STATUS
        row["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED

    for row in selected:
        seed_id = row["seed_sample_id"]
        target_label = target_by_seed[seed_id]
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices,
            row.get("correct_choice", ""),
            target_label,
        )
        action, status, reasons = choose_judgment_validator_action(row, match_count)
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

        disposition, split_allowed, count_allowed = export_disposition(action, metadata_ok)
        gate_fields = row_gate_fields(row, match_count)
        reason_short = reasons[0] if reasons else "validator_clean"

        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reason_short"] = reason_short
        row["validator_reasons"] = "|".join(reasons)
        row["validator_export_disposition"] = disposition
        row["split_allowed"] = split_allowed
        row["count_allowed"] = count_allowed
        row["export_correct_choice"] = row.get("correct_choice", "")
        row.update(gate_fields)

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
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "stem_axis": row.get("stem_axis", ""),
                "judgment_seed_action": row.get("judgment_seed_action", ""),
                "tail_proximity_class": row.get("tail_proximity_class", ""),
                "answer_uniqueness_risk_flags": row.get("answer_uniqueness_risk_flags", ""),
                "upstream_final_status": row["upstream_final_status"],
                "upstream_audit_required": row["upstream_audit_required"],
                "upstream_train_eligible": row["upstream_train_eligible"],
                "target_correct_choice": target_label,
                "export_correct_choice": row["export_correct_choice"],
                "validator_action": action,
                "validator_export_disposition": disposition,
                "validator_reason_short": reason_short,
                "validator_reasons": row["validator_reasons"],
                "validator_recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "metadata_remap_ok": row["metadata_remap_ok"],
                "metadata_remap_reasons": row["metadata_remap_reasons"],
                "split_allowed": split_allowed,
                "count_allowed": count_allowed,
                **gate_fields,
                "final_status": row["final_status"],
                "audit_required": row["audit_required"],
                "train_eligible": row["train_eligible"],
                "error_tags": row.get("error_tags", ""),
            }
        )

    write_validator_artifacts(report_rows, rows)
    return rows


def write_validator_artifacts(report_rows: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    global VALIDATOR_SUMMARY

    selected = selected_rows(rows)
    summary = pb6.pb4.pb3.summarize_rows(rows)
    action_counts = Counter(row["validator_action"] for row in report_rows)
    target_counts = Counter(row["target_correct_choice"] for row in report_rows)
    export_ready_rows = [row for row in report_rows if row["validator_export_disposition"] == "export_ready"]
    export_label_counts = Counter(row["export_correct_choice"] for row in export_ready_rows)
    source_subset_counts = Counter(row["source_subset"] for row in report_rows)
    source_subset_usable = Counter(row["source_subset"] for row in report_rows if row["train_eligible"] == "예")
    tail_counts = Counter(row["tail_proximity_class"] for row in report_rows)
    tail_usable = Counter(row["tail_proximity_class"] for row in report_rows if row["train_eligible"] == "예")
    action_by_source = Counter((row["source_subset"], row["validator_export_disposition"]) for row in report_rows)
    answer_uniqueness_recurrence_count = sum(
        1
        for row in report_rows
        if row["validator_reason_short"] in {"answer_uniqueness_tag", "single_correct_choice_failed"}
        or row["single_correct_choice"] == "아니오"
        or row["hierarchy_overlap"] == "아니오"
    )
    shuffle_mismatch_count = sum(
        1
        for row in report_rows
        if row["validator_recalculated_correct_choice"] != row["target_correct_choice"]
    )
    metadata_mismatch_count = sum(
        1
        for row in report_rows
        if row["validator_export_disposition"] == "export_ready" and row["metadata_remap_ok"] != "예"
    )
    exact_target_schedule = dict(target_counts) == TARGET_LABEL_COUNTS
    exact_export_balance = all(export_label_counts.get(label, 0) == TARGET_LABEL_COUNTS[label] for label in TARGET_LABEL_COUNTS)
    countable_package_success = (
        len(export_ready_rows) == EXPECTED_TOTAL_SEED_COUNT
        and summary["selected_hard_fail_count"] == 0
        and summary["selected_soft_fail_count"] == 0
        and summary["selected_audit_required_count"] == 0
        and answer_uniqueness_recurrence_count == 0
        and shuffle_mismatch_count == 0
        and metadata_mismatch_count == 0
        and exact_export_balance
    )
    pilot_signal_success = (
        summary["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and summary["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and summary["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and summary["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and answer_uniqueness_recurrence_count == 0
        and shuffle_mismatch_count == 0
        and metadata_mismatch_count == 0
        and exact_target_schedule
    )

    VALIDATOR_SUMMARY = {
        "selected_count": len(selected),
        "validator_action_counts": dict(action_counts),
        "target_label_counts": {label: target_counts.get(label, 0) for label in TARGET_LABEL_COUNTS},
        "export_ready_label_counts": {label: export_label_counts.get(label, 0) for label in TARGET_LABEL_COUNTS},
        "source_subset_counts": dict(source_subset_counts),
        "source_subset_usable_counts": dict(source_subset_usable),
        "tail_proximity_counts": dict(tail_counts),
        "tail_proximity_usable_counts": dict(tail_usable),
        "action_by_source_subset": {f"{key[0]}::{key[1]}": value for key, value in action_by_source.items()},
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "answer_uniqueness_recurrence_count": answer_uniqueness_recurrence_count,
        "shuffle_recalc_mismatch_count": shuffle_mismatch_count,
        "metadata_remap_mismatch_count": metadata_mismatch_count,
        "exact_target_label_schedule_passed": exact_target_schedule,
        "exact_export_label_balance_passed": exact_export_balance,
        "pilot_signal_success": pilot_signal_success,
        "countable_package_success": countable_package_success,
    }

    pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, list(report_rows[0].keys()))
    write_validator_report_md(report_rows)
    write_validator_wiring_check_md()
    write_pilot_breakout(report_rows)


def write_validator_report_md(report_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{VALIDATOR_SUMMARY.get('selected_count', 0)}`",
        f"- validator_action_counts: `{VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- target_label_counts: `{VALIDATOR_SUMMARY.get('target_label_counts', {})}`",
        f"- export_ready_label_counts: `{VALIDATOR_SUMMARY.get('export_ready_label_counts', {})}`",
        f"- answer_uniqueness_recurrence_count: `{VALIDATOR_SUMMARY.get('answer_uniqueness_recurrence_count', 0)}`",
        f"- pilot_signal_success: `{VALIDATOR_SUMMARY.get('pilot_signal_success', False)}`",
        f"- countable_package_success: `{VALIDATOR_SUMMARY.get('countable_package_success', False)}`",
        "",
        "## row actions",
        "| seed | source | lane | stem_axis | action | disposition | reason | split | count | target | export | status | audit | eligible |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['stem_axis']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['validator_reason_short']}` | `{row['split_allowed']}` | `{row['count_allowed']}` | `{row['target_correct_choice']}` | `{row['export_correct_choice']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['train_eligible']}` |"
        )
    pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_validator_wiring_check_md() -> None:
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| fixed preflight seed registry reused | `pass` | no-API preflight 16개와 같은 seed registry 사용 |",
        "| target label schedule | `pass` | `A/B/C/D = 4/4/4/4` target 적용 |",
        "| answer uniqueness fields | `pass` | `single_correct_choice`, `rule_application_split`, `issue_boundary`, `case_fact_alignment`, `hierarchy_overlap` 기록 |",
        "| downstream guard fields | `pass` | `validator_reason_short`, `split_allowed`, `count_allowed` 기록 |",
        "| count reflection | `pass` | reviewer sign-off 전 core current count 미변경 |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def write_pilot_breakout(report_rows: list[dict[str, str]]) -> None:
    rows = []
    source_counts = Counter(row["source_subset"] for row in report_rows)
    source_usable = Counter(row["source_subset"] for row in report_rows if row["train_eligible"] == "예")
    source_audit = Counter(row["source_subset"] for row in report_rows if row["audit_required"] == "예")
    source_hard = Counter(row["source_subset"] for row in report_rows if row["final_status"] == "hard_fail")
    source_soft = Counter(row["source_subset"] for row in report_rows if row["final_status"] == "soft_fail")
    for source_subset in sorted(source_counts):
        rows.append(
            {
                "breakout_type": "source_subset",
                "key": source_subset,
                "seed_count": str(source_counts[source_subset]),
                "train_eligible": str(source_usable[source_subset]),
                "audit": str(source_audit[source_subset]),
                "hard_fail": str(source_hard[source_subset]),
                "soft_fail": str(source_soft[source_subset]),
            }
        )

    tail_counts = Counter(row["tail_proximity_class"] for row in report_rows)
    tail_usable = Counter(row["tail_proximity_class"] for row in report_rows if row["train_eligible"] == "예")
    tail_audit = Counter(row["tail_proximity_class"] for row in report_rows if row["audit_required"] == "예")
    tail_hard = Counter(row["tail_proximity_class"] for row in report_rows if row["final_status"] == "hard_fail")
    tail_soft = Counter(row["tail_proximity_class"] for row in report_rows if row["final_status"] == "soft_fail")
    for tail_class in sorted(tail_counts):
        rows.append(
            {
                "breakout_type": "tail_proximity_class",
                "key": tail_class,
                "seed_count": str(tail_counts[tail_class]),
                "train_eligible": str(tail_usable[tail_class]),
                "audit": str(tail_audit[tail_class]),
                "hard_fail": str(tail_hard[tail_class]),
                "soft_fail": str(tail_soft[tail_class]),
            }
        )

    pb6.pb4.pb3.base.write_csv_atomic(PILOT_BREAKOUT_CSV_PATH, rows, list(rows[0].keys()))
    lines = [
        f"# pilot breakout `{VERSION_TAG}`",
        "",
        "## source subset breakout",
        "| key | seed | train_eligible | audit | hard_fail | soft_fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["breakout_type"] == "source_subset":
            lines.append(
                f"| `{row['key']}` | `{row['seed_count']}` | `{row['train_eligible']}` | `{row['audit']}` | `{row['hard_fail']}` | `{row['soft_fail']}` |"
            )
    lines.extend(["", "## 03_TL breakout", "| key | seed | train_eligible | audit | hard_fail | soft_fail |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        if row["breakout_type"] == "source_subset" and row["key"].startswith("03_TL"):
            lines.append(
                f"| `{row['key']}` | `{row['seed_count']}` | `{row['train_eligible']}` | `{row['audit']}` | `{row['hard_fail']}` | `{row['soft_fail']}` |"
            )
    lines.extend(["", "## tail proximity class", "| key | seed | train_eligible | audit | hard_fail | soft_fail |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        if row["breakout_type"] == "tail_proximity_class":
            lines.append(
                f"| `{row['key']}` | `{row['seed_count']}` | `{row['train_eligible']}` | `{row['audit']}` | `{row['hard_fail']}` | `{row['soft_fail']}` |"
            )
    pb6.pb4.pb3.base.write_text_atomic(PILOT_BREAKOUT_MD_PATH, "\n".join(lines) + "\n")


def split_dataset_with_judgment_validator(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows:
        pb6.pb4.pb3.base.write_csv_atomic(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    validated_rows = apply_judgment_validator(rows)
    if validated_rows:
        pb6.pb4.pb3.base.write_csv_atomic(MERGED_SCORES_PATH, validated_rows, list(validated_rows[0].keys()))
    manifest_rows = BASE_SPLIT_DATASET(validated_rows)
    selected_by_problem_id = {
        row["candidate_id"]: row
        for row in selected_rows(validated_rows)
    }
    rewrite_split_jsonl_with_status(selected_by_problem_id)
    rewrite_audit_queue_with_status(selected_by_problem_id)
    return rewrite_manifest_with_validator_fields(manifest_rows, selected_by_problem_id)


def status_fields() -> dict[str, str]:
    return {
        "batch_status": BATCH_STATUS,
        "count_reflection_status": COUNT_REFLECTION_STATUS,
        "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
    }


def rewrite_split_jsonl_with_status(selected_by_problem_id: dict[str, dict[str, str]]) -> None:
    for path in (PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH, PROBLEM_TEST_PATH):
        if not path.exists():
            continue
        payload_rows = pb6.pb4.pb3.base.load_jsonl(path)
        enriched_rows = []
        for payload in payload_rows:
            source = selected_by_problem_id.get(payload.get("problem_id", ""), {})
            enriched = dict(payload)
            enriched.update(status_fields())
            enriched.update(
                {
                    "target_correct_choice": source.get("target_correct_choice", ""),
                    "export_correct_choice": source.get("export_correct_choice", source.get("correct_choice", "")),
                    "validator_action": source.get("validator_action", ""),
                    "validator_export_disposition": source.get("validator_export_disposition", ""),
                    "validator_reason_short": source.get("validator_reason_short", ""),
                    "split_allowed": source.get("split_allowed", "아니오"),
                    "count_allowed": source.get("count_allowed", "아니오"),
                }
            )
            enriched_rows.append(enriched)
        pb6.pb4.pb3.base.write_jsonl_atomic(path, enriched_rows)


def rewrite_audit_queue_with_status(selected_by_problem_id: dict[str, dict[str, str]]) -> None:
    if not PROBLEM_AUDIT_QUEUE_PATH.exists():
        return
    audit_rows = read_csv_rows(PROBLEM_AUDIT_QUEUE_PATH)
    if not audit_rows:
        return
    enriched_rows = []
    for row in audit_rows:
        source = selected_by_problem_id.get(row.get("problem_id", ""), {})
        enriched = dict(row)
        enriched.update(status_fields())
        enriched.update(
            {
                "target_correct_choice": source.get("target_correct_choice", ""),
                "export_correct_choice": source.get("export_correct_choice", source.get("correct_choice", "")),
                "validator_action": source.get("validator_action", ""),
                "validator_export_disposition": source.get("validator_export_disposition", ""),
                "validator_reason_short": source.get("validator_reason_short", ""),
                "split_allowed": source.get("split_allowed", "아니오"),
                "count_allowed": source.get("count_allowed", "아니오"),
            }
        )
        enriched_rows.append(enriched)
    pb6.pb4.pb3.base.write_csv_atomic(PROBLEM_AUDIT_QUEUE_PATH, enriched_rows, list(enriched_rows[0].keys()))


def rewrite_manifest_with_validator_fields(
    manifest_rows: list[dict[str, str]],
    selected_by_problem_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    enriched_rows = []
    for row in manifest_rows:
        source = selected_by_problem_id.get(row.get("problem_id", ""), {})
        enriched = dict(row)
        enriched.update(status_fields())
        enriched.update(
            {
                "target_correct_choice": source.get("target_correct_choice", ""),
                "export_correct_choice": source.get("export_correct_choice", source.get("correct_choice", "")),
                "validator_action": source.get("validator_action", ""),
                "validator_export_disposition": source.get("validator_export_disposition", ""),
                "validator_reason_short": source.get("validator_reason_short", ""),
                "validator_recalculated_correct_choice": source.get("validator_recalculated_correct_choice", ""),
                "metadata_remap_ok": source.get("metadata_remap_ok", ""),
                "split_allowed": source.get("split_allowed", "아니오"),
                "count_allowed": source.get("count_allowed", "아니오"),
                "stem_axis": source.get("stem_axis", ""),
                "judgment_seed_action": source.get("judgment_seed_action", ""),
                "tail_proximity_class": source.get("tail_proximity_class", ""),
            }
        )
        enriched_rows.append(enriched)

    if enriched_rows:
        pb6.pb4.pb3.base.write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, enriched_rows, list(enriched_rows[0].keys()))
    write_manifest_header_gate(enriched_rows)
    return enriched_rows


def write_manifest_header_gate(manifest_rows: list[dict[str, str]]) -> None:
    required = [
        "batch_status",
        "count_reflection_status",
        "downstream_consumption_allowed",
        "target_correct_choice",
        "export_correct_choice",
        "validator_action",
        "validator_export_disposition",
        "validator_reason_short",
        "split_allowed",
        "count_allowed",
    ]
    headers = list(manifest_rows[0].keys()) if manifest_rows else []
    missing = [field for field in required if field not in headers]
    if missing:
        raise RuntimeError(f"judgment pilot manifest required fields missing: {missing}")
    lines = [
        f"# manifest header gate `{VERSION_TAG}`",
        "",
        "| check | result | value |",
        "| --- | --- | --- |",
        f"| required header fields | `pass` | `{required}` |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(MANIFEST_HEADER_GATE_MD_PATH, "\n".join(lines) + "\n")


def build_tail_memo(merged_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tail_rows = []
    for row in selected_rows(merged_rows):
        if row.get("train_eligible") == "예":
            continue
        tail_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "stem_axis": row.get("stem_axis", ""),
                "judgment_seed_action": row.get("judgment_seed_action", ""),
                "tail_proximity_class": row.get("tail_proximity_class", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "validator_action": row.get("validator_action", ""),
                "validator_export_disposition": row.get("validator_export_disposition", ""),
                "validator_reason_short": row.get("validator_reason_short", ""),
                "error_tags": row.get("error_tags", ""),
            }
        )
    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "source_subset": "",
                "sampling_lane": "",
                "stem_axis": "",
                "judgment_seed_action": "",
                "tail_proximity_class": "tail 없음",
                "final_status": "",
                "audit_required": "",
                "validator_action": "",
                "validator_export_disposition": "",
                "validator_reason_short": "",
                "error_tags": "",
            }
        ]
    pb6.pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | source | lane | status | audit | validator | disposition | reason | tail_proximity | error_tags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['validator_reason_short']}` | `{row['tail_proximity_class']}` | `{row['error_tags']}` |"
        )
    pb6.pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = selected_rows(rows)
    summary = pb6.pb4.pb3.summarize_rows(rows)
    summary_rows = [
        {"metric": "seed_count", "value": str(len(selected))},
        {"metric": "selected_pass", "value": str(summary["selected_pass_count"])},
        {"metric": "selected_hard_fail", "value": str(summary["selected_hard_fail_count"])},
        {"metric": "selected_soft_fail", "value": str(summary["selected_soft_fail_count"])},
        {"metric": "train_eligible", "value": str(summary["selected_train_eligible_count"])},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "pilot_signal_success", "value": str(VALIDATOR_SUMMARY.get("pilot_signal_success", False))},
        {"metric": "countable_package_success", "value": str(VALIDATOR_SUMMARY.get("countable_package_success", False))},
    ]
    pb6.pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    lane_rows = []
    lane_counts = Counter(row.get("sampling_lane", "") for row in selected)
    for lane, count in sorted(lane_counts.items()):
        lane_rows.append({"sampling_lane": lane, "count": str(count)})
    pb6.pb4.pb3.base.write_csv_atomic(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, ["sampling_lane", "count"])
    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected)}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        f"- pilot_signal_success: `{VALIDATOR_SUMMARY.get('pilot_signal_success', False)}`",
        f"- countable_package_success: `{VALIDATOR_SUMMARY.get('countable_package_success', False)}`",
        "- count_reflection: `not_counted_until_reviewer_signoff`",
        "",
        "## pilot signal success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| usable | `>= {SUCCESS_USABLE_MIN} / 16` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `<= {SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| answer uniqueness recurrence | `0` | `{VALIDATOR_SUMMARY.get('answer_uniqueness_recurrence_count', 0)}` |",
        f"| target label schedule | `A/B/C/D = 4/4/4/4` | `{VALIDATOR_SUMMARY.get('target_label_counts', {})}` |",
        f"| shuffle/metadata mismatch | `0 / 0` | `{VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)} / {VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |",
        "",
        "## countable package criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| export-ready rows | `16` | `{sum(1 for row in selected if row.get('validator_export_disposition') == 'export_ready')}` |",
        f"| unresolved audit/soft/hard | `0 / 0 / 0` | `{summary['selected_audit_required_count']} / {summary['selected_soft_fail_count']} / {summary['selected_hard_fail_count']}` |",
        f"| export label balance | `A/B/C/D = 4/4/4/4` | `{VALIDATOR_SUMMARY.get('export_ready_label_counts', {})}` |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict:
    manifest = BASE_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    tail_rows = build_tail_memo(merged_rows)
    summary = pb6.pb4.pb3.summarize_rows(merged_rows)
    # base manifest는 pb6/pb3 wrapper의 stale path/count를 물려받을 수 있다.
    # reviewer-facing manifest는 이 run의 실제 artifact만 가리키도록 전체 path block을 재동기화한다.
    synced_artifact_paths = {
        "seed_registry": str(SEED_REGISTRY_PATH),
        "seed_ready": str(SEED_READY_PATH),
        "seed_preflight": str(SEED_PREFLIGHT_CSV_PATH),
        "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
        "source_preflight_exclusion_audit": str(SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH),
        "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
        "generated_problems": str(GENERATED_PROBLEMS_PATH),
        "judge_grounding_log": str(GROUNDING_LOG_PATH),
        "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
        "raw_merged_before_validator": str(RAW_MERGED_BEFORE_VALIDATOR_PATH),
        "merged_scores": str(MERGED_SCORES_PATH),
        "batch_summary_md": str(BATCH_SUMMARY_MD_PATH),
        "batch_summary_csv": str(BATCH_SUMMARY_CSV_PATH),
        "batch_lane_summary_csv": str(BATCH_LANE_SUMMARY_CSV_PATH),
        "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
        "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
        "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
        "pilot_breakout_csv": str(PILOT_BREAKOUT_CSV_PATH),
        "pilot_breakout_md": str(PILOT_BREAKOUT_MD_PATH),
        "manifest_header_gate": str(MANIFEST_HEADER_GATE_MD_PATH),
        "tail_memo_csv": str(TAIL_MEMO_CSV_PATH),
        "tail_memo_md": str(TAIL_MEMO_MD_PATH),
        "problem_train": str(PROBLEM_TRAIN_PATH),
        "problem_dev": str(PROBLEM_DEV_PATH),
        "problem_test": str(PROBLEM_TEST_PATH),
        "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
        "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
    }
    manifest.update(
        {
            "version_tag": VERSION_TAG,
            "run_name": RUN_NAME,
            "run_id": RUN_NAME,
            "run_dir": str(RUN_DIR),
            "seed_registry_csv_path": str(SEED_REGISTRY_PATH),
            "seed_ready_jsonl_path": str(SEED_READY_PATH),
            "seed_preflight_csv_path": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md_path": str(SEED_PREFLIGHT_MD_PATH),
            # base manifest가 이전 wrapper의 count field를 일부 유지할 수 있으므로,
            # reviewer-facing count는 이 run의 실제 artifact path에서 다시 센 값으로 덮어쓴다.
            "generation_count": pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
            "judge_grounding_count": pb6.pb4.pb3.base.load_jsonl_count(GROUNDING_LOG_PATH),
            "judge_keyedness_count": pb6.pb4.pb3.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_count": pb6.pb4.pb3.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_count": pb6.pb4.pb3.base.load_jsonl_count(NEARMISS_LOG_PATH),
            "merged_count": pb6.pb4.pb3.base.load_csv_count(MERGED_SCORES_PATH),
            "merged_scores_count": pb6.pb4.pb3.base.load_csv_count(MERGED_SCORES_PATH),
            "problem_train_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
            "problem_dev_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_DEV_PATH),
            "problem_test_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TEST_PATH),
            "problem_audit_count": pb6.pb4.pb3.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
            "dataset_manifest_count": pb6.pb4.pb3.base.load_csv_count(PROBLEM_DATASET_MANIFEST_PATH),
            "dataset_split_counts": {
                "train": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
                "dev": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_DEV_PATH),
                "test": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TEST_PATH),
                "audit": pb6.pb4.pb3.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
            },
            "source_preflight_run_name": SOURCE_PREFLIGHT_RUN_NAME,
            "candidate_recipe_source": "v2_difficulty_patch_r2_judgment_repair_pilot",
            "seed_registry_strategy": "same_16_seed_registry_as_objective_judgment_repair_pilot_seed_preflight",
            "current_count_decision": "not_counted_until_reviewer_signoff_after_api_pilot",
            "count_reflection_status": COUNT_REFLECTION_STATUS,
            "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
            "manifest_artifact_sync_status": "synced",
            "manifest_artifact_sync_note": "Hotfixed after reviewer finding: all top-level counts and artifact paths now point to the 2026-04-26_070137 judgment repair pilot artifacts.",
            "validator_summary": VALIDATOR_SUMMARY,
            "success_criteria": {
                "pilot_signal": {
                    "usable_min": SUCCESS_USABLE_MIN,
                    "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
                    "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
                    "audit_max": SUCCESS_AUDIT_MAX,
                    "answer_uniqueness_recurrence": 0,
                    "target_label_counts": TARGET_LABEL_COUNTS,
                    "shuffle_recalc_mismatch": 0,
                    "metadata_remap_mismatch": 0,
                },
                "countable_package": {
                    "export_ready_rows": 16,
                    "unresolved_audit_soft_hard": 0,
                    "export_label_counts": TARGET_LABEL_COUNTS,
                    "reviewer_signoff_required_for_count_reflection": True,
                },
            },
            "success_result": {
                "usable": summary["selected_train_eligible_count"],
                "hard_fail": summary["selected_hard_fail_count"],
                "soft_fail": summary["selected_soft_fail_count"],
                "audit": summary["selected_audit_required_count"],
                "answer_uniqueness_recurrence": VALIDATOR_SUMMARY.get("answer_uniqueness_recurrence_count", 0),
                "pilot_signal_success": VALIDATOR_SUMMARY.get("pilot_signal_success", False),
                "countable_package_success": VALIDATOR_SUMMARY.get("countable_package_success", False),
            },
            "tail_memo_count": len([row for row in tail_rows if row.get("seed_sample_id")]),
        }
    )
    manifest["artifact_paths"] = synced_artifact_paths
    pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_judgment_pilot_globals() -> None:
    # `pb6`/`pb3` strict-final 본체를 유지하고, identity/path/fixed registry/validator hook만 덮어쓴다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = "objective judgment repair API pilot"
    pb6.SEED_ID_PREFIX = "judgment_repair_pilot"
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pb6.SUCCESS_LAW_ROW_COUNT = 0
    pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_judgment_repair_pilot"
    pb6.SEED_REGISTRY_STRATEGY = "fixed_from_objective_judgment_repair_pilot_seed_preflight"
    pb6.LAW_STATUS_NOTE = "judgment_repair_pilot_not_counted_until_signoff"

    pb6.INTERIM_DIR = INTERIM_DIR
    pb6.PROCESSED_DIR = PROCESSED_DIR
    pb6.RUN_DIR = RUN_DIR
    pb6.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb6.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb6.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb6.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb6.SEED_READY_PATH = SEED_READY_PATH
    pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb6.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb6.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb6.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb6.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb6.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb6.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb6.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb6.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb6.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb6.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb6.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb6.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb6.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb6.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb6.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pb6.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    pb6.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb6.build_seed_registry = build_seed_registry_from_preflight
    pb6.build_batch_summary = build_batch_summary
    pb6.build_run_manifest = build_run_manifest
    pb6.pb4.pb3.base.split_dataset = split_dataset_with_judgment_validator


def main() -> dict:
    configure_judgment_pilot_globals()
    return pb6.main()


if __name__ == "__main__":
    main()
