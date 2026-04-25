from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `pb9` 실패 원인이 04_TL 결정례 generalization weak distractor tail에 모였으므로,
# 전체 40개 재실행이 아니라 8개 calibration pilot으로 generator/Judge/validator 배선을 확인한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_micro_pilot as micro,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_04tl_calibration_wiring_check as wiring_check,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_decision_only as pb9_smoke,
)


VERSION_TAG = "pb9_04tl_decision_weak_distractor_calibration_pilot"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_04tl_8seed_targeted_calibration_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = micro.pb8.pb6.pb4.pb3.base.PROJECT_ROOT
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
JUDGE_STRUCTURED_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"judge_structured_contract_{VERSION_TAG}.md"
STRUCTURED_FIELD_GATE_MD_PATH = RUN_EXPORTS_DIR / f"structured_field_gate_{VERSION_TAG}.md"

REFERENCE_PB9_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb9_decision_only_controlled_production_with_choice_validator"
    / "seed_registry.csv"
)

SOURCE_COUNTS = {"04_TL_결정례_QA": 8}
EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 8}
EXPECTED_LANE_BY_DOC = {("결정례_QA", "generalization_03_04"): 8}
TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}
SUCCESS_USABLE_MIN = 7
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0

BATCH_STATUS = "calibration_not_counted"
COUNT_REFLECTION_STATUS = "not_counted"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

# Judge 함수는 `r2` wrapper가 아니라 재배선된 `base` module에서 실제 호출되므로,
# structured NearMiss 요구사항도 같은 호출 지점에 주입해야 한다.
ORIGINAL_BASE_BUILD_JUDGE_PROMPT = micro.pb8.pb6.pb4.pb3.base.build_judge_prompt
ORIGINAL_BASE_BUILD_JUDGE_ROW = micro.pb8.pb6.pb4.pb3.base.build_judge_row
ORIGINAL_BUILD_GENERATION_MESSAGES = micro.build_generation_messages
ORIGINAL_SPLIT_DATASET = micro.ORIGINAL_SPLIT_DATASET
ORIGINAL_BUILD_RUN_MANIFEST = micro.build_run_manifest

VALIDATOR_SUMMARY: dict[str, object] = {}


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return micro.pb8.pb6.load_csv_rows_if_exists(path)


def collect_excluded_rows_for_calibration() -> list[dict[str, str]]:
    # 실패한 `pb9` seed도 이미 API/Judge를 본 seed이므로 이번 calibration에서는 fresh pool에서 제외한다.
    rows = pb9_smoke.collect_excluded_rows()
    rows.extend(load_csv_rows_if_exists(REFERENCE_PB9_SEED_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(micro.REFERENCE_PB8_SEED_REGISTRY_PATH))
    rows.extend(load_csv_rows_if_exists(micro.REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH))
    return rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## 04TL decision weak-distractor calibration 추가 지시
- 이번 run은 `04_TL_결정례_QA generalization_03_04` `8개` calibration pilot이다.
- 오답 3개 모두 정답과 같은 결정 이유, 판단 기준, 적용 사실 중 하나를 공유해야 한다.
- 각 오답은 같은 anchor 안에서 요건, 시점, 주체, 효과, 예외, 절차 중 정확히 한 축만 다르게 비튼다.
- 오답 중 하나라도 일반 정의, 기관 역할, 상식적 반대말, 너무 포괄적인 문장으로 빠지면 weak distractor로 본다.
- 생성 단계에서는 label 위치가 아니라 정답 유일성, 오답 의미 분리, one-axis near-miss 품질에 집중한다.
"""
    return messages


def build_structured_judge_prompt(seed: dict[str, str], generation: dict[str, str], role_name: str) -> str:
    prompt = ORIGINAL_BASE_BUILD_JUDGE_PROMPT(seed, generation, role_name)
    if role_name != "NearMiss":
        return prompt
    return prompt + """

## 04TL calibration structured output 추가 요구
위 NearMiss 점수와 별도로 아래 구조화 필드를 반드시 함께 출력한다.

- `weak_distractor_count`: 정답을 제외한 오답 3개 중 weak distractor로 보이는 선택지 수. 정수 `0`∼`3`.
- `weak_distractor_labels`: weak distractor label. 없으면 빈 문자열 `""`, 여러 개면 `B|D`처럼 `|`로 연결. 정답 label은 절대 포함하지 않는다.
- `all_three_near_miss`: 오답 3개가 모두 같은 legal anchor를 공유하면서 한 축만 어긋나면 `예`, 아니면 `아니오`.
- `one_axis_perturbation_count`: 오답 3개 중 요건/시점/주체/효과/예외/절차 중 한 축만 어긋난 선택지 수. 정수 `0`∼`3`.

출력 JSON은 반드시 아래 key를 모두 포함한다.

{
  "score": 1,
  "pass_or_fail": "pass 또는 fail",
  "error_tags": ["오답약함"],
  "one_sentence_reason": "한 문장 이유",
  "weak_distractor_count": 0,
  "weak_distractor_labels": "",
  "all_three_near_miss": "예",
  "one_axis_perturbation_count": 3
}
"""


def build_structured_judge_row(seed: dict[str, str], generation: dict[str, str], role_name: str, response: dict) -> dict:
    row = ORIGINAL_BASE_BUILD_JUDGE_ROW(seed, generation, role_name, response)
    if role_name == "NearMiss":
        parsed = response["json"]
        row["weak_distractor_count"] = str(parsed.get("weak_distractor_count", ""))
        row["weak_distractor_labels"] = str(parsed.get("weak_distractor_labels", ""))
        row["all_three_near_miss"] = str(parsed.get("all_three_near_miss", ""))
        row["one_axis_perturbation_count"] = str(parsed.get("one_axis_perturbation_count", ""))
    return row


def classify_validator_tail(row: dict[str, str]) -> str:
    # Reviewer-facing tail memo는 Judge 실패와 validator 차단을 같은 표에서 추적해야 하므로
    # 기존 tail taxonomy보다 answer uniqueness / weak distractor 원인을 먼저 드러낸다.
    error_tags = row.get("error_tags", "")
    validator_action = row.get("validator_action", "")
    weak_count = wiring_check.parse_int(row.get("weak_distractor_count", ""))
    if "정답 비유일" in error_tags or "오답이 정답 가능" in error_tags:
        return "decision answer uniqueness failure"
    if validator_action == "regenerate" or weak_count > 0 or "오답약함" in error_tags or "near_miss_부족" in error_tags:
        return "decision weak distractor"
    if row.get("final_status") == "hard_fail":
        return "hard_fail_tail"
    if row.get("final_status") == "soft_fail":
        return "soft_fail_tail"
    if row.get("audit_required") == "예":
        return "audit_tail"
    return "tail"


def build_tail_memo_with_validator_fields(merged_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # `pb9_04tl` 이후 review는 validator/Judge calibration 자체를 봐야 하므로,
    # tail memo에도 split/count 허용 여부와 label shuffle 결과를 같이 보존한다.
    tail_rows = []
    for row in merged_rows:
        if row.get("selected_for_seed") != "예":
            continue
        is_tail = (
            row.get("train_eligible") != "예"
            or row.get("audit_required") == "예"
            or row.get("validator_action") not in {"", "accept"}
            or row.get("validator_export_disposition") not in {"", "export_ready"}
        )
        if not is_tail:
            continue
        tail_rows.append(
            {
                "seed_sample_id": row.get("seed_sample_id", ""),
                "doc_type_name": row.get("doc_type_name", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "tail_class": classify_validator_tail(row),
                "validator_action": row.get("validator_action", ""),
                "validator_export_disposition": row.get("validator_export_disposition", ""),
                "validator_status": row.get("validator_status", ""),
                "validator_reasons": row.get("validator_reasons", ""),
                "nearmiss_score": row.get("nearmiss_score", ""),
                "weak_distractor_count": row.get("weak_distractor_count", ""),
                "weak_distractor_labels": row.get("weak_distractor_labels", ""),
                "all_three_near_miss": row.get("all_three_near_miss", ""),
                "one_axis_perturbation_count": row.get("one_axis_perturbation_count", ""),
                "target_correct_choice": row.get("target_correct_choice", ""),
                "export_correct_choice": row.get("export_correct_choice", row.get("correct_choice", "")),
                "split_allowed": row.get("split_allowed", ""),
                "count_allowed": row.get("count_allowed", ""),
                "generated_stem": row.get("generated_stem", ""),
            }
        )

    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "doc_type_name": "",
                "source_subset": "",
                "sampling_lane": "",
                "final_status": "",
                "audit_required": "",
                "error_tags": "",
                "tail_class": "tail 없음",
                "validator_action": "",
                "validator_export_disposition": "",
                "validator_status": "",
                "validator_reasons": "",
                "nearmiss_score": "",
                "weak_distractor_count": "",
                "weak_distractor_labels": "",
                "all_three_near_miss": "",
                "one_axis_perturbation_count": "",
                "target_correct_choice": "",
                "export_correct_choice": "",
                "split_allowed": "",
                "count_allowed": "",
                "generated_stem": "",
            }
        ]

    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | source | lane | status | action | disposition | tail_class | nearmiss | weak | target/export | split/count |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['tail_class']}` | `{row['nearmiss_score']}` | `{row['weak_distractor_count']}:{row['weak_distractor_labels']}` | `{row['target_correct_choice']}/{row['export_correct_choice']}` | `{row['split_allowed']}/{row['count_allowed']}` |"
        )
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def target_label_for_selected_index(index: int) -> str:
    return validator_replay.CHOICE_LABELS[index % len(validator_replay.CHOICE_LABELS)]


def write_target_label_schedule(selected: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(selected):
        rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "candidate_id": row["candidate_id"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "target_correct_choice": target_label_for_selected_index(index),
            }
        )
    counts = Counter(row["target_correct_choice"] for row in rows)
    if dict(counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"04TL target label schedule mismatch: {dict(counts)}")
    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    micro.pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    return rows


def nearmiss_structured_by_candidate() -> dict[str, dict[str, str]]:
    rows = micro.pb8.pb6.pb4.pb3.base.load_jsonl(NEARMISS_LOG_PATH)
    return {row["candidate_id"]: row for row in rows}


def enrich_rows_with_structured_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    nearmiss_map = nearmiss_structured_by_candidate()
    for row in rows:
        nearmiss = nearmiss_map.get(row["candidate_id"], {})
        row["weak_distractor_count"] = str(nearmiss.get("weak_distractor_count", ""))
        row["weak_distractor_labels"] = str(nearmiss.get("weak_distractor_labels", ""))
        row["all_three_near_miss"] = str(nearmiss.get("all_three_near_miss", ""))
        row["one_axis_perturbation_count"] = str(nearmiss.get("one_axis_perturbation_count", ""))
    return rows


def append_tag(existing_tags: str, tag: str) -> str:
    tags = [value for value in (existing_tags or "").split("|") if value]
    if tag not in tags:
        tags.append(tag)
    return "|".join(tags)


def final_action_for_row(row: dict[str, str]) -> tuple[str, str, list[str]]:
    calibration_action, calibration_status, calibration_reasons = wiring_check.choose_calibration_action(row)
    if row.get("upstream_final_status") == "hard_fail":
        return "hard_block", "upstream_hard_fail", [*calibration_reasons, "upstream_hard_fail"]
    if row.get("upstream_audit_required") == "예" and calibration_action == "accept":
        return "audit", "upstream_audit_required", ["upstream_audit_required"]
    return calibration_action, calibration_status, calibration_reasons


def apply_calibration_validator(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    global VALIDATOR_SUMMARY

    enriched_rows = enrich_rows_with_structured_fields(rows)
    selected = selected_rows(enriched_rows)
    schedule_rows = write_target_label_schedule(selected)
    target_by_candidate = {row["candidate_id"]: row["target_correct_choice"] for row in schedule_rows}
    report_rows = []

    for row in enriched_rows:
        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
        row["upstream_train_eligible"] = row.get("train_eligible", "")
        row["validator_action"] = ""
        row["validator_export_disposition"] = ""
        row["validator_status"] = ""
        row["validator_reasons"] = ""
        row["split_allowed"] = "아니오"
        row["count_allowed"] = "아니오"
        row["target_correct_choice"] = ""
        row["validator_recalculated_correct_choice"] = ""
        row["validator_correct_choice_match_count"] = ""
        row["validator_shuffle_recalc_ok"] = ""
        row["metadata_remap_ok"] = ""
        row["metadata_remap_reasons"] = ""
        row["batch_status"] = BATCH_STATUS
        row["count_reflection_status"] = COUNT_REFLECTION_STATUS
        row["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
        row["export_correct_choice"] = row.get("correct_choice", "")

    for row in selected:
        target_label = target_by_candidate[row["candidate_id"]]
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices,
            row.get("correct_choice", ""),
            target_label,
        )
        action, status, reasons = final_action_for_row(row)
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
            if not metadata_ok:
                action = "hard_block"
                status = "metadata_remap_block"
                reasons = [*reasons, *metadata_reasons]
        else:
            row["metadata_remap_ok"] = "대상아님"
            row["metadata_remap_reasons"] = ""

        disposition = wiring_check.disposition_for_action(action)
        split_allowed = wiring_check.split_allowed_for_action(action)
        row["validator_action"] = action
        row["validator_export_disposition"] = disposition
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["split_allowed"] = split_allowed
        row["count_allowed"] = "아니오"
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
                "seed_sample_id": row["seed_sample_id"],
                "candidate_id": row["candidate_id"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "upstream_final_status": row["upstream_final_status"],
                "upstream_audit_required": row["upstream_audit_required"],
                "nearmiss_score": row.get("nearmiss_score", ""),
                "weak_distractor_count": row.get("weak_distractor_count", ""),
                "weak_distractor_labels": row.get("weak_distractor_labels", ""),
                "all_three_near_miss": row.get("all_three_near_miss", ""),
                "one_axis_perturbation_count": row.get("one_axis_perturbation_count", ""),
                "validator_action": action,
                "validator_export_disposition": disposition,
                "validator_status": status,
                "validator_reasons": row["validator_reasons"],
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "shuffle_recalc_ok": row["validator_shuffle_recalc_ok"],
                "metadata_remap_ok": row["metadata_remap_ok"],
                "metadata_remap_reasons": row["metadata_remap_reasons"],
                "split_allowed": split_allowed,
                "count_allowed": "아니오",
                "final_status": row["final_status"],
                "audit_required": row["audit_required"],
                "train_eligible": row["train_eligible"],
                "export_correct_choice": row["export_correct_choice"],
            }
        )

    action_counts = Counter(row["validator_action"] for row in selected)
    disposition_counts = Counter(row["validator_export_disposition"] for row in selected)
    export_ready_rows = [row for row in selected if row["validator_export_disposition"] == "export_ready"]
    export_label_counts = Counter(row["correct_choice"] for row in export_ready_rows)
    target_label_counts = Counter(row["target_correct_choice"] for row in selected)
    summary = micro.pb8.pb6.pb4.pb3.summarize_rows(enriched_rows)
    structured_missing_count = sum(
        1
        for row in selected
        if any(str(row.get(field, "")).strip() == "" for field in wiring_check.JUDGE_STRUCTURED_REQUIRED_FIELDS if field != "weak_distractor_labels")
    )
    label_consistency_failure_count = sum(
        1 for row in selected if row["validator_status"] == "weak_distractor_label_consistency_failure"
    )
    export_ready_weak_count = sum(
        1
        for row in export_ready_rows
        if wiring_check.parse_int(row.get("weak_distractor_count", "")) not in {0}
    )
    export_ready_all_three_no_count = sum(1 for row in export_ready_rows if row.get("all_three_near_miss") != "예")
    shuffle_mismatch_count = sum(1 for row in selected if row["validator_shuffle_recalc_ok"] != "예")
    metadata_mismatch_count = sum(
        1
        for row in selected
        if row["validator_export_disposition"] in {"export_ready", "audit_queue"}
        and row.get("metadata_remap_ok") != "예"
    )
    VALIDATOR_SUMMARY = {
        "selected_count": len(selected),
        "validator_action_counts": dict(action_counts),
        "validator_export_disposition_counts": dict(disposition_counts),
        "target_label_counts": {label: target_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_ready_label_counts": {label: export_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "structured_missing_count": structured_missing_count,
        "weak_label_consistency_failure_count": label_consistency_failure_count,
        "export_ready_weak_distractor_count": export_ready_weak_count,
        "export_ready_all_three_near_miss_no_count": export_ready_all_three_no_count,
        "shuffle_recalc_mismatch_count": shuffle_mismatch_count,
        "metadata_remap_mismatch_count": metadata_mismatch_count,
    }
    VALIDATOR_SUMMARY["pilot_success_passed"] = (
        VALIDATOR_SUMMARY["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and VALIDATOR_SUMMARY["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and structured_missing_count == 0
        and label_consistency_failure_count == 0
        and export_ready_weak_count == 0
        and export_ready_all_three_no_count == 0
        and shuffle_mismatch_count == 0
        and metadata_mismatch_count == 0
    )

    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, list(report_rows[0].keys()))
    write_validator_report_md(report_rows)
    write_structured_field_gate_md(selected)
    write_validator_wiring_check_md()
    return enriched_rows


def write_judge_structured_contract_md() -> None:
    lines = [
        f"# judge structured contract `{VERSION_TAG}`",
        "",
        "## required fields",
        "| field | allowed value | note |",
        "| --- | --- | --- |",
        "| `weak_distractor_count` | integer `0`∼`3` | weak near-miss distractor count among wrong choices |",
        "| `weak_distractor_labels` | blank or `A|B|C|D` labels | required when count is positive; correct label excluded |",
        "| `all_three_near_miss` | `예` / `아니오` | all wrong choices share legal anchor and one-axis perturbation |",
        "| `one_axis_perturbation_count` | integer `0`∼`3` | number of wrong choices that differ by exactly one axis |",
        "",
        "## count reflection",
        "- this pilot is calibration-only and `not_counted`.",
    ]
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(JUDGE_STRUCTURED_CONTRACT_MD_PATH, "\n".join(lines) + "\n")


def write_validator_report_md(report_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{VALIDATOR_SUMMARY.get('selected_count', 0)}`",
        f"- validator_action_counts: `{VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- validator_export_disposition_counts: `{VALIDATOR_SUMMARY.get('validator_export_disposition_counts', {})}`",
        f"- target_label_counts: `{VALIDATOR_SUMMARY.get('target_label_counts', {})}`",
        f"- export_ready_label_counts: `{VALIDATOR_SUMMARY.get('export_ready_label_counts', {})}`",
        f"- pilot_success_passed: `{VALIDATOR_SUMMARY.get('pilot_success_passed', False)}`",
        "",
        "## row actions",
        "| seed | weak_count | weak_labels | all_three | one_axis | action | disposition | final | audit | train |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['weak_distractor_count']}` | `{row['weak_distractor_labels']}` | `{row['all_three_near_miss']}` | `{row['one_axis_perturbation_count']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['train_eligible']}` |"
        )
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_structured_field_gate_md(selected: list[dict[str, str]]) -> None:
    lines = [
        f"# structured field gate `{VERSION_TAG}`",
        "",
        "| gate | result | value |",
        "| --- | --- | --- |",
        f"| required field missing/parse | `{'pass' if VALIDATOR_SUMMARY.get('structured_missing_count', 0) == 0 else 'fail'}` | `{VALIDATOR_SUMMARY.get('structured_missing_count', 0)}` |",
        f"| weak label consistency failure | `{'pass' if VALIDATOR_SUMMARY.get('weak_label_consistency_failure_count', 0) == 0 else 'fail'}` | `{VALIDATOR_SUMMARY.get('weak_label_consistency_failure_count', 0)}` |",
        f"| export-ready weak distractor | `{'pass' if VALIDATOR_SUMMARY.get('export_ready_weak_distractor_count', 0) == 0 else 'fail'}` | `{VALIDATOR_SUMMARY.get('export_ready_weak_distractor_count', 0)}` |",
        f"| export-ready all_three_near_miss | `{'pass' if VALIDATOR_SUMMARY.get('export_ready_all_three_near_miss_no_count', 0) == 0 else 'fail'}` | `{VALIDATOR_SUMMARY.get('export_ready_all_three_near_miss_no_count', 0)}` |",
        f"| shuffle/metadata mismatch | `{'pass' if VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0) == 0 and VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0) == 0 else 'fail'}` | `shuffle {VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)} / metadata {VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |",
        "",
        "## selected rows",
        "| seed | weak_count | labels | all_three | one_axis | status |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row.get('weak_distractor_count', '')}` | `{row.get('weak_distractor_labels', '')}` | `{row.get('all_three_near_miss', '')}` | `{row.get('one_axis_perturbation_count', '')}` | `{row.get('validator_status', '')}` |"
        )
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(STRUCTURED_FIELD_GATE_MD_PATH, "\n".join(lines) + "\n")


def write_validator_wiring_check_md() -> None:
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| Judge structured contract | `pass` | `weak_distractor_count`, `weak_distractor_labels`, `all_three_near_miss`, `one_axis_perturbation_count` 요구 |",
        "| split_dataset hook connected | `pass` | `merge_scores -> structured enrichment -> validator -> split_dataset/export` |",
        "| correct_choice recalculation gate | `pass` | mismatch 발생 시 `hard_block` |",
        "| label-keyed metadata remap gate | `pass` | export/audit 후보만 post-shuffle metadata remap |",
        "| export/split/count gate | `pass` | calibration pilot count는 전체 `아니오` |",
    ]
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def split_dataset_with_calibration_validator(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows:
        micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    validated_rows = apply_calibration_validator(rows)
    if validated_rows:
        micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(MERGED_SCORES_PATH, validated_rows, list(validated_rows[0].keys()))
    manifest_rows = ORIGINAL_SPLIT_DATASET(validated_rows)
    rewrite_candidate_outputs_with_status(validated_rows, manifest_rows)
    return manifest_rows


def rewrite_candidate_outputs_with_status(validated_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> None:
    selected_by_problem_id = {row["candidate_id"]: row for row in selected_rows(validated_rows)}
    for path in (PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH, PROBLEM_TEST_PATH):
        if not path.exists():
            continue
        payload_rows = micro.pb8.pb6.pb4.pb3.base.load_jsonl(path)
        enriched_payloads = []
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
                    "weak_distractor_count": source.get("weak_distractor_count", ""),
                    "weak_distractor_labels": source.get("weak_distractor_labels", ""),
                    "all_three_near_miss": source.get("all_three_near_miss", ""),
                    "one_axis_perturbation_count": source.get("one_axis_perturbation_count", ""),
                }
            )
            enriched_payloads.append(enriched)
        micro.pb8.pb6.pb4.pb3.base.write_jsonl_atomic(path, enriched_payloads)

    if manifest_rows:
        enriched_manifest_rows = []
        for row in manifest_rows:
            source = selected_by_problem_id.get(row.get("problem_id", ""), {})
            enriched = dict(row)
            enriched.update(
                {
                    "batch_status": BATCH_STATUS,
                    "count_reflection_status": COUNT_REFLECTION_STATUS,
                    "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
                    "validator_action": source.get("validator_action", ""),
                    "validator_export_disposition": source.get("validator_export_disposition", ""),
                    "target_correct_choice": source.get("target_correct_choice", ""),
                    "export_correct_choice": source.get("export_correct_choice", source.get("correct_choice", "")),
                    "weak_distractor_count": source.get("weak_distractor_count", ""),
                    "weak_distractor_labels": source.get("weak_distractor_labels", ""),
                    "all_three_near_miss": source.get("all_three_near_miss", ""),
                    "one_axis_perturbation_count": source.get("one_axis_perturbation_count", ""),
                    "count_allowed": source.get("count_allowed", "아니오"),
                }
            )
            enriched_manifest_rows.append(enriched)
        micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(
            PROBLEM_DATASET_MANIFEST_PATH,
            enriched_manifest_rows,
            list(enriched_manifest_rows[0].keys()),
        )


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = micro.pb8.pb6.pb4.pb3.summarize_rows(rows)
    selected = selected_rows(rows)
    summary_rows = [
        {"metric": "seed_count", "value": str(len(selected))},
        {"metric": "selected_pass", "value": str(summary["selected_pass_count"])},
        {"metric": "selected_hard_fail", "value": str(summary["selected_hard_fail_count"])},
        {"metric": "selected_soft_fail", "value": str(summary["selected_soft_fail_count"])},
        {"metric": "train_eligible", "value": str(summary["selected_train_eligible_count"])},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "pilot_success_passed", "value": str(VALIDATOR_SUMMARY.get("pilot_success_passed", False))},
    ]
    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(
        BATCH_LANE_SUMMARY_CSV_PATH,
        [{"sampling_lane": "generalization_03_04", "count": str(len(selected))}],
        ["sampling_lane", "count"],
    )
    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected)}`",
        "- doc_type_counts: `{'결정례_QA': 8}`",
        "- lane_counts: `{'generalization_03_04': 8}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        f"- validator_action_counts: `{VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- pilot_success_passed: `{VALIDATOR_SUMMARY.get('pilot_success_passed', False)}`",
        "",
        "## success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| usable | `>= {SUCCESS_USABLE_MIN} / 8` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| structured field missing/parse | `0` | `{VALIDATOR_SUMMARY.get('structured_missing_count', 0)}` |",
        f"| weak label consistency failure | `0` | `{VALIDATOR_SUMMARY.get('weak_label_consistency_failure_count', 0)}` |",
        f"| export-ready weak distractor | `0` | `{VALIDATOR_SUMMARY.get('export_ready_weak_distractor_count', 0)}` |",
        f"| export-ready all_three_near_miss = 아니오 | `0` | `{VALIDATOR_SUMMARY.get('export_ready_all_three_near_miss_no_count', 0)}` |",
        f"| shuffle/metadata mismatch | `0` | `shuffle {VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)} / metadata {VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |",
        "- count_reflection: `not_counted`",
    ]
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict:
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    # `VALIDATOR_SUMMARY`는 validator action 중심이라 selected pass 수를 직접 갖지 않는다.
    # reviewer 검산용 manifest에는 merged selected row에서 final status를 다시 세어 넣는다.
    selected = selected_rows(merged_rows)
    selected_final_counts = Counter(row.get("final_status", "") for row in selected)
    manifest["version_tag"] = VERSION_TAG
    manifest["run_name"] = RUN_NAME
    manifest["run_purpose"] = RUN_PURPOSE
    manifest["seed_count"] = len(seed_rows)
    manifest["judge_count"] = len(seed_rows) * 4
    manifest["selected_counts"] = {
        "pass": selected_final_counts.get("pass", 0),
        "hard_fail": selected_final_counts.get("hard_fail", 0),
        "soft_fail": selected_final_counts.get("soft_fail", 0),
    }
    manifest["train_eligible_count"] = VALIDATOR_SUMMARY.get("selected_train_eligible_count", 0)
    manifest["audit_required_count"] = VALIDATOR_SUMMARY.get("selected_audit_required_count", 0)
    manifest["count_reflection_status"] = COUNT_REFLECTION_STATUS
    manifest["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
    manifest["current_count_decision"] = "not_counted_calibration_signal_only"
    manifest["judge_structured_required_fields"] = wiring_check.JUDGE_STRUCTURED_REQUIRED_FIELDS
    manifest["validator_summary"] = VALIDATOR_SUMMARY
    manifest["success_criteria"] = {
        "usable_min": SUCCESS_USABLE_MIN,
        "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
        "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
        "audit_max": SUCCESS_AUDIT_MAX,
        "structured_missing_or_parse": 0,
        "weak_label_consistency_failure": 0,
        "export_ready_weak_distractor_count": 0,
        "export_ready_all_three_near_miss_no_count": 0,
        "shuffle_recalc_mismatch": 0,
        "metadata_remap_mismatch": 0,
        "count_reflection": COUNT_REFLECTION_STATUS,
    }
    manifest["success_result"] = {
        "usable": VALIDATOR_SUMMARY.get("selected_train_eligible_count", 0),
        "hard_fail": VALIDATOR_SUMMARY.get("selected_hard_fail_count", 0),
        "soft_fail": VALIDATOR_SUMMARY.get("selected_soft_fail_count", 0),
        "audit": VALIDATOR_SUMMARY.get("selected_audit_required_count", 0),
        "structured_missing_count": VALIDATOR_SUMMARY.get("structured_missing_count", 0),
        "weak_label_consistency_failure_count": VALIDATOR_SUMMARY.get("weak_label_consistency_failure_count", 0),
        "export_ready_weak_distractor_count": VALIDATOR_SUMMARY.get("export_ready_weak_distractor_count", 0),
        "export_ready_all_three_near_miss_no_count": VALIDATOR_SUMMARY.get(
            "export_ready_all_three_near_miss_no_count", 0
        ),
        "shuffle_recalc_mismatch_count": VALIDATOR_SUMMARY.get("shuffle_recalc_mismatch_count", 0),
        "metadata_remap_mismatch_count": VALIDATOR_SUMMARY.get("metadata_remap_mismatch_count", 0),
        "passed": bool(VALIDATOR_SUMMARY.get("pilot_success_passed", False)),
    }
    manifest["artifact_paths"].update(
        {
            "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "raw_merged_before_validator": str(RAW_MERGED_BEFORE_VALIDATOR_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
            "judge_structured_contract": str(JUDGE_STRUCTURED_CONTRACT_MD_PATH),
            "structured_field_gate": str(STRUCTURED_FIELD_GATE_MD_PATH),
        }
    )
    micro.pb8.pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_calibration_pilot_globals() -> None:
    micro.VERSION_TAG = VERSION_TAG
    micro.RUN_DATE = RUN_DATE
    micro.RUN_PURPOSE = RUN_PURPOSE
    micro.RUN_NAME = RUN_NAME
    micro.INTERIM_DIR = INTERIM_DIR
    micro.PROCESSED_DIR = PROCESSED_DIR
    micro.RUN_DIR = RUN_DIR
    micro.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    micro.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    micro.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    micro.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    micro.RUN_MERGED_DIR = RUN_MERGED_DIR
    micro.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    micro.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    micro.SEED_READY_PATH = SEED_READY_PATH
    micro.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    micro.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    micro.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    micro.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    micro.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    micro.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    micro.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    micro.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    micro.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    micro.RAW_MERGED_BEFORE_VALIDATOR_PATH = RAW_MERGED_BEFORE_VALIDATOR_PATH
    micro.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    micro.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    micro.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    micro.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    micro.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    micro.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    micro.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    micro.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    micro.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    micro.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    micro.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    micro.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    micro.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    micro.MICRO_SOURCE_COUNTS = SOURCE_COUNTS
    micro.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    micro.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    micro.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    micro.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    micro.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    micro.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    micro.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    micro.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    micro.collect_excluded_rows_for_micro = collect_excluded_rows_for_calibration
    micro.build_generation_messages = build_generation_messages
    micro.build_batch_summary = build_batch_summary
    micro.build_run_manifest = build_run_manifest
    micro.split_dataset_with_validator = split_dataset_with_calibration_validator
    micro.write_validator_wiring_check_md = write_validator_wiring_check_md
    micro.pb8.pb6.build_tail_memo = build_tail_memo_with_validator_fields


def main() -> dict:
    configure_calibration_pilot_globals()
    micro.configure_micro_globals()
    micro.pb8.pb6.RUN_LABEL = "pb9 04TL decision weak distractor calibration pilot"
    micro.pb8.pb6.SEED_ID_PREFIX = "pb9_04tl_calibration"
    micro.pb8.pb6.SEED_SELECTION_ROLE = "objective_pb9_04tl_decision_weak_distractor_calibration_seed"
    micro.pb8.pb6.SEED_SELECTION_NOTE = "04_TL_결정례_QA generalization_03_04 weak distractor calibration pilot seed"
    micro.pb8.pb6.SEED_FILTER_NOTE = "pb9_04tl_seen_seed_pool_excluded"
    micro.pb8.pb6.SCOPE_NOTE = "04_TL_결정례_QA generalization_03_04 only; calibration signal only; current count 미합산"
    micro.pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = 8
    micro.pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_pb9_04tl_calibration"
    micro.pb8.pb6.SEED_REGISTRY_STRATEGY = "fresh_04tl_decision_generalization_pool_excluding_current_failed_repair_and_pb9_seen_seed"
    micro.pb8.pb6.LAW_STATUS_NOTE = "04tl_decision_calibration_not_counted"
    micro.pb8.pb6.OVERLAP_CHECK_LABEL = "no current/failed/repaired/pb9-seen seed overlap"
    micro.pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`pb9` actual API execution seed까지 seen seed로 제외한다.",
        "이번 run은 `04_TL_결정례_QA generalization_03_04` `8개` calibration signal 전용이며 current count에는 합산하지 않는다.",
    ]
    micro.pb8.pb6.pb4.pb3.base.build_judge_prompt = build_structured_judge_prompt
    micro.pb8.pb6.pb4.pb3.base.build_judge_row = build_structured_judge_row
    write_judge_structured_contract_md()
    return micro.pb8.pb6.main()


if __name__ == "__main__":
    main()
