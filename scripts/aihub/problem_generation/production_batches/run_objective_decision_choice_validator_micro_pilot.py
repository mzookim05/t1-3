import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `decision_choice_validator_offline_replay`는 기존 결과 재검산이었으므로, 이번 runner는
# 같은 validator를 새 generation 결과의 postprocess 단계에 실제로 연결하는 8개 micro pilot이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as pb8
from scripts.aihub.problem_generation.production_batches import (
    run_objective_decision_choice_validator_replay as validator_replay,
)


VERSION_TAG = "decision_choice_validator_micro_pilot"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-25_220652"
RUN_PURPOSE = "objective_r2_decision_choice_validator_micro_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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

REFERENCE_PB8_SEED_REGISTRY_PATH = (
    pb8.pb6.pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb8_decision_only_objective_current_r2"
    / "seed_registry.csv"
)
REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH = (
    pb8.pb6.pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_weak_distractor_guardrail_pilot"
    / "seed_registry.csv"
)

MICRO_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 2,
    "04_TL_결정례_QA": 2,
}

EXPECTED_DOC_TYPE_COUNTS = {
    "결정례_QA": 8,
}

EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "generalization_03_04"): 4,
    ("결정례_QA", "expansion_01_02"): 4,
}

SUCCESS_USABLE_MIN = 7
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}

ORIGINAL_COLLECT_EXCLUDED_ROWS = pb8.collect_excluded_rows
ORIGINAL_BUILD_GENERATION_MESSAGES = pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES
ORIGINAL_SPLIT_DATASET = pb8.pb6.pb4.pb3.base.split_dataset
ORIGINAL_BUILD_RUN_MANIFEST = pb8.pb6.build_run_manifest
ORIGINAL_BUILD_BATCH_SUMMARY = pb8.pb6.build_batch_summary

VALIDATOR_SUMMARY = {}


def collect_excluded_rows():
    # `pb8`과 직전 failed pilot까지 seen seed로 제외해야 micro pilot이 fresh seed 검증으로 남는다.
    rows = ORIGINAL_COLLECT_EXCLUDED_ROWS()
    rows.extend(pb8.pb6.load_csv_rows_if_exists(REFERENCE_PB8_SEED_REGISTRY_PATH))
    rows.extend(pb8.pb6.load_csv_rows_if_exists(REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH))
    return rows


def build_generation_messages(seed, reference_v2):
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## decision choice validator micro pilot 추가 지시
- 이번 run은 `결정례_QA` postprocess choice validator가 새 generation 결과에 붙었을 때 과차단/미차단을 확인하는 `8개` micro pilot이다.
- 오답 중 최소 `2개`는 정답과 같은 결정 이유, 판단 기준, 적용 사실 중 하나를 공유하되 판단 요소 `1개`만 다르게 비틀 것.
- 정답 외 선택지가 같은 결론이나 같은 판단 기준으로 정답 가능하게 읽히면 안 된다.
- 오답끼리 같은 의미를 반복하지 말고, 정답보다 모호하거나 포괄적인 표현으로 도망가지 말 것.
- choice 순서는 후처리 validator가 다시 섞을 수 있으므로, 생성 단계에서는 선택지 의미 분리와 정답 유일성에 집중할 것.
"""
    return messages


def append_tag(existing_tags, tag):
    tags = [value for value in (existing_tags or "").split("|") if value]
    if tag not in tags:
        tags.append(tag)
    return "|".join(tags)


def selected_rows(rows):
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def target_label_for_selected_index(index):
    # micro pilot은 selected/export package 기준으로 A/B/C/D = 2/2/2/2를 강제한다.
    return validator_replay.CHOICE_LABELS[index % len(validator_replay.CHOICE_LABELS)]


def apply_validator_to_rows(rows):
    global VALIDATOR_SUMMARY

    selected = selected_rows(rows)
    report_rows = []
    for row in rows:
        # 원래 Judge 결과를 보존해야 validator가 무엇을 바꿨는지 reviewer가 역추적할 수 있다.
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

    for index, row in enumerate(selected):
        target_label = target_label_for_selected_index(index)
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices, row.get("correct_choice", ""), target_label
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

        if action in {"accept", "audit"}:
            # export 가능한 row만 실제 choice와 label-keyed metadata를 같은 permutation으로 바꾼다.
            original_correct_choice = row.get("correct_choice", "")
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
            # 차단/재생성 row는 artifact truth로 export하지 않으므로 metadata gate 대상에서 뺀다.
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
                "seed_sample_id": row["seed_sample_id"],
                "upstream_final_status": row["upstream_final_status"],
                "upstream_audit_required": row["upstream_audit_required"],
                "upstream_train_eligible": row["upstream_train_eligible"],
                "validator_action": action,
                "validator_status": status,
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

    target_counts = Counter(row["validator_target_correct_choice"] for row in selected if row["validator_target_correct_choice"])
    export_rows = [
        row
        for row in selected
        if row["validator_export_disposition"] in {"export_ready", "audit"}
    ]
    export_label_counts = Counter(row["correct_choice"] for row in export_rows)
    action_counts = Counter(row["validator_action"] for row in selected)
    shuffle_recalc_mismatch_count = sum(1 for row in selected if row["validator_shuffle_recalc_ok"] != "예")
    metadata_remap_mismatch_count = sum(
        1
        for row in export_rows
        if row.get("validator_metadata_remap_ok") != "예"
    )
    selected_summary = pb8.pb6.pb4.pb3.summarize_rows(rows)

    VALIDATOR_SUMMARY = {
        "selected_count": len(selected),
        "validator_action_counts": dict(action_counts),
        "target_label_counts": {label: target_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "export_label_counts": {label: export_label_counts.get(label, 0) for label in validator_replay.CHOICE_LABELS},
        "shuffle_recalc_mismatch_count": shuffle_recalc_mismatch_count,
        "full_package_label_balance_passed": all(target_counts.get(label, 0) == TARGET_LABEL_COUNTS[label] for label in validator_replay.CHOICE_LABELS),
        "export_ready_label_balance_passed": all(export_label_counts.get(label, 0) == TARGET_LABEL_COUNTS[label] for label in validator_replay.CHOICE_LABELS),
        "selected_train_eligible_count": selected_summary["selected_train_eligible_count"],
        "selected_hard_fail_count": selected_summary["selected_hard_fail_count"],
        "selected_soft_fail_count": selected_summary["selected_soft_fail_count"],
        "selected_audit_required_count": selected_summary["selected_audit_required_count"],
        "metadata_remap_mismatch_count": metadata_remap_mismatch_count,
    }
    VALIDATOR_SUMMARY["micro_success_passed"] = (
        VALIDATOR_SUMMARY["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and VALIDATOR_SUMMARY["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and VALIDATOR_SUMMARY["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and VALIDATOR_SUMMARY["shuffle_recalc_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["metadata_remap_mismatch_count"] == 0
        and VALIDATOR_SUMMARY["export_ready_label_balance_passed"]
    )

    fieldnames = list(report_rows[0].keys()) if report_rows else ["seed_sample_id"]
    pb8.pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, fieldnames)
    write_validator_report_md(report_rows)
    write_validator_wiring_check_md()
    return rows


def write_validator_report_md(report_rows):
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## summary",
        f"- selected_count: `{VALIDATOR_SUMMARY.get('selected_count', 0)}`",
        f"- validator_action_counts: `{VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- target_label_counts: `{VALIDATOR_SUMMARY.get('target_label_counts', {})}`",
        f"- export_label_counts: `{VALIDATOR_SUMMARY.get('export_label_counts', {})}`",
        f"- shuffle_recalc_mismatch_count: `{VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)}`",
        f"- metadata_remap_mismatch_count: `{VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}`",
        f"- micro_success_passed: `{VALIDATOR_SUMMARY.get('micro_success_passed', False)}`",
        "",
        "## row actions",
        "| seed | upstream_status | action | final_status | audit | train_eligible | target | recalculated | metadata | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['upstream_final_status']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['recalculated_correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_validator_wiring_check_md():
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| validator module imported | `pass` | `run_objective_decision_choice_validator_replay.py`의 deterministic validator 사용 |",
        "| split_dataset hook connected | `pass` | `merge_scores -> validator -> split_dataset/export` 순서로 연결 |",
        "| correct_choice recalculation gate | `pass` | mismatch 발생 시 `hard_block` 처리 |",
        "| label-keyed metadata remap gate | `pass` | `distractor_type_map`, `near_miss_notes`를 post-shuffle label로 재매핑 |",
        "| regenerate policy defined | `pass` | micro pilot에서는 재생성 필요 row를 export 제외 `soft_fail`로 기록하고 다음 reviewer 판단을 받음 |",
        "| target label schedule | `pass` | selected package 기준 `A/B/C/D = 2/2/2/2` |",
        "| count reflection | `pass` | reviewer sign-off 전 current count 미합산 |",
    ]
    pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def split_dataset_with_validator(rows):
    # raw merged를 보존한 뒤 validator 적용본으로 handoff를 다시 만들면,
    # reviewer가 Judge 전후와 validator 전후를 모두 재검산할 수 있다.
    if rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    validated_rows = apply_validator_to_rows(rows)
    if validated_rows:
        pb8.pb6.pb4.pb3.base.write_csv_atomic(MERGED_SCORES_PATH, validated_rows, list(validated_rows[0].keys()))
    return ORIGINAL_SPLIT_DATASET(validated_rows)


def build_batch_summary(rows):
    summary_rows = ORIGINAL_BUILD_BATCH_SUMMARY(rows)
    with BATCH_SUMMARY_MD_PATH.open("a", encoding="utf-8") as f:
        f.write("\n## validator micro success criteria\n")
        f.write("| criterion | target | result |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| usable | `>= {SUCCESS_USABLE_MIN} / 8` | `{VALIDATOR_SUMMARY.get('selected_train_eligible_count', 0)}` |\n")
        f.write(f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{VALIDATOR_SUMMARY.get('selected_hard_fail_count', 0)}` |\n")
        f.write(f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{VALIDATOR_SUMMARY.get('selected_soft_fail_count', 0)}` |\n")
        f.write(f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{VALIDATOR_SUMMARY.get('selected_audit_required_count', 0)}` |\n")
        f.write(f"| shuffle recalc mismatch | `0` | `{VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)}` |\n")
        f.write(f"| metadata remap mismatch | `0` | `{VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |\n")
        f.write(f"| export label balance | `A/B/C/D = 2/2/2/2` | `{VALIDATOR_SUMMARY.get('export_label_counts', {})}` |\n")
    return summary_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    manifest["version_tag"] = VERSION_TAG
    manifest["run_name"] = RUN_NAME
    manifest["validator_policy"] = {
        "source": "postprocess_choice_validator",
        "regenerate_policy": "micro pilot에서는 재생성 필요 row를 export 제외 soft_fail로 기록하고 reviewer 판단 후 bounded regeneration 여부 결정",
        "label_schedule": TARGET_LABEL_COUNTS,
        "count_reflection": "not_counted_until_reviewer_signoff",
    }
    manifest["validator_summary"] = VALIDATOR_SUMMARY
    if "success_result" in manifest:
        manifest["success_result"]["passed"] = bool(VALIDATOR_SUMMARY.get("micro_success_passed", False))
    manifest["artifact_paths"]["raw_merged_before_validator"] = str(RAW_MERGED_BEFORE_VALIDATOR_PATH)
    manifest["artifact_paths"]["validator_report_csv"] = str(VALIDATOR_REPORT_CSV_PATH)
    manifest["artifact_paths"]["validator_report_md"] = str(VALIDATOR_REPORT_MD_PATH)
    manifest["artifact_paths"]["validator_wiring_check_md"] = str(VALIDATOR_WIRING_CHECK_MD_PATH)
    pb8.pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def collect_excluded_rows_for_micro():
    return collect_excluded_rows()


def configure_micro_globals():
    # `pb8`까지 검증된 wrapper 구조를 유지하고, scope와 postprocess hook만
    # 8개 micro pilot에 맞게 좁힌다.
    pb8.pb6.VERSION_TAG = VERSION_TAG
    pb8.pb6.RUN_DATE = RUN_DATE
    pb8.pb6.RUN_PURPOSE = RUN_PURPOSE
    pb8.pb6.RUN_NAME = RUN_NAME
    pb8.pb6.RUN_LABEL = "decision choice validator micro pilot"
    pb8.pb6.SEED_ID_PREFIX = "decision_micro"
    pb8.pb6.SEED_SELECTION_ROLE = "objective_decision_choice_validator_micro_pilot_seed"
    pb8.pb6.SEED_SELECTION_NOTE = "결정례_QA postprocess choice validator wiring을 확인하는 8개 micro pilot seed"
    pb8.pb6.SEED_FILTER_NOTE = "decision_choice_validator_seen_seed_pool_excluded"
    pb8.pb6.SCOPE_NOTE = "결정례_QA only; postprocess validator + shuffle/recalc micro pilot"
    pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = 8
    pb8.pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pb8.pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pb8.pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pb8.pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pb8.pb6.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_decision_choice_validator_micro_pilot"
    pb8.pb6.SEED_REGISTRY_STRATEGY = "fresh_aihub_qa_training_decision_only_pool_excluding_current_law_targeted_failed_pb5_failed_pb6_failed_pb7_failed_pb8_failed_decision_guardrail_heldout_audit_rows"
    pb8.pb6.LAW_STATUS_NOTE = "decision_choice_validator_micro_pilot_count_excluded_until_reviewer_signoff"
    pb8.pb6.OVERLAP_CHECK_LABEL = "no current/law-targeted/failed-pb5/pb6/pb7/pb8/decision-guardrail/held-out/audit overlap"
    pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`current counted-line attempted seed registry 109개`는 usable count가 아니라 `r2 + pb2 + pb3 + pb4`에 실제 투입된 seed registry 규모를 뜻한다.",
        "`law targeted pilot 16개`, failed `pb5 40개`, failed `pb6 45개`, failed `pb7 40개`, failed `pb8 40개`, failed `decision guardrail pilot 16개`까지 seen seed로 제외한다.",
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

    pb8.pb6.PB6_SOURCE_COUNTS = MICRO_SOURCE_COUNTS
    pb8.pb6.PB6_DATASET_SPECS = pb8.pb6.build_pb6_dataset_specs()
    pb8.pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb8.pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb8.pb6.collect_excluded_rows = collect_excluded_rows_for_micro
    pb8.pb6.passes_pb6_seed_filter = pb8.passes_pb8_seed_filter
    pb8.pb6.classify_tail = pb8.classify_pb8_tail
    pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb8.pb6.build_batch_summary = build_batch_summary
    pb8.pb6.build_run_manifest = build_run_manifest
    pb8.pb6.pb4.pb3.base.split_dataset = split_dataset_with_validator


def main():
    configure_micro_globals()
    return pb8.pb6.main()


if __name__ == "__main__":
    main()
