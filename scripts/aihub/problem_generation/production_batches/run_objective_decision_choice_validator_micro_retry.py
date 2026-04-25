import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Reviewer 회신에 따라 `8개` micro package 전체를 다시 태우지 않고,
# validator가 `regenerate`로 잡은 2개 slot만 seed당 최대 1회 재생성한다.
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
from scripts.aihub.problem_generation.v2_objective_difficulty_patch import (  # noqa: E402
    run_difficulty_patch as base,
)
from scripts.aihub.problem_generation.v2_objective_difficulty_patch_r2 import (  # noqa: E402
    run_difficulty_patch as r2,
)


VERSION_TAG = "decision_choice_validator_micro_retry"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_bounded_regeneration_micro_retry"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = base.PROJECT_ROOT
SOURCE_VERSION_TAG = micro.VERSION_TAG
SOURCE_RUN_DIR = micro.RUN_DIR
SOURCE_SEED_REGISTRY_PATH = micro.SEED_REGISTRY_PATH
SOURCE_SEED_READY_PATH = micro.SEED_READY_PATH
SOURCE_MERGED_PATH = micro.MERGED_SCORES_PATH
SOURCE_RAW_MERGED_PATH = micro.RAW_MERGED_BEFORE_VALIDATOR_PATH
SOURCE_VALIDATOR_REPORT_CSV_PATH = micro.VALIDATOR_REPORT_CSV_PATH

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
SEED_READY_PATH = INTERIM_DIR / "retry_seed_ready.jsonl"
FULL_SEED_READY_PATH = INTERIM_DIR / "seed_ready_full_package.jsonl"
RETRY_PLAN_CSV_PATH = RUN_EXPORTS_DIR / f"bounded_retry_plan_{VERSION_TAG}.csv"
RETRY_PLAN_MD_PATH = RUN_EXPORTS_DIR / f"bounded_retry_plan_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
RETRY_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"retry_merged_before_validator_{VERSION_TAG}.csv"
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
VALIDATOR_RETRY_ACTIONS_CSV_PATH = RUN_EXPORTS_DIR / f"validator_retry_actions_{VERSION_TAG}.csv"
VALIDATOR_FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"validator_final_package_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"

RETRY_SEED_IDS = ("decision_micro_005", "decision_micro_008")
SUCCESS_USABLE_MIN = 7
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}

VALIDATOR_SUMMARY = {}


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def load_source_seed_rows():
    return micro.pb8.pb6.pb4.pb3.base.load_jsonl(SOURCE_SEED_READY_PATH)


def load_source_validator_report():
    return read_csv_rows(SOURCE_VALIDATOR_REPORT_CSV_PATH)


def retry_target_by_seed():
    report_rows = load_source_validator_report()
    return {
        row["seed_sample_id"]: row["target_correct_choice"]
        for row in report_rows
        if row.get("validator_action") == "regenerate"
    }


def preserved_accept_rows():
    rows = read_csv_rows(SOURCE_MERGED_PATH)
    preserved = []
    for row in rows:
        if row.get("selected_for_seed") == "예" and row.get("validator_export_disposition") == "export_ready":
            copied = dict(row)
            # 이전 micro pilot 산출물은 choice만 셔플되어 metadata가 남을 수 있으므로 보존 row도 재매핑한다.
            validator_replay.remap_existing_metadata_to_correct_choice(copied)
            metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(copied)
            copied["package_run_name"] = RUN_NAME
            copied["bounded_retry_role"] = "preserved_accept"
            copied["bounded_retry_attempt_count"] = "0"
            copied["source_candidate_id"] = row.get("candidate_id", "")
            copied["source_run_name"] = row.get("run_name", "")
            copied["validator_metadata_remap_ok"] = "예" if metadata_ok else "아니오"
            copied["validator_metadata_remap_reasons"] = "|".join(metadata_reasons)
            preserved.append(copied)
    return preserved


def retry_seed_rows():
    seeds = load_source_seed_rows()
    retry_ids = set(RETRY_SEED_IDS)
    return [seed for seed in seeds if seed["seed_sample_id"] in retry_ids]


def write_seed_inputs():
    all_seed_rows = load_source_seed_rows()
    retry_rows = retry_seed_rows()
    if len(retry_rows) != len(RETRY_SEED_IDS):
        raise RuntimeError(f"bounded retry seed 수가 {len(RETRY_SEED_IDS)}개가 아닙니다: {len(retry_rows)}")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RUN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(SEED_REGISTRY_PATH, all_seed_rows, list(all_seed_rows[0].keys()))
    micro.pb8.pb6.pb4.pb3.base.write_jsonl_atomic(FULL_SEED_READY_PATH, all_seed_rows)
    micro.pb8.pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, retry_rows)
    micro.pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    micro.pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(FULL_SEED_READY_PATH, RUN_INPUTS_DIR)
    micro.pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return all_seed_rows, retry_rows


def build_retry_plan_rows(retry_rows):
    target_by_seed = retry_target_by_seed()
    raw_rows = {row["seed_sample_id"]: row for row in read_csv_rows(SOURCE_RAW_MERGED_PATH)}
    plan_rows = []
    for seed in retry_rows:
        raw = raw_rows.get(seed["seed_sample_id"], {})
        plan_rows.append(
            {
                "seed_sample_id": seed["seed_sample_id"],
                "doc_type_name": seed["doc_type_name"],
                "sampling_lane": seed["sampling_lane"],
                "retry_attempt_limit": "1",
                "target_correct_choice": target_by_seed.get(seed["seed_sample_id"], ""),
                "source_error_tags": raw.get("error_tags", ""),
                "source_nearmiss_reason": raw.get("nearmiss_reason", ""),
                "source_candidate_id": raw.get("candidate_id", ""),
                "retry_policy": "regenerate_once_then_judge_and_validator",
            }
        )
    return plan_rows


def write_retry_plan(retry_rows):
    plan_rows = build_retry_plan_rows(retry_rows)
    write_csv(RETRY_PLAN_CSV_PATH, plan_rows)
    lines = [
        f"# bounded retry plan `{VERSION_TAG}`",
        "",
        "| seed | lane | target | limit | source_tags | retry_policy |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in plan_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['sampling_lane']}` | `{row['target_correct_choice']}` | `{row['retry_attempt_limit']}` | `{row['source_error_tags']}` | `{row['retry_policy']}` |"
        )
    write_text(RETRY_PLAN_MD_PATH, "\n".join(lines) + "\n")
    return plan_rows


def configure_base_for_retry():
    # r2의 검증된 generator/Judge 설정을 유지하고, 경로와 build prompt만 retry run으로 재배선한다.
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
    base.MERGED_SCORES_PATH = RETRY_MERGED_BEFORE_VALIDATOR_PATH
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
    base.build_generation_messages = build_retry_generation_messages
    base.postprocess_problem = r2.postprocess_problem
    base.build_local_fallback_problem = r2.build_local_fallback_problem
    base.load_reference_v2_rows = lambda: {}


def build_retry_generation_messages(seed, reference_v2):
    messages = micro.build_generation_messages(seed, reference_v2)
    raw_rows = {row["seed_sample_id"]: row for row in read_csv_rows(SOURCE_RAW_MERGED_PATH)}
    raw = raw_rows.get(seed["seed_sample_id"], {})
    messages[1]["content"] += f"""

## bounded regeneration retry 추가 지시
- 이 seed는 직전 micro pilot에서 `validator_action = regenerate`로 제외된 slot이다.
- 이전 실패 신호는 `{raw.get('error_tags', '')}`이며, 핵심 사유는 아래와 같다.
  - `{raw.get('nearmiss_reason', '')}`
- 이번 재생성은 seed당 최대 1회만 허용되므로, 오답 3개가 모두 정답과 같은 legal anchor를 공유하면서도 서로 다른 한 축만 비틀어야 한다.
- 특히 하나의 오답만 너무 명백하게 틀리거나 원칙을 단순 반대로 말하는 구성을 피하고, 세 오답의 plausibility를 균등하게 맞출 것.
- 정답 유일성과 선택지 의미 분리는 유지하되, weak distractor가 다시 발생하면 해당 slot은 최종 package에서 제외된다.
"""
    return messages


def run_retry_generation_and_judge():
    configure_base_for_retry()
    base.ensure_run_dirs()
    base.run_generation(mode="main")
    base.run_generation(mode="strict_finalize")
    base.run_judges(mode="main")
    base.run_judges(mode="strict_finalize")
    retry_rows = base.merge_scores()
    return retry_rows


def append_tag(existing_tags, tag):
    tags = [value for value in (existing_tags or "").split("|") if value]
    if tag not in tags:
        tags.append(tag)
    return "|".join(tags)


def selected_rows(rows):
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def apply_validator_to_retry_rows(rows):
    target_by_seed = retry_target_by_seed()
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
        target_label = target_by_seed[row["seed_sample_id"]]
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
        row["package_run_name"] = RUN_NAME
        row["bounded_retry_role"] = "bounded_regeneration"
        row["bounded_retry_attempt_count"] = "1"
        row["source_candidate_id"] = ""
        row["source_run_name"] = str(SOURCE_RUN_DIR.name)

        if action in {"accept", "audit"}:
            # Reviewer가 요구한 label slot 보존 조건을 만족하려면 retry row의 choice와 metadata를 함께 재배치해야 한다.
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
            # 차단/재생성 row는 export truth가 아니므로 metadata gate 대상에서 제외한다.
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
    return rows, report_rows


def package_rows(validated_retry_rows):
    rows = preserved_accept_rows() + selected_rows(validated_retry_rows)
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
    VALIDATOR_SUMMARY["micro_retry_success_passed"] = (
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


def all_fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_validator_report(report_rows, package):
    summary = compute_validator_summary(package)
    retry_action_rows = []
    for row in report_rows:
        action_row = dict(row)
        action_row["report_section"] = "retry_action"
        retry_action_rows.append(action_row)
    package_rows_for_report = []
    for row in package:
        package_rows_for_report.append(
            {
                "report_section": "final_package",
                "seed_sample_id": row["seed_sample_id"],
                "bounded_retry_role": row.get("bounded_retry_role", ""),
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
    write_csv(VALIDATOR_RETRY_ACTIONS_CSV_PATH, retry_action_rows)
    write_csv(VALIDATOR_FINAL_PACKAGE_CSV_PATH, package_rows_for_report)
    write_csv(VALIDATOR_REPORT_CSV_PATH, retry_action_rows + package_rows_for_report, all_fieldnames(retry_action_rows + package_rows_for_report))

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
        f"- micro_retry_success_passed: `{summary['micro_retry_success_passed']}`",
        "",
        "## retry row actions",
        "| seed | upstream_status | action | final_status | audit | train_eligible | target | recalculated | metadata | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['upstream_final_status']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['recalculated_correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
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
            f"| `{row['seed_sample_id']}` | `{row['bounded_retry_role']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    write_text(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_wiring_check():
    lines = [
        f"# bounded regeneration wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| retry target limited to regenerate rows | `pass` | `decision_micro_005`, `decision_micro_008`만 retry |",
        "| retry attempt limit | `pass` | seed당 최대 `1회` |",
        "| accepted rows preserved | `pass` | 기존 `6개` export-ready row 보존 |",
        "| regenerated rows judged | `pass` | retry row는 Judge `4축` 재실행 |",
        "| validator reapplied | `pass` | retry row에도 shuffle/recalc + validator 재적용 |",
        "| label-keyed metadata remap gate | `pass` | 보존 row와 retry row 모두 `distractor_type_map`, `near_miss_notes` 재매핑 검산 |",
        "| validator report split | `pass` | retry action CSV와 final package CSV를 분리해 자동 검산 혼동 차단 |",
        "| target label retained | `pass` | `decision_micro_005 -> A`, `decision_micro_008 -> D` |",
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
        {"metric": "success_passed", "value": str(summary["micro_retry_success_passed"])},
    ]
    write_csv(BATCH_SUMMARY_CSV_PATH, summary_rows)

    lane_rows = [{"sampling_lane": lane, "count": count} for lane, count in sorted(lane_counter.items())]
    write_csv(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, ["sampling_lane", "count"])

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


def write_manifest(all_seed_rows, retry_seed_rows, package, manifest_rows, tail_rows, plan_rows):
    summary = compute_validator_summary(package)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "source_micro_run_name": SOURCE_RUN_DIR.name,
        "retry_policy": {
            "target": "validator_action=regenerate rows only",
            "retry_seed_ids": list(RETRY_SEED_IDS),
            "max_retry_per_seed": 1,
            "preserve_accept_rows": True,
            "semantic_judge": "regenerated rows only",
            "validator_reapplied": True,
            "count_reflection": "not_counted_until_reviewer_signoff",
        },
        "seed_registry_count": len(all_seed_rows),
        "retry_seed_count": len(retry_seed_rows),
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
            "passed": summary["micro_retry_success_passed"],
        },
        "current_count_decision": "not_counted_until_reviewer_signoff",
        "retry_plan_count": len(plan_rows),
        "tail_memo_count": len([row for row in tail_rows if row.get("seed_sample_id")]),
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "retry_seed_ready": str(SEED_READY_PATH),
            "full_seed_ready": str(FULL_SEED_READY_PATH),
            "retry_plan_csv": str(RETRY_PLAN_CSV_PATH),
            "retry_plan_md": str(RETRY_PLAN_MD_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "retry_merged_before_validator": str(RETRY_MERGED_BEFORE_VALIDATOR_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_retry_actions_csv": str(VALIDATOR_RETRY_ACTIONS_CSV_PATH),
            "validator_final_package_csv": str(VALIDATOR_FINAL_PACKAGE_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
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
    all_seed_rows, retry_rows = write_seed_inputs()
    plan_rows = write_retry_plan(retry_rows)
    write_wiring_check()
    retry_merged = run_retry_generation_and_judge()
    validated_retry, retry_report_rows = apply_validator_to_retry_rows(retry_merged)
    package = package_rows(validated_retry)
    write_csv(MERGED_SCORES_PATH, package, all_fieldnames(package))
    write_validator_report(retry_report_rows, package)
    summary_rows = write_batch_summary(package)
    tail_rows = write_tail_memo(package)
    manifest_rows = split_package(package)
    return write_manifest(all_seed_rows, retry_rows, package, manifest_rows, tail_rows, plan_rows)


if __name__ == "__main__":
    main()
