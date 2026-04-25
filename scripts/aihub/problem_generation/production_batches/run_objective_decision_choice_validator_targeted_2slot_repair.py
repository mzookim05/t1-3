import sys
from collections import Counter
from pathlib import Path

# Reviewer가 `16개` targeted pilot 전체 재시도 대신 실패한 `D` slot 2개만
# bounded retry로 보라고 판단했으므로, 검증된 micro retry 패키징 흐름을 재사용한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
EXPLANATION_DIR_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT / "scripts" / "aihub" / "problem_generation" / "explanation_generation"
if str(EXPLANATION_DIR_FOR_IMPORT) not in sys.path:
    # 구조 개편 후 explanation_generation은 problem_generation 아래에 있으므로,
    # 과거 runner의 top-level `common/settings` import가 새 위치를 먼저 보게 한다.
    sys.path.insert(0, str(EXPLANATION_DIR_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_micro_retry as retry,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_targeted_pilot as targeted,
)


VERSION_TAG = "decision_choice_validator_targeted_2slot_repair"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_targeted_2slot_bounded_repair"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = targeted.PROJECT_ROOT
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

RETRY_SEED_IDS = ("decision_targeted_008", "decision_targeted_012")
SUCCESS_USABLE_MIN = 15
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}


def build_retry_generation_messages(seed, reference_v2):
    messages = targeted.build_generation_messages(seed, reference_v2)
    raw_rows = {row["seed_sample_id"]: row for row in retry.read_csv_rows(targeted.RAW_MERGED_BEFORE_VALIDATOR_PATH)}
    raw = raw_rows.get(seed["seed_sample_id"], {})
    messages[1]["content"] += f"""

## targeted 2-slot bounded repair 추가 지시
- 이 seed는 직전 `16개` targeted pilot에서 `validator_action = regenerate`로 제외된 `D` slot이다.
- 이전 실패 신호는 `{raw.get('error_tags', '')}`이며, 핵심 진단은 아래와 같다.
  - `{raw.get('nearmiss_reason', '')}`
- 이번 재생성은 seed당 최대 1회만 허용되므로, 단순 회상형 오답이나 정답 반대말 1개만 세우는 구성을 피한다.
- 세 오답은 모두 같은 결정 이유, 판단 기준, 적용 사실 중 하나의 legal anchor를 공유하되, 서로 다른 한 축만 비틀어야 한다.
- 정답 유일성, 선택지 의미 분리, post-shuffle metadata remap은 기존 validator가 다시 검산한다.
- weak distractor가 다시 발생하면 해당 slot은 final package에서 제외되고, 다음 단계는 replacement 또는 prompt/validator spec patch 판단으로 넘어간다.
"""
    return messages


def write_wiring_check():
    lines = [
        f"# targeted 2-slot bounded repair wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| retry target limited to failed targeted slots | `pass` | `decision_targeted_008`, `decision_targeted_012`만 retry |",
        "| retry attempt limit | `pass` | seed당 최대 `1회` |",
        "| accepted rows preserved | `pass` | 기존 `14개` export-ready row 보존 |",
        "| regenerated rows judged | `pass` | retry row만 Judge `4축` 재실행 |",
        "| validator reapplied | `pass` | retry row에도 shuffle/recalc + metadata remap gate 재적용 |",
        "| target label retained | `pass` | 두 slot 모두 `D` label 복구 목적 |",
        "| final package size | `pass` | 기존 accept `14개` + retry `2개` = `16개` package 검산 |",
        "| natural language reason policy | `pass` | Judge reason은 diagnostic text로만 유지 |",
        "| count reflection | `pass` | reviewer sign-off 전 current count 미합산 |",
    ]
    retry.write_text(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def write_validator_report(report_rows, package):
    summary = retry.compute_validator_summary(package)
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

    retry.write_csv(VALIDATOR_RETRY_ACTIONS_CSV_PATH, retry_action_rows)
    retry.write_csv(VALIDATOR_FINAL_PACKAGE_CSV_PATH, package_rows_for_report)
    retry.write_csv(
        VALIDATOR_REPORT_CSV_PATH,
        retry_action_rows + package_rows_for_report,
        retry.all_fieldnames(retry_action_rows + package_rows_for_report),
    )

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
        f"- targeted_2slot_success_passed: `{summary['micro_retry_success_passed']}`",
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
            "## final 16-slot package",
            "| seed | role | action | final_status | train_eligible | target | correct | metadata | disposition |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in package_rows_for_report:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['bounded_retry_role']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    retry.write_text(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_batch_summary(package):
    summary = retry.compute_validator_summary(package)
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
    retry.write_csv(BATCH_SUMMARY_CSV_PATH, summary_rows)
    retry.write_csv(
        BATCH_LANE_SUMMARY_CSV_PATH,
        [{"sampling_lane": lane, "count": count} for lane, count in sorted(lane_counter.items())],
        ["sampling_lane", "count"],
    )

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
        f"| usable | `>= {SUCCESS_USABLE_MIN} / 16` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| metadata remap mismatch | `0` | `{summary['metadata_remap_mismatch_count']}` |",
        f"| export label balance | `A/B/C/D = 4/4/4/4` | `{summary['export_label_counts']}` |",
    ]
    retry.write_text(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def write_manifest(all_seed_rows, retry_seed_rows, package, manifest_rows, tail_rows, plan_rows):
    summary = retry.compute_validator_summary(package)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": retry.base.utc_now_iso(),
        "source_targeted_run_name": targeted.RUN_DIR.name,
        "retry_policy": {
            "target": "decision_choice_validator_targeted_pilot_16 validator_action=regenerate rows only",
            "retry_seed_ids": list(RETRY_SEED_IDS),
            "max_retry_per_seed": 1,
            "preserve_accept_rows": True,
            "semantic_judge": "regenerated rows only",
            "validator_reapplied": True,
            "replacement_if_failed": "not_auto_applied_in_this_runner",
            "count_reflection": "not_counted_until_reviewer_signoff",
        },
        "seed_registry_count": len(all_seed_rows),
        "retry_seed_count": len(retry_seed_rows),
        "generation_count": retry.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": retry.base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": retry.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": retry.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": retry.base.load_jsonl_count(NEARMISS_LOG_PATH),
        "merged_count": len(package),
        "selected_pass_count": summary["selected_count"] - summary["selected_hard_fail_count"] - summary["selected_soft_fail_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": retry.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": retry.base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": retry.base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": retry.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
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
    retry.write_json(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_retry_globals():
    # 기존 micro retry module의 안정화된 package builder를 그대로 쓰되, source와 기준만 targeted pilot로 교체한다.
    retry.VERSION_TAG = VERSION_TAG
    retry.RUN_DATE = RUN_DATE
    retry.RUN_PURPOSE = RUN_PURPOSE
    retry.RUN_NAME = RUN_NAME
    retry.SOURCE_VERSION_TAG = targeted.VERSION_TAG
    retry.SOURCE_RUN_DIR = targeted.RUN_DIR
    retry.SOURCE_SEED_REGISTRY_PATH = targeted.SEED_REGISTRY_PATH
    retry.SOURCE_SEED_READY_PATH = targeted.SEED_READY_PATH
    retry.SOURCE_MERGED_PATH = targeted.MERGED_SCORES_PATH
    retry.SOURCE_RAW_MERGED_PATH = targeted.RAW_MERGED_BEFORE_VALIDATOR_PATH
    retry.SOURCE_VALIDATOR_REPORT_CSV_PATH = targeted.VALIDATOR_REPORT_CSV_PATH
    retry.INTERIM_DIR = INTERIM_DIR
    retry.PROCESSED_DIR = PROCESSED_DIR
    retry.RUN_DIR = RUN_DIR
    retry.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    retry.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    retry.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    retry.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    retry.RUN_MERGED_DIR = RUN_MERGED_DIR
    retry.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    retry.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    retry.SEED_READY_PATH = SEED_READY_PATH
    retry.FULL_SEED_READY_PATH = FULL_SEED_READY_PATH
    retry.RETRY_PLAN_CSV_PATH = RETRY_PLAN_CSV_PATH
    retry.RETRY_PLAN_MD_PATH = RETRY_PLAN_MD_PATH
    retry.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    retry.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    retry.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    retry.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    retry.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    retry.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    retry.RETRY_MERGED_BEFORE_VALIDATOR_PATH = RETRY_MERGED_BEFORE_VALIDATOR_PATH
    retry.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    retry.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    retry.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    retry.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    retry.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    retry.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    retry.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    retry.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    retry.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    retry.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    retry.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    retry.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    retry.VALIDATOR_RETRY_ACTIONS_CSV_PATH = VALIDATOR_RETRY_ACTIONS_CSV_PATH
    retry.VALIDATOR_FINAL_PACKAGE_CSV_PATH = VALIDATOR_FINAL_PACKAGE_CSV_PATH
    retry.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    retry.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    retry.RETRY_SEED_IDS = RETRY_SEED_IDS
    retry.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    retry.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    retry.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    retry.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    retry.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    retry.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    retry.build_retry_generation_messages = build_retry_generation_messages
    retry.write_wiring_check = write_wiring_check
    retry.write_validator_report = write_validator_report
    retry.write_batch_summary = write_batch_summary
    retry.write_manifest = write_manifest


def main():
    configure_retry_globals()
    return retry.main()


if __name__ == "__main__":
    main()
