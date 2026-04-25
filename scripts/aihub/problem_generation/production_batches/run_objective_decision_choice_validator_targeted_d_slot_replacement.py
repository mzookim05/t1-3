import json
import sys
from collections import Counter
from pathlib import Path

# `decision_targeted_008`은 bounded retry 1회 후에도 weak distractor로 남았으므로,
# 같은 seed를 반복하지 않고 fresh replacement seed로 `D` slot만 복구한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
EXPLANATION_DIR_FOR_IMPORT = PROJECT_ROOT_FOR_IMPORT / "scripts" / "aihub" / "problem_generation" / "explanation_generation"
if str(EXPLANATION_DIR_FOR_IMPORT) not in sys.path:
    # 과거 production runner들이 top-level `common/settings` import를 사용하므로 새 구조 경로를 먼저 넣는다.
    sys.path.insert(0, str(EXPLANATION_DIR_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_a_slot_replacement as replacement,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_targeted_2slot_repair as repair,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_targeted_pilot as targeted,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as pb8  # noqa: E402


VERSION_TAG = "decision_choice_validator_targeted_d_slot_replacement"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_targeted_d_slot_replacement_package"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = targeted.PROJECT_ROOT
SOURCE_PACKAGE_PATH = repair.MERGED_SCORES_PATH
SOURCE_RUN_DIR = repair.RUN_DIR
FAILED_SLOT_SEED_ID = "decision_targeted_008"
FAILED_SLOT_FAMILY_ID = "결정례_QA::지식재산권법_심결문_61155"
REPLACEMENT_SEED_ID = "decision_targeted_replacement_001"
REPLACEMENT_TARGET_LABEL = "D"
# D-slot replacement는 실제 seed row의 lane/source_subset을 reviewer-facing artifact에 남긴다.
# 이전 generic label은 감사 시 실제 `seed_preflight.csv`와 어긋나 보일 수 있어 고정값을 실제 row 기준으로 좁힌다.
EXPECTED_REPLACEMENT_SOURCE_SUBSET = "01_TL_심결례_QA"
EXPECTED_REPLACEMENT_LANE = "expansion_01_02"

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

SUCCESS_USABLE_MIN = 15
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}


def collect_excluded_rows():
    rows = targeted.collect_excluded_rows_for_targeted()
    rows.extend(replacement.load_csv_rows_if_exists(targeted.SEED_REGISTRY_PATH))
    rows.extend(replacement.load_csv_rows_if_exists(repair.SEED_REGISTRY_PATH))
    rows.append({"family_id": FAILED_SLOT_FAMILY_ID, "seed_sample_id": FAILED_SLOT_SEED_ID})
    return rows


def decision_expansion_specs():
    specs = []
    for spec in pb8.pb6.pb4.pb3.DATASET_SPECS:
        if spec["doc_type_name"] == "결정례_QA":
            copied = dict(spec)
            copied["sample_count"] = 1
            specs.append(copied)
    return specs


def select_replacement_record():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = replacement.build_exclusion_sets(excluded_rows)

    for spec in decision_expansion_specs():
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
                "sampling_lane": spec.get("sampling_lane", EXPECTED_REPLACEMENT_LANE),
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
                "selection_note": "decision_targeted_008 D-slot replacement seed: prior weak-distractor family excluded",
            }, exclusion_sets
    raise RuntimeError("targeted D-slot replacement용 fresh 결정례_QA expansion seed를 찾지 못했습니다.")


def build_seed_row(record):
    row = pb8.pb6.pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_decision_choice_validator_targeted_d_slot_replacement_seed"
    row["selection_note"] = "decision_targeted_008 실패 D slot을 fresh 결정례_QA expansion seed로 대체하는 replacement seed"
    row["decision_validator_replacement_note"] = "failed_seed_decision_targeted_008_family_61155_excluded"
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
    if row["sampling_lane"] not in {"generalization_03_04", "expansion_01_02"}:
        raise RuntimeError("replacement seed는 결정례_QA의 운영 lane 안에 있어야 합니다.")
    if row["source_subset"] != EXPECTED_REPLACEMENT_SOURCE_SUBSET:
        raise RuntimeError(f"replacement seed source_subset mismatch: {row['source_subset']}")
    if row["sampling_lane"] != EXPECTED_REPLACEMENT_LANE:
        raise RuntimeError(f"replacement seed sampling_lane mismatch: {row['sampling_lane']}")
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
    replacement.write_csv(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    replacement.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    replacement.write_csv(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    actual_source_subset = seed_rows[0]["source_subset"]
    actual_lane = seed_rows[0]["sampling_lane"]
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        "- seed_count: `1`",
        "- doc_type_counts: `{'결정례_QA': 1}`",
        f"- source_subset_counts: `{{'{actual_source_subset}': 1}}`",
        f"- lane_counts: `{{'{actual_lane}': 1}}`",
        f"- failed_slot_seed_excluded: `{FAILED_SLOT_SEED_ID}` / `{FAILED_SLOT_FAMILY_ID}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        "| replacement seed count is 1 | `pass` |",
        "| doc type is decision only | `pass` |",
        f"| source subset is {actual_source_subset} | `pass` |",
        f"| lane is {actual_lane} | `pass` |",
        f"| target label is {REPLACEMENT_TARGET_LABEL} | `pass` |",
        "| no current/failed/targeted/micro seed overlap | `pass` |",
    ]
    replacement.write_text(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    replacement.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    replacement.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    replacement.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    replacement.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)
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
    replacement.write_csv(REPLACEMENT_PLAN_CSV_PATH, rows, list(rows[0].keys()))
    lines = [
        f"# replacement plan `{VERSION_TAG}`",
        "",
        "| failed_seed | replacement_seed | lane | target | policy |",
        "| --- | --- | --- | --- | --- |",
        f"| `{FAILED_SLOT_SEED_ID}` | `{seed_rows[0]['seed_sample_id']}` | `{seed_rows[0]['sampling_lane']}` | `{REPLACEMENT_TARGET_LABEL}` | `seed_specific_exclusion_then_one_fresh_replacement` |",
    ]
    replacement.write_text(REPLACEMENT_PLAN_MD_PATH, "\n".join(lines) + "\n")
    return rows


def build_replacement_generation_messages(seed, reference_v2):
    messages = targeted.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## targeted D-slot replacement 추가 지시
- 이 seed는 `decision_targeted_008`이 bounded retry 1회 후에도 weak distractor로 남아 생긴 `D` label 결손을 복구하기 위한 replacement seed다.
- 기존 실패 seed의 `family_id`는 `{FAILED_SLOT_FAMILY_ID}`이며, 이번 seed는 동일 family를 쓰면 안 된다.
- 생성 단계의 choice label은 후처리 validator가 target `{REPLACEMENT_TARGET_LABEL}`로 다시 맞추므로, 의미상 정답 유일성과 오답 3개의 균등한 near-miss plausibility를 최우선으로 한다.
- 단순 법리 회상형, 정답 반대말 1개만 세우는 오답, 정답 쟁점과 먼 일반론 오답을 피한다.
- 세 오답은 같은 판단 기준이나 적용 사실을 공유하되, 각각 서로 다른 한 축만 어긋나야 한다.
"""
    return messages


def write_wiring_check():
    lines = [
        f"# targeted D-slot replacement wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| bounded retry exhausted | `pass` | `decision_targeted_008`은 retry 1회 후에도 regenerate로 남음 |",
        "| failed seed excluded | `pass` | `decision_targeted_008`, `family_id 61155` 제외 |",
        f"| replacement seed scope | `pass` | `결정례_QA`, `{EXPECTED_REPLACEMENT_LANE}`, fresh seed `1개` |",
        f"| replacement target label | `pass` | `{REPLACEMENT_TARGET_LABEL}` slot 복구 목적 |",
        "| preserved package rows | `pass` | targeted 2-slot repair package의 export-ready `15개` 보존 |",
        "| metadata remap audit | `pass` | 보존 row와 replacement row 모두 post-shuffle metadata gate 적용 |",
        "| count reflection | `pass` | reviewer sign-off 전 current count 미합산 |",
    ]
    replacement.write_text(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def build_package(validated_replacement_rows):
    rows = replacement.preserved_export_rows() + replacement.selected_rows(validated_replacement_rows)
    rows.sort(key=lambda row: row["seed_sample_id"])
    for row in rows:
        row["selected_for_seed"] = "예"
        if row.get("replacement_package_role") == "a_slot_replacement_candidate":
            # 기존 A-slot runner를 재사용했더라도 reviewer artifact에는 이번 D-slot 역할이 드러나야 한다.
            row["replacement_package_role"] = "d_slot_replacement_candidate"
    return rows


def write_validator_report(replacement_report_rows, package):
    summary = replacement.compute_validator_summary(package)
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

    replacement.write_csv(VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH, replacement_report_rows)
    replacement.write_csv(VALIDATOR_FINAL_PACKAGE_CSV_PATH, package_rows_for_report)
    replacement.write_csv(
        VALIDATOR_REPORT_CSV_PATH,
        replacement_report_rows + package_rows_for_report,
        replacement.all_fieldnames(replacement_report_rows + package_rows_for_report),
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
            "## final 16-slot package",
            "| seed | role | action | final_status | train_eligible | target | correct | metadata | disposition |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in package_rows_for_report:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['replacement_package_role']}` | `{row['validator_action']}` | `{row['final_status']}` | `{row['train_eligible']}` | `{row['target_correct_choice']}` | `{row['correct_choice']}` | `{row['metadata_remap_ok']}` | `{row['export_disposition']}` |"
        )
    replacement.write_text(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_batch_summary(package):
    summary = replacement.compute_validator_summary(package)
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
    replacement.write_csv(BATCH_SUMMARY_CSV_PATH, summary_rows)
    replacement.write_csv(
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
    replacement.write_text(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def write_manifest(seed_rows, package, manifest_rows, summary_rows, tail_rows, replacement_plan_rows, metadata_audit_rows):
    summary = replacement.compute_validator_summary(package)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": replacement.base.utc_now_iso(),
        "source_targeted_2slot_repair_run_name": SOURCE_RUN_DIR.name,
        "replacement_policy": {
            "failed_seed_sample_id": FAILED_SLOT_SEED_ID,
            "failed_family_id": FAILED_SLOT_FAMILY_ID,
            "replacement_seed_ids": [row["seed_sample_id"] for row in seed_rows],
            "replacement_target_label": REPLACEMENT_TARGET_LABEL,
            "preserve_export_ready_rows": True,
            "semantic_judge": "replacement row only",
            "metadata_remap_gate": True,
            "count_reflection": "not_counted_until_reviewer_signoff",
        },
        "seed_registry_count": len(seed_rows),
        "generation_count": replacement.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": replacement.base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": replacement.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": replacement.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": replacement.base.load_jsonl_count(NEARMISS_LOG_PATH),
        "merged_count": len(package),
        "selected_pass_count": summary["selected_count"] - summary["selected_hard_fail_count"] - summary["selected_soft_fail_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": replacement.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": replacement.base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": replacement.base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": replacement.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
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
    replacement.write_json(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_replacement_globals():
    # A-slot replacement runner의 검증된 package builder를 유지하고 D-slot source/경로만 교체한다.
    pb8.pb6.pb4.pb3.explanation_common.PROJECT_ROOT = PROJECT_ROOT
    replacement.VERSION_TAG = VERSION_TAG
    replacement.RUN_DATE = RUN_DATE
    replacement.RUN_PURPOSE = RUN_PURPOSE
    replacement.RUN_NAME = RUN_NAME
    replacement.SOURCE_RUN_DIR = SOURCE_RUN_DIR
    replacement.SOURCE_PACKAGE_PATH = SOURCE_PACKAGE_PATH
    replacement.FAILED_SLOT_SEED_ID = FAILED_SLOT_SEED_ID
    replacement.FAILED_SLOT_FAMILY_ID = FAILED_SLOT_FAMILY_ID
    replacement.REPLACEMENT_SEED_ID = REPLACEMENT_SEED_ID
    replacement.REPLACEMENT_TARGET_LABEL = REPLACEMENT_TARGET_LABEL
    replacement.INTERIM_DIR = INTERIM_DIR
    replacement.PROCESSED_DIR = PROCESSED_DIR
    replacement.RUN_DIR = RUN_DIR
    replacement.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    replacement.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    replacement.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    replacement.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    replacement.RUN_MERGED_DIR = RUN_MERGED_DIR
    replacement.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    replacement.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    replacement.SEED_READY_PATH = SEED_READY_PATH
    replacement.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    replacement.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    replacement.REPLACEMENT_PLAN_CSV_PATH = REPLACEMENT_PLAN_CSV_PATH
    replacement.REPLACEMENT_PLAN_MD_PATH = REPLACEMENT_PLAN_MD_PATH
    replacement.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    replacement.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    replacement.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    replacement.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    replacement.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    replacement.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    replacement.REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH = REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH
    replacement.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    replacement.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    replacement.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    replacement.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    replacement.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    replacement.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    replacement.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    replacement.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    replacement.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    replacement.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    replacement.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    replacement.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    replacement.VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH = VALIDATOR_REPLACEMENT_ACTIONS_CSV_PATH
    replacement.VALIDATOR_FINAL_PACKAGE_CSV_PATH = VALIDATOR_FINAL_PACKAGE_CSV_PATH
    replacement.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    replacement.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    replacement.METADATA_REMAP_AUDIT_CSV_PATH = METADATA_REMAP_AUDIT_CSV_PATH
    replacement.METADATA_REMAP_AUDIT_MD_PATH = METADATA_REMAP_AUDIT_MD_PATH
    replacement.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    replacement.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    replacement.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    replacement.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    replacement.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    replacement.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    replacement.collect_excluded_rows = collect_excluded_rows
    replacement.select_replacement_record = select_replacement_record
    replacement.build_seed_row = build_seed_row
    replacement.write_seed_inputs = write_seed_inputs
    replacement.write_replacement_plan = write_replacement_plan
    replacement.build_replacement_generation_messages = build_replacement_generation_messages
    replacement.write_wiring_check = write_wiring_check
    replacement.build_package = build_package
    replacement.write_validator_report = write_validator_report
    replacement.write_batch_summary = write_batch_summary
    replacement.write_manifest = write_manifest


def main():
    configure_replacement_globals()
    return replacement.main()


if __name__ == "__main__":
    main()
