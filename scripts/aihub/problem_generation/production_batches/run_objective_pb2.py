import csv
import sys
from collections import Counter
from pathlib import Path

# production batch runner 경로는 `.../scripts/aihub/problem_generation/production_batches`이므로,
# repo root는 `parents[3]`에서 끊어야 `scripts` 패키지를 정확히 찾는다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.v2_objective_difficulty_patch_r2 import run_difficulty_patch as r2


base = r2.base

# `pb2`는 current default를 바로 교체하는 batch가 아니라,
# `pb1` residual seed를 `r2` recipe로 다시 태워 production robustness를 보는 shadow candidate batch다.
VERSION_TAG = "pb2_objective_candidate"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_shadow_batch"
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

REFERENCE_PB1_SEED_READY_PATH = (
    base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb1_objective"
    / "seed_ready.jsonl"
)
REFERENCE_PB1_MERGED_PATH = (
    base.PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-22_164512_pb1_objective_objective_v2_default_production_batch"
    / "merged"
    / "merged_problem_scores_pb1_objective.csv"
)

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
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

BATCH_COMPARE_MD_PATH = RUN_EXPORTS_DIR / f"batch_compare_pb1_vs_{VERSION_TAG}.md"
BATCH_COMPARE_CSV_PATH = RUN_EXPORTS_DIR / f"batch_compare_pb1_vs_{VERSION_TAG}.csv"


def configure_pb2():
    # `r2` runner를 그대로 재사용하되 산출물 경로와 reference를 `pb2` shadow batch 기준으로 재배선한다.
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
    r2.PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
    r2.PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
    r2.PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
    r2.PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
    r2.PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"
    r2.SIDE_BY_SIDE_MD_PATH = BATCH_COMPARE_MD_PATH
    r2.SIDE_BY_SIDE_CSV_PATH = BATCH_COMPARE_CSV_PATH
    r2.REFERENCE_PATCH_MERGED_PATH = REFERENCE_PB1_MERGED_PATH
    r2.REFERENCE_PATCH_ROWS_CACHE = None
    r2.configure_base()

    # generation prompt의 reference row도 comparator `v2`가 아니라 같은 residual seed의 current default `pb1`을 보게 한다.
    base.REFERENCE_V2_MERGED_PATH = REFERENCE_PB1_MERGED_PATH
    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Keyedness": KEYEDNESS_LOG_PATH,
        "DistractorFit": DISTRACTORFIT_LOG_PATH,
        "NearMiss": NEARMISS_LOG_PATH,
    }


def load_reference_pb1_rows():
    rows = base.load_csv_rows(REFERENCE_PB1_MERGED_PATH)
    return {row["seed_sample_id"]: row for row in rows if row.get("selected_for_seed") == "예"}


def build_seed_registry():
    # `pb2`는 새로운 seed hunting이 아니라 이미 `pb1`에서 쓴 non-comparator residual seed를 그대로 재사용한다.
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
    seed_rows = base.load_jsonl(REFERENCE_PB1_SEED_READY_PATH)
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["sampling_lane"], row["seed_sample_id"]))
    for row in seed_rows:
        row["selection_role"] = "objective_pb2_candidate_seed"
        row["selection_note"] = (
            "pb1 residual objective seed를 그대로 재사용해, current default(v2 recipe)와 candidate default(r2 recipe)를 "
            "같은 non-comparator 배치에서 비교하기 위한 shadow production batch"
        )
        row["selected_at_utc"] = base.utc_now_iso()

    base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    base.copy_file_to_run_inputs(REFERENCE_PB1_MERGED_PATH, RUN_INPUTS_DIR, "reference_merged_problem_scores_pb1_objective.csv")
    return seed_rows


def summarize_rows(rows):
    selected_rows = [row for row in rows if row["selected_for_seed"] == "예"]
    return {
        "selected_pass_count": sum(1 for row in selected_rows if row["final_status"] == "pass"),
        "selected_hard_fail_count": sum(1 for row in selected_rows if row["final_status"] == "hard_fail"),
        "selected_soft_fail_count": sum(1 for row in selected_rows if row["final_status"] == "soft_fail"),
        "selected_train_eligible_count": sum(1 for row in selected_rows if row.get("train_eligible") == "예"),
        "selected_audit_required_count": sum(1 for row in selected_rows if row.get("audit_required") == "예"),
        "doc_type_pass_counter": Counter(
            row["doc_type_name"]
            for row in selected_rows
            if row["final_status"] == "pass" and row.get("train_eligible") == "예"
        ),
    }


def build_batch_compare(pb2_rows):
    # reviewer가 다음 stop line에서 side-by-side보다 batch-level yield를 더 보라고 했으므로,
    # 동일 seed 기준 `pb1` 대 `pb2` 비교표를 export로 남긴다.
    reference_pb1_map = load_reference_pb1_rows()
    selected_pb2_rows = [row for row in pb2_rows if row["selected_for_seed"] == "예"]
    selected_pb2_rows.sort(key=lambda row: (row["doc_type_name"], row["seed_sample_id"]))

    compare_rows = []
    for row in selected_pb2_rows:
        reference = reference_pb1_map.get(row["seed_sample_id"], {})
        compare_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "pb1_final_status": reference.get("final_status", ""),
                "pb1_train_eligible": reference.get("train_eligible", ""),
                "pb1_audit_required": reference.get("audit_required", ""),
                "pb1_weighted_score": reference.get("weighted_score", ""),
                "pb1_error_tags": reference.get("error_tags", ""),
                "pb2_final_status": row["final_status"],
                "pb2_train_eligible": row.get("train_eligible", ""),
                "pb2_audit_required": row.get("audit_required", ""),
                "pb2_weighted_score": row["weighted_score"],
                "pb2_nearmiss_score": row.get("nearmiss_score", ""),
                "pb2_error_tags": row.get("error_tags", ""),
            }
        )

    pb1_summary = summarize_rows(list(reference_pb1_map.values()))
    pb2_summary = summarize_rows(pb2_rows)
    doc_types = sorted({row["doc_type_name"] for row in selected_pb2_rows})

    markdown_lines = [
        f"# batch compare `pb1_objective` vs `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(compare_rows)}`",
        f"- pb1 selected: `{pb1_summary['selected_pass_count']} pass / {pb1_summary['selected_hard_fail_count']} hard_fail / {pb1_summary['selected_soft_fail_count']} soft_fail`",
        f"- pb1 train/audit: `train_eligible {pb1_summary['selected_train_eligible_count']} / audit_required {pb1_summary['selected_audit_required_count']}`",
        f"- pb2 selected: `{pb2_summary['selected_pass_count']} pass / {pb2_summary['selected_hard_fail_count']} hard_fail / {pb2_summary['selected_soft_fail_count']} soft_fail`",
        f"- pb2 train/audit: `train_eligible {pb2_summary['selected_train_eligible_count']} / audit_required {pb2_summary['selected_audit_required_count']}`",
        "",
        "## doc type usable yield",
        "| doc_type | pb1_train_eligible | pb2_train_eligible |",
        "| --- | ---: | ---: |",
    ]
    for doc_type_name in doc_types:
        markdown_lines.append(
            f"| `{doc_type_name}` | `{pb1_summary['doc_type_pass_counter'].get(doc_type_name, 0)}` | `{pb2_summary['doc_type_pass_counter'].get(doc_type_name, 0)}` |"
        )

    markdown_lines.extend(["", "## row-level compare", "| seed_sample_id | doc_type | pb1 | pb2 |", "| --- | --- | --- | --- |"])
    for row in compare_rows:
        pb1_text = (
            f"{row['pb1_final_status']} / train={row['pb1_train_eligible']} / audit={row['pb1_audit_required']} / "
            f"score={row['pb1_weighted_score']}"
        )
        pb2_text = (
            f"{row['pb2_final_status']} / train={row['pb2_train_eligible']} / audit={row['pb2_audit_required']} / "
            f"score={row['pb2_weighted_score']} / nearmiss={row['pb2_nearmiss_score']}"
        )
        markdown_lines.append(f"| `{row['seed_sample_id']}` | `{row['doc_type_name']}` | `{pb1_text}` | `{pb2_text}` |")

    base.write_csv_atomic(BATCH_COMPARE_CSV_PATH, compare_rows, list(compare_rows[0].keys()) if compare_rows else ["seed_sample_id"])
    base.write_text_atomic(BATCH_COMPARE_MD_PATH, "\n".join(markdown_lines) + "\n")
    return compare_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, compare_rows):
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "reference_pb1_seed_ready_path": str(REFERENCE_PB1_SEED_READY_PATH),
        "reference_pb1_merged_path": str(REFERENCE_PB1_MERGED_PATH),
        "candidate_recipe_source": "v2_difficulty_patch_r2",
        "seed_registry_strategy": "reuse_pb1_non_comparator_residual_seed_set_as_pb2_shadow_batch",
        "seed_registry_count": len(seed_rows),
        "generation_count": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": base.load_jsonl_count(NEARMISS_LOG_PATH),
        "merged_count": base.load_csv_count(MERGED_SCORES_PATH),
        "selected_pass_count": sum(
            1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "pass"
        ),
        "selected_hard_fail_count": sum(
            1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "hard_fail"
        ),
        "selected_soft_fail_count": sum(
            1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "soft_fail"
        ),
        "selected_train_eligible_count": sum(
            1 for row in merged_rows if row["selected_for_seed"] == "예" and row.get("train_eligible") == "예"
        ),
        "selected_audit_required_count": sum(
            1 for row in merged_rows if row["selected_for_seed"] == "예" and row.get("audit_required") == "예"
        ),
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "batch_compare_rows_count": len(compare_rows),
        "r2_shadow_batch_focus": [
            "pb1 residual non-comparator objective seed 재사용",
            "r2 recipe production robustness 확인",
            "current usable count 미합산 candidate batch",
        ],
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "batch_compare_md": str(BATCH_COMPARE_MD_PATH),
            "batch_compare_csv": str(BATCH_COMPARE_CSV_PATH),
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
    configure_pb2()
    base.build_seed_registry = build_seed_registry
    base.load_reference_v2_rows = load_reference_pb1_rows
    base.build_local_fallback_problem = r2.build_local_fallback_problem
    base.postprocess_problem = r2.postprocess_problem
    base.build_generation_messages = r2.build_generation_messages
    base.build_side_by_side_examples = build_batch_compare
    base.build_run_manifest = build_run_manifest
    return base.main()


if __name__ == "__main__":
    main()
