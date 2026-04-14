import json

from build_problem_seed_registry_v2 import main as build_problem_seed_registry_v2
from common_v2 import build_run_manifest, write_json_atomic
from export_problem_examples_v2 import main as export_problem_examples_v2
from generate_mcq_problems import main as generate_mcq_problems
from merge_problem_judge_scores_v2 import main as merge_problem_judge_scores_v2
from run_problem_judges_v2 import main as run_problem_judges_v2
from settings_v2 import (
    DISTRACTORFIT_LOG_PATH,
    GENERATED_PROBLEMS_PATH,
    GROUNDING_LOG_PATH,
    KEYEDNESS_LOG_PATH,
    MERGED_SCORES_PATH,
    PROBLEM_AUDIT_QUEUE_PATH,
    PROBLEM_DATASET_MANIFEST_PATH,
    PROBLEM_DEV_PATH,
    PROBLEM_EXAMPLES_CSV_PATH,
    PROBLEM_EXAMPLES_MD_PATH,
    PROBLEM_TEST_PATH,
    PROBLEM_TRAIN_PATH,
    RUN_MANIFEST_PATH,
    SEED_READY_PATH,
    SEED_REGISTRY_PATH,
)


def load_jsonl_count(path):
    with open(path, encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def load_csv_count(path):
    with open(path, encoding="utf-8-sig") as input_file:
        return max(0, sum(1 for _ in input_file) - 1)


def main():
    build_problem_seed_registry_v2()
    generate_mcq_problems(mode="main")
    generate_mcq_problems(mode="strict_finalize")
    run_problem_judges_v2(mode="main")
    run_problem_judges_v2(mode="strict_finalize")
    merged_rows = merge_problem_judge_scores_v2()

    import sys
    from pathlib import Path

    dataset_build_dir = Path(__file__).resolve().parents[1] / "dataset_build"
    if str(dataset_build_dir) not in sys.path:
        sys.path.insert(0, str(dataset_build_dir))
    from split_problem_generation_v2 import main as split_problem_generation_v2_main

    manifest_rows = split_problem_generation_v2_main(merged_rows)
    example_rows = export_problem_examples_v2()

    manifest = build_run_manifest()
    manifest.update(
        {
            "seed_registry_count": load_csv_count(SEED_REGISTRY_PATH),
            "seed_ready_count": load_jsonl_count(SEED_READY_PATH),
            "generation_count": load_jsonl_count(GENERATED_PROBLEMS_PATH),
            "judge_grounding_count": load_jsonl_count(GROUNDING_LOG_PATH),
            "judge_keyedness_count": load_jsonl_count(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_count": load_jsonl_count(DISTRACTORFIT_LOG_PATH),
            "merged_count": load_csv_count(MERGED_SCORES_PATH),
            "selected_pass_count": sum(
                1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "pass"
            ),
            "selected_hard_fail_count": sum(
                1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "hard_fail"
            ),
            "selected_soft_fail_count": sum(
                1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "soft_fail"
            ),
            "dataset_manifest_count": len(manifest_rows),
            "problem_train_count": load_jsonl_count(PROBLEM_TRAIN_PATH),
            "problem_dev_count": load_jsonl_count(PROBLEM_DEV_PATH),
            "problem_test_count": load_jsonl_count(PROBLEM_TEST_PATH),
            "problem_audit_count": load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
            "problem_examples_count": len(example_rows),
            "artifact_paths": {
                "seed_registry": str(SEED_REGISTRY_PATH),
                "seed_ready": str(SEED_READY_PATH),
                "generated_problems": str(GENERATED_PROBLEMS_PATH),
                "judge_grounding_log": str(GROUNDING_LOG_PATH),
                "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
                "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
                "merged_scores": str(MERGED_SCORES_PATH),
                "problem_examples_md": str(PROBLEM_EXAMPLES_MD_PATH),
                "problem_examples_csv": str(PROBLEM_EXAMPLES_CSV_PATH),
                "problem_train": str(PROBLEM_TRAIN_PATH),
                "problem_dev": str(PROBLEM_DEV_PATH),
                "problem_test": str(PROBLEM_TEST_PATH),
                "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
                "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
            },
        }
    )
    write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    main()
