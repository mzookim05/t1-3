import json
import sys
from pathlib import Path

from build_sample_registry import main as build_sample_registry
from common import ensure_run_dirs, write_json_atomic
from export_meeting_examples import main as export_meeting_examples
from extract_evidence_cards import main as extract_evidence_cards
from generate_explanations import main as generate_explanations
from merge_judge_scores import main as merge_judge_scores
from run_judges import main as run_judges
from settings import RUN_MANIFEST_PATH
from transform_problems import main as transform_problems


DATASET_BUILD_DIR = Path(__file__).resolve().parents[1] / "dataset_build"
if str(DATASET_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_BUILD_DIR))

from split_by_family import main as split_by_family


def main():
    ensure_run_dirs()
    registry_rows = build_sample_registry()
    evidence_rows = extract_evidence_cards()
    transformed_rows = transform_problems()
    generate_explanations(mode="main")
    generation_rows = generate_explanations(mode="strict_finalize")
    run_judges(mode="main")
    judge_logs = run_judges(mode="strict_finalize")
    merged_rows = merge_judge_scores()
    split_manifest_rows = split_by_family(merged_rows)
    example_rows = export_meeting_examples(merged_rows)

    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "sample_registry_count": len(registry_rows),
            "evidence_card_count": len(evidence_rows),
            "transformed_count": len(transformed_rows),
            "generation_count": len(generation_rows),
            "judge_log_count": sum(len(rows) for rows in judge_logs.values()),
            "merged_count": len(merged_rows),
            "generator_models_used": sorted({row["generator_model"] for row in generation_rows}),
            "generation_modes_used": sorted({row["generation_mode"] for row in generation_rows}),
            "judge_models_used": sorted(
                {
                    row["judge_model"]
                    for rows in judge_logs.values()
                    for row in rows
                }
            ),
            "judge_modes_used": sorted(
                {
                    row["judge_mode"]
                    for rows in judge_logs.values()
                    for row in rows
                }
            ),
            "selected_pass_count": sum(
                1
                for row in merged_rows
                if row["selected_for_sample"] == "예" and row["final_status"] == "pass"
            ),
            "selected_train_eligible_count": sum(
                1
                for row in merged_rows
                if row["selected_for_sample"] == "예" and row["train_eligible"] == "예"
            ),
            "selected_audit_required_count": sum(
                1
                for row in merged_rows
                if row["selected_for_sample"] == "예" and row["audit_required"] == "예"
            ),
            "selected_hard_fail_count": sum(
                1
                for row in merged_rows
                if row["selected_for_sample"] == "예" and row["final_status"] == "hard_fail"
            ),
            "selected_doc_type_status": {
                row["doc_type_name"]: row["final_status"]
                for row in merged_rows
                if row["selected_for_sample"] == "예"
            },
            "meeting_example_count": len(example_rows),
            "dataset_manifest_count": len(split_manifest_rows),
        }
    )
    write_json_atomic(RUN_MANIFEST_PATH, run_manifest)
    return run_manifest


if __name__ == "__main__":
    main()
