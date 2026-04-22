from common_v3 import load_jsonl, write_csv_atomic
from settings_v3 import (
    ANSWERABILITY_LOG_PATH,
    GENERATED_PROBLEMS_PATH,
    GROUNDING_LOG_PATH,
    HARD_FAIL_TAGS,
    MERGED_SCORES_PATH,
    RUN_NAME,
    SCORE_WEIGHTS,
    TASKFIT_LOG_PATH,
    VERSION_TAG,
)


def index_rows(rows):
    return {row["candidate_id"]: row for row in rows}


def merge_tags(*tag_lists):
    merged = []
    for tag_list in tag_lists:
        for tag in tag_list:
            if tag not in merged:
                merged.append(tag)
    return merged


def finalize_status(grounding_score, answerability_score, taskfit_score, error_tags, weighted_score):
    if grounding_score < 4 or answerability_score < 4:
        return "hard_fail"
    if any(tag in HARD_FAIL_TAGS for tag in error_tags):
        return "hard_fail"
    if taskfit_score < 3 or weighted_score < 3.8:
        return "soft_fail"
    return "pass"


def build_selected_flags(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed_sample_id"], []).append(row)
    selected_candidate_ids = set()
    for seed_sample_id, seed_rows in grouped.items():
        best_row = sorted(
            seed_rows,
            key=lambda row: (row["final_status"] != "pass", -row["weighted_score"], row["candidate_id"]),
        )[0]
        selected_candidate_ids.add(best_row["candidate_id"])
    return selected_candidate_ids


def main():
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    grounding_map = index_rows(load_jsonl(GROUNDING_LOG_PATH))
    answerability_map = index_rows(load_jsonl(ANSWERABILITY_LOG_PATH))
    taskfit_map = index_rows(load_jsonl(TASKFIT_LOG_PATH))

    rows = []
    for generation in generations:
        grounding = grounding_map[generation["candidate_id"]]
        answerability = answerability_map[generation["candidate_id"]]
        taskfit = taskfit_map[generation["candidate_id"]]
        error_tags = merge_tags(
            grounding.get("error_tags", []),
            answerability.get("error_tags", []),
            taskfit.get("error_tags", []),
        )
        weighted_score = round(
            grounding["score"] * SCORE_WEIGHTS["Grounding"]
            + answerability["score"] * SCORE_WEIGHTS["Answerability"]
            + taskfit["score"] * SCORE_WEIGHTS["TaskFit"],
            4,
        )
        final_status = finalize_status(
            grounding_score=grounding["score"],
            answerability_score=answerability["score"],
            taskfit_score=taskfit["score"],
            error_tags=error_tags,
            weighted_score=weighted_score,
        )
        audit_required = "예" if final_status == "pass" and error_tags else "아니오"
        train_eligible = "예" if final_status == "pass" and audit_required == "아니오" else "아니오"
        rows.append(
            {
                "seed_sample_id": generation["seed_sample_id"],
                "candidate_id": generation["candidate_id"],
                "problem_task_type": generation["problem_task_type"],
                "problem_generation_mode": generation["problem_generation_mode"],
                "doc_type_name": generation["doc_type_name"],
                "source_subset": generation["source_subset"],
                "sampling_lane": generation["sampling_lane"],
                "family_id": generation["family_id"],
                "generated_problem": generation["generated_problem"],
                "split_strategy": generation.get("split_strategy", ""),
                "focus_issue": generation.get("focus_issue", ""),
                "multi_query_signal": generation.get("multi_query_signal", ""),
                "split_focus_hint": generation.get("split_focus_hint", ""),
                "gold_short_answer": generation["gold_short_answer"],
                "gold_reference_explanation": generation["gold_reference_explanation"],
                "answer_mode": generation.get("answer_mode", ""),
                "explanation_target": generation.get("explanation_target", ""),
                "rule_basis": generation.get("rule_basis", ""),
                "fact_basis": generation.get("fact_basis", ""),
                "grounding_score": grounding["score"],
                "answerability_score": answerability["score"],
                "taskfit_score": taskfit["score"],
                "weighted_score": weighted_score,
                "error_tags": "|".join(error_tags),
                "final_status": final_status,
                "audit_required": audit_required,
                "audit_reason": "|".join(error_tags) if audit_required == "예" else "",
                "train_eligible": train_eligible,
                "generator_model": generation["generation_model"],
                "generation_mode": generation["generation_mode"],
                "grounding_judge_model": grounding["judge_model"],
                "answerability_judge_model": answerability["judge_model"],
                "taskfit_judge_model": taskfit["judge_model"],
                "version_tag": VERSION_TAG,
                "run_name": RUN_NAME,
                "label_path": generation.get("label_path", ""),
                "raw_path": generation.get("raw_path", ""),
            }
        )

    selected_candidate_ids = build_selected_flags(rows)
    for row in rows:
        row["selected_for_seed"] = "예" if row["candidate_id"] in selected_candidate_ids else "아니오"

    rows.sort(key=lambda row: (row["seed_sample_id"], row["candidate_id"]))
    write_csv_atomic(MERGED_SCORES_PATH, rows, list(rows[0].keys()))
    return rows


if __name__ == "__main__":
    main()
