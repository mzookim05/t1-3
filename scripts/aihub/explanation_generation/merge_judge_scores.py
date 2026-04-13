from collections import defaultdict

from common import load_jsonl, write_csv_atomic
from settings import (
    ABLATION_SUMMARY_PATH,
    ACTIVE_GENERATION_VARIANT,
    ANSWER_LOG_PATH,
    GROUNDING_LOG_PATH,
    HARD_FAIL_TAGS,
    MERGED_SCORES_PATH,
    PEDAGOGY_LOG_PATH,
    SCORE_WEIGHTS,
    GENERATIONS_PATH,
)


def normalize_tags(value):
    if isinstance(value, list):
        return value
    return []


def score_entry(row):
    grounding = int(row["Grounding_score"])
    answer = int(row["Answer_score"])
    pedagogy = int(row["Pedagogy_score"])
    return round(
        grounding * SCORE_WEIGHTS["Grounding"]
        + answer * SCORE_WEIGHTS["Answer"]
        + pedagogy * SCORE_WEIGHTS["Pedagogy"],
        2,
    )


def build_merged_rows():
    generations = {row["candidate_id"]: row for row in load_jsonl(GENERATIONS_PATH)}
    role_logs = defaultdict(dict)

    for path in (GROUNDING_LOG_PATH, ANSWER_LOG_PATH, PEDAGOGY_LOG_PATH):
        for row in load_jsonl(path):
            role_logs[row["candidate_id"]][row["role_name"]] = row

    merged = []
    for candidate_id, generation in generations.items():
        grounding = role_logs[candidate_id]["Grounding"]
        answer = role_logs[candidate_id]["Answer"]
        pedagogy = role_logs[candidate_id]["Pedagogy"]
        all_tags = normalize_tags(grounding["error_tags"]) + normalize_tags(answer["error_tags"]) + normalize_tags(pedagogy["error_tags"])
        unique_tags = sorted(set(all_tags))

        row = {
            "sample_id": generation["sample_id"],
            "candidate_id": candidate_id,
            "style_name": generation["style_name"],
            "ablation_variant": generation.get("ablation_variant", ""),
            "ablation_label": generation.get("ablation_label", ""),
            "doc_type_name": generation["doc_type_name"],
            "family_id": generation["family_id"],
            "source_subset": generation["source_subset"],
            "transformed_problem": generation["transformed_problem"],
            "short_answer": generation["short_answer"],
            "generated_explanation": generation["generated_explanation"],
            "Grounding_score": grounding["score"],
            "Answer_score": answer["score"],
            "Pedagogy_score": pedagogy["score"],
            "Grounding_reason": grounding["one_sentence_reason"],
            "Answer_reason": answer["one_sentence_reason"],
            "Pedagogy_reason": pedagogy["one_sentence_reason"],
            "error_tags": "|".join(unique_tags),
        }
        row["weighted_score"] = score_entry(row)
        hard_fail = (
            int(row["Grounding_score"]) < 4
            or int(row["Answer_score"]) < 4
            or any(tag in HARD_FAIL_TAGS for tag in unique_tags)
        )
        soft_pass = (
            int(row["Grounding_score"]) >= 4
            and int(row["Answer_score"]) >= 4
            and int(row["Pedagogy_score"]) >= 3
            and row["weighted_score"] >= 3.8
        )
        row["hard_fail"] = "예" if hard_fail else "아니오"
        row["soft_pass"] = "예" if soft_pass else "아니오"
        if hard_fail:
            row["final_status"] = "hard_fail"
        elif soft_pass:
            row["final_status"] = "pass"
        else:
            row["final_status"] = "soft_fail"
        row["word_count"] = len(row["generated_explanation"].split())
        merged.append(row)

    selected = []
    by_sample = defaultdict(list)
    for row in merged:
        by_sample[row["sample_id"]].append(row)

    for sample_rows in by_sample.values():
        active_rows = [row for row in sample_rows if row["ablation_variant"] == ACTIVE_GENERATION_VARIANT]
        selection_pool = active_rows or sample_rows
        passed_rows = [row for row in selection_pool if row["final_status"] == "pass"]
        if passed_rows:
            passed_rows.sort(key=lambda row: (-row["weighted_score"], row["word_count"]))
            best_row = passed_rows[0]
            if len(passed_rows) > 1 and abs(passed_rows[0]["weighted_score"] - passed_rows[1]["weighted_score"]) <= 0.2:
                best_row = min(passed_rows[:2], key=lambda row: row["word_count"])
        else:
            selection_pool.sort(key=lambda row: (-row["weighted_score"], row["word_count"]))
            best_row = selection_pool[0]

        for row in sample_rows:
            row["selected_for_sample"] = "예" if row["candidate_id"] == best_row["candidate_id"] else "아니오"
            selected.append(row)

    selected.sort(key=lambda row: row["sample_id"])
    return selected


def build_ablation_summary(rows):
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["ablation_variant"]].append(row)

    summary_rows = []
    for variant_name, variant_rows in sorted(by_variant.items()):
        by_sample = defaultdict(list)
        for row in variant_rows:
            by_sample[row["sample_id"]].append(row)

        selected_rows = []
        for sample_rows in by_sample.values():
            passed_rows = [row for row in sample_rows if row["final_status"] == "pass"]
            if passed_rows:
                passed_rows.sort(key=lambda row: (-row["weighted_score"], row["word_count"]))
                best_row = passed_rows[0]
            else:
                sample_rows.sort(key=lambda row: (-row["weighted_score"], row["word_count"]))
                best_row = sample_rows[0]
            selected_rows.append(best_row)

        average_weighted_score = round(
            sum(float(row["weighted_score"]) for row in selected_rows) / len(selected_rows),
            2,
        ) if selected_rows else 0.0
        summary_rows.append(
            {
                "ablation_variant": variant_name,
                "sample_count": len(selected_rows),
                "pass_count": sum(1 for row in selected_rows if row["final_status"] == "pass"),
                "hard_fail_count": sum(1 for row in selected_rows if row["final_status"] == "hard_fail"),
                "average_weighted_score": average_weighted_score,
                "selected_for_release": "예" if variant_name == ACTIVE_GENERATION_VARIANT else "아니오",
            }
        )

    return summary_rows


def main():
    rows = build_merged_rows()
    fieldnames = list(rows[0].keys())
    write_csv_atomic(MERGED_SCORES_PATH, rows, fieldnames)
    ablation_summary = build_ablation_summary(rows)
    write_csv_atomic(ABLATION_SUMMARY_PATH, ablation_summary, list(ablation_summary[0].keys()))
    return rows


if __name__ == "__main__":
    main()
