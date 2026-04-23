import csv

from common import write_csv_atomic, write_text_atomic
from settings import MERGED_SCORES_PATH, PROBLEM_EXAMPLES_CSV_PATH, PROBLEM_EXAMPLES_MD_PATH, RUN_NAME


def main():
    with open(MERGED_SCORES_PATH, encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    selected_rows = [
        row
        for row in rows
        if row["selected_for_seed"] == "예" and row["final_status"] == "pass"
    ]
    selected_rows.sort(key=lambda row: (-float(row["weighted_score"]), row["seed_sample_id"]))
    example_rows = selected_rows[:3]

    markdown_blocks = [f"# problem_generation examples `{RUN_NAME}`", ""]
    for index, row in enumerate(example_rows, start=1):
        markdown_blocks.extend(
            [
                f"## example {index}",
                f"- seed_sample_id: `{row['seed_sample_id']}`",
                f"- doc_type_name: `{row['doc_type_name']}`",
                f"- source_subset: `{row['source_subset']}`",
                f"- weighted_score: `{row['weighted_score']}`",
                f"- generated_stem: {row['generated_stem']}",
                f"- A. {row['choice_a']}",
                f"- B. {row['choice_b']}",
                f"- C. {row['choice_c']}",
                f"- D. {row['choice_d']}",
                f"- correct_choice: `{row['correct_choice']}`",
                f"- gold_short_answer: {row['gold_short_answer']}",
                "",
            ]
        )

    write_csv_atomic(PROBLEM_EXAMPLES_CSV_PATH, example_rows, list(example_rows[0].keys()) if example_rows else ["seed_sample_id"])
    write_text_atomic(PROBLEM_EXAMPLES_MD_PATH, "\n".join(markdown_blocks) + "\n")
    return example_rows


if __name__ == "__main__":
    main()
