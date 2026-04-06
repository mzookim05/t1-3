import csv

from common import write_csv_atomic, write_text_atomic
from settings import MEETING_EXAMPLE_PRIORITY, MEETING_EXAMPLES_CSV_PATH, MEETING_EXAMPLES_MD_PATH, MEETING_EXAMPLES_TITLE, MERGED_SCORES_PATH


def choose_examples(rows):
    passed_rows = [row for row in rows if row["selected_for_sample"] == "예" and row["final_status"] == "pass"]
    by_doc_type = {}
    for row in passed_rows:
        by_doc_type.setdefault(row["doc_type_name"], []).append(row)

    chosen = []
    for doc_type_name in MEETING_EXAMPLE_PRIORITY:
        candidates = sorted(by_doc_type.get(doc_type_name, []), key=lambda row: -float(row["weighted_score"]))
        if candidates:
            chosen.append(candidates[0])

    if len(chosen) < 3:
        existing_ids = {row["candidate_id"] for row in chosen}
        extras = [row for row in sorted(passed_rows, key=lambda row: -float(row["weighted_score"])) if row["candidate_id"] not in existing_ids]
        chosen.extend(extras[: 3 - len(chosen)])
    return chosen[:3]


def build_markdown(rows):
    lines = [f"# `{MEETING_EXAMPLES_TITLE}`", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## 예시 {index}",
                f"- `DocuType`: `{row['doc_type_name']}`",
                f"- `source_subset`: `{row['source_subset']}`",
                f"- `family_id`: `{row['family_id']}`",
                f"- `weighted_score`: `{row['weighted_score']}`",
                f"- 문제: {row['transformed_problem']}",
                f"- 정답: {row['short_answer']}",
                f"- 해설: {row['generated_explanation']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main(rows):
    chosen = choose_examples(rows)
    if chosen:
        write_csv_atomic(MEETING_EXAMPLES_CSV_PATH, chosen, list(chosen[0].keys()))
        write_text_atomic(MEETING_EXAMPLES_MD_PATH, build_markdown(chosen))
    else:
        write_text_atomic(MEETING_EXAMPLES_MD_PATH, f"# `{MEETING_EXAMPLES_TITLE}`\n\n- pass 예시가 아직 없습니다.\n")
    return chosen


if __name__ == "__main__":
    with open(MERGED_SCORES_PATH, encoding="utf-8-sig", newline="") as input_file:
        main(list(csv.DictReader(input_file)))
