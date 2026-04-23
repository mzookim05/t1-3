import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "aihub" / "dataset_build" / "realrun"
CHOICE_LETTERS = ["A", "B", "C", "D"]


def resolve_processed_split_paths(version_tag: str) -> tuple[Path, Path, Path]:
    # objective consumer는 line별 processed 폴더를 직접 읽어 새 batch/patch와 기준선을 구분한다.
    folder_by_version = {
        "v2": "v2_objective",
        "v2_difficulty_patch": "v2_objective_difficulty_patch",
        "pb1_objective": "production_batches/pb1_objective",
    }
    folder_name = folder_by_version.get(version_tag) or version_tag
    processed_dir = PROCESSED_ROOT / folder_name
    return processed_dir / "train.jsonl", processed_dir / "dev.jsonl", processed_dir / "test.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def build_problem_query_text(row):
    # 문제 소비 기준선에서는 생성된 stem과 source 메타를 같이 써서
    # train problem retrieval이 지나치게 문면 일치에만 기대지 않게 한다.
    fields = [
        f"문서유형: {row.get('doc_type_name', '')}",
        f"대분류: {row.get('source_subset', '')}",
        f"정답모드: {row.get('answer_mode', '')}",
        f"문제유형: {row.get('problem_task_type', '')}",
        f"문제: {row.get('generated_stem', '')}",
    ]
    return "\n".join(normalize_text(field) for field in fields if normalize_text(field))


def build_answer_reference_text(row):
    # 선택지 정답성은 train sample의 gold answer와 reference explanation을
    # 함께 reference로 써서 너무 짧은 short_answer 편향을 줄인다.
    answer_text = normalize_text(row.get("gold_short_answer", ""))
    explanation_text = normalize_text(row.get("gold_reference_explanation", ""))
    return "\n".join(text for text in [answer_text, explanation_text] if text)


def build_choice_lookup(row):
    return {letter: normalize_text(row[f"choice_{letter.lower()}"]) for letter in CHOICE_LETTERS}


def reciprocal_rank(rank_value):
    return 1.0 / rank_value if rank_value > 0 else 0.0


def summarize_predictions(split_name, rows):
    if not rows:
        return {
            "split": split_name,
            "row_count": 0,
            "choice_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "top2_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "doc_type_breakdown": {},
        }

    summary = {
        "split": split_name,
        "row_count": len(rows),
        "choice_accuracy": round(mean(row["choice_accuracy"] for row in rows), 6),
        "mean_reciprocal_rank": round(mean(row["choice_reciprocal_rank"] for row in rows), 6),
        "top2_hit_rate": round(mean(row["top2_hit"] for row in rows), 6),
        "top3_hit_rate": round(mean(row["top3_hit"] for row in rows), 6),
        "doc_type_breakdown": {},
    }

    rows_by_doc_type = defaultdict(list)
    for row in rows:
        rows_by_doc_type[row["doc_type_name"]].append(row)

    for doc_type_name, doc_rows in rows_by_doc_type.items():
        summary["doc_type_breakdown"][doc_type_name] = {
            "row_count": len(doc_rows),
            "choice_accuracy": round(mean(row["choice_accuracy"] for row in doc_rows), 6),
            "mean_reciprocal_rank": round(mean(row["choice_reciprocal_rank"] for row in doc_rows), 6),
            "top2_hit_rate": round(mean(row["top2_hit"] for row in doc_rows), 6),
            "top3_hit_rate": round(mean(row["top3_hit"] for row in doc_rows), 6),
        }

    return summary


def build_markdown_report(version_tag, run_name, run_dir, config, train_summary, dev_summary, test_summary):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {version_tag} problem downstream consumer baseline 결과",
        "",
        f"- 생성 시각: {generated_at}",
        f"- run 이름: `{run_name}`",
        f"- 출력 경로: `{run_dir}`",
        "- baseline 종류: `tfidf_dual_retrieval_choice_ranker`",
        f"- top-k retrieval: `{config['top_k']}`",
        "- 목적: `problem_generation v2`가 만든 객관식 문제셋을 직접 소비하는 경량 downstream baseline을 하나 잠그고, 입력/출력/평가지표 구조를 end-to-end로 고정하는 것",
        "",
        "## 소비 구조",
        "",
        "- 입력: `problem_train/dev/test_v2.jsonl`의 generated stem, 선택지, 문서유형/대분류 메타",
        "- 학습/참조 자원: train split의 gold short answer + gold reference explanation",
        "- 예측 방식: test problem과 가장 가까운 train problem을 `top-k`로 retrieval한 뒤, retrieved answer reference와 가장 잘 맞는 선택지를 고른다",
        "- 평가 지표: choice accuracy, MRR, top-2 hit, top-3 hit",
        "",
        "## split별 요약",
        "",
        "| split | rows | choice accuracy | MRR | top-2 hit | top-3 hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_summary['row_count']} | {train_summary['choice_accuracy']:.4f} | {train_summary['mean_reciprocal_rank']:.4f} | {train_summary['top2_hit_rate']:.4f} | {train_summary['top3_hit_rate']:.4f} |",
        f"| dev | {dev_summary['row_count']} | {dev_summary['choice_accuracy']:.4f} | {dev_summary['mean_reciprocal_rank']:.4f} | {dev_summary['top2_hit_rate']:.4f} | {dev_summary['top3_hit_rate']:.4f} |",
        f"| test | {test_summary['row_count']} | {test_summary['choice_accuracy']:.4f} | {test_summary['mean_reciprocal_rank']:.4f} | {test_summary['top2_hit_rate']:.4f} | {test_summary['top3_hit_rate']:.4f} |",
        "",
        "## 해석 메모",
        "",
        "- 이 baseline은 LLM fine-tuning 결과가 아니라, 현재 `problem_generation v2` 산출물을 실제로 소비하는 경량 consumer baseline이다.",
        "- train 점수는 self-retrieval이 포함되므로 sanity check 성격으로 읽고, 실제 해석은 dev/test 위주로 본다.",
        "- 핵심은 높은 절대 성능보다, `problem_train/dev/test_v2`를 바로 먹는 입력 포맷, retrieval reference, 선택지 ranking, 지표 계산 흐름을 실제로 잠갔다는 점이다.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_split(
    rows,
    split_name,
    train_rows,
    problem_vectorizer,
    train_problem_matrix,
    answer_vectorizer,
    train_answer_matrix,
    top_k,
):
    prediction_rows = []
    for row in rows:
        problem_query = problem_vectorizer.transform([build_problem_query_text(row)])
        problem_similarities = linear_kernel(problem_query, train_problem_matrix)[0]
        top_indices = sorted(range(len(train_rows)), key=lambda idx: float(problem_similarities[idx]), reverse=True)[:top_k]

        choice_lookup = build_choice_lookup(row)
        choice_matrix = answer_vectorizer.transform(choice_lookup[letter] for letter in CHOICE_LETTERS)

        aggregated_scores = [0.0] * len(CHOICE_LETTERS)
        retrieved_problem_ids = []
        retrieved_similarity_scores = []
        total_weight = 0.0

        for train_index in top_indices:
            similarity_weight = max(float(problem_similarities[train_index]), 0.0)
            if similarity_weight == 0.0:
                similarity_weight = 1e-6

            total_weight += similarity_weight
            retrieved_row = train_rows[train_index]
            retrieved_problem_ids.append(retrieved_row["problem_id"])
            retrieved_similarity_scores.append(round(similarity_weight, 6))

            answer_similarities = linear_kernel(choice_matrix, train_answer_matrix[train_index])[:, 0]
            for candidate_index, candidate_score in enumerate(answer_similarities):
                aggregated_scores[candidate_index] += similarity_weight * float(candidate_score)

        if total_weight:
            aggregated_scores = [score / total_weight for score in aggregated_scores]

        ranked_choices = sorted(
            zip(CHOICE_LETTERS, aggregated_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        predicted_choice = ranked_choices[0][0]
        gold_choice = row["correct_choice"]
        ranked_choice_letters = [letter for letter, _ in ranked_choices]
        gold_rank = ranked_choice_letters.index(gold_choice) + 1

        prediction_rows.append(
            {
                "split": split_name,
                "problem_id": row["problem_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "generated_stem": row["generated_stem"],
                "choice_a": row["choice_a"],
                "choice_b": row["choice_b"],
                "choice_c": row["choice_c"],
                "choice_d": row["choice_d"],
                "gold_correct_choice": gold_choice,
                "predicted_choice": predicted_choice,
                "choice_accuracy": round(1.0 if predicted_choice == gold_choice else 0.0, 6),
                "gold_choice_rank": gold_rank,
                "choice_reciprocal_rank": round(reciprocal_rank(gold_rank), 6),
                "top2_hit": round(1.0 if gold_rank <= 2 else 0.0, 6),
                "top3_hit": round(1.0 if gold_rank <= 3 else 0.0, 6),
                "retrieved_train_problem_ids": " | ".join(retrieved_problem_ids),
                "retrieved_train_similarity_scores": " | ".join(str(score) for score in retrieved_similarity_scores),
                "ranked_choice_order": " > ".join(ranked_choice_letters),
                "choice_a_score": round(float(aggregated_scores[0]), 6),
                "choice_b_score": round(float(aggregated_scores[1]), 6),
                "choice_c_score": round(float(aggregated_scores[2]), 6),
                "choice_d_score": round(float(aggregated_scores[3]), 6),
            }
        )
    return prediction_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", help="problem_generation version tag")
    parser.add_argument("--top-k", type=int, default=2, help="retrieval top-k")
    parser.add_argument("--run-name", default=None, help="override run directory name")
    args = parser.parse_args()

    version_tag = args.version
    train_path, dev_path, test_path = resolve_processed_split_paths(version_tag)

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    test_rows = load_jsonl(test_path)

    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d')}_problem_{version_tag}_consumer_real_dual_retrieval_ranker"
    run_dir = ANALYSIS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    problem_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )
    train_problem_matrix = problem_vectorizer.fit_transform(build_problem_query_text(row) for row in train_rows)

    answer_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )
    train_answer_matrix = answer_vectorizer.fit_transform(build_answer_reference_text(row) for row in train_rows)

    train_predictions = evaluate_split(
        rows=train_rows,
        split_name="train",
        train_rows=train_rows,
        problem_vectorizer=problem_vectorizer,
        train_problem_matrix=train_problem_matrix,
        answer_vectorizer=answer_vectorizer,
        train_answer_matrix=train_answer_matrix,
        top_k=args.top_k,
    )
    dev_predictions = evaluate_split(
        rows=dev_rows,
        split_name="dev",
        train_rows=train_rows,
        problem_vectorizer=problem_vectorizer,
        train_problem_matrix=train_problem_matrix,
        answer_vectorizer=answer_vectorizer,
        train_answer_matrix=train_answer_matrix,
        top_k=args.top_k,
    )
    test_predictions = evaluate_split(
        rows=test_rows,
        split_name="test",
        train_rows=train_rows,
        problem_vectorizer=problem_vectorizer,
        train_problem_matrix=train_problem_matrix,
        answer_vectorizer=answer_vectorizer,
        train_answer_matrix=train_answer_matrix,
        top_k=args.top_k,
    )

    train_summary = summarize_predictions("train", train_predictions)
    dev_summary = summarize_predictions("dev", dev_predictions)
    test_summary = summarize_predictions("test", test_predictions)

    summary_payload = {
        "run_name": run_name,
        "version_tag": version_tag,
        "baseline_type": "tfidf_dual_retrieval_choice_ranker",
        "top_k": args.top_k,
        "problem_train_count": len(train_rows),
        "problem_dev_count": len(dev_rows),
        "problem_test_count": len(test_rows),
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "test_summary": test_summary,
    }

    prediction_fieldnames = [
        "split",
        "problem_id",
        "seed_sample_id",
        "family_id",
        "doc_type_name",
        "source_subset",
        "generated_stem",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "gold_correct_choice",
        "predicted_choice",
        "choice_accuracy",
        "gold_choice_rank",
        "choice_reciprocal_rank",
        "top2_hit",
        "top3_hit",
        "retrieved_train_problem_ids",
        "retrieved_train_similarity_scores",
        "ranked_choice_order",
        "choice_a_score",
        "choice_b_score",
        "choice_c_score",
        "choice_d_score",
    ]

    write_json(run_dir / "summary.json", summary_payload)
    write_text(
        run_dir / "summary.md",
        build_markdown_report(
            version_tag=version_tag,
            run_name=run_name,
            run_dir=run_dir,
            config={"top_k": args.top_k},
            train_summary=train_summary,
            dev_summary=dev_summary,
            test_summary=test_summary,
        ),
    )
    write_csv(run_dir / "train_predictions.csv", train_predictions, prediction_fieldnames)
    write_csv(run_dir / "dev_predictions.csv", dev_predictions, prediction_fieldnames)
    write_csv(run_dir / "test_predictions.csv", test_predictions, prediction_fieldnames)

    # retrieval baseline도 이후 비교를 위해 그대로 재사용할 수 있게 저장한다.
    joblib.dump(
        {
            "problem_vectorizer": problem_vectorizer,
            "answer_vectorizer": answer_vectorizer,
            "train_problem_ids": [row["problem_id"] for row in train_rows],
            "train_problem_matrix": train_problem_matrix,
            "train_answer_matrix": train_answer_matrix,
            "top_k": args.top_k,
            "version_tag": version_tag,
        },
        run_dir / "model_bundle.joblib",
    )
    return summary_payload


if __name__ == "__main__":
    main()
