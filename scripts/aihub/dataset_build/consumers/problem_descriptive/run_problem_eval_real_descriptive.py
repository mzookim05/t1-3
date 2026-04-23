import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "aihub" / "dataset_build" / "realrun"


def resolve_processed_split_paths(version_tag: str) -> tuple[Path, Path, Path]:
    # descriptive consumer는 subtype/batch별 processed 폴더를 직접 읽어 v1/v3/pb1 결과를 분리한다.
    folder_by_version = {
        "v1": "v1_descriptive",
        "v3": "v3_split_descriptive",
        "pb1_descriptive": "production_batches/pb1_descriptive",
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


def tokenize(text):
    return re.findall(r"[0-9A-Za-z가-힣]+", normalize_text(text).lower())


def content_tokens(text):
    # descriptive consumer에서는 조사 수준의 우연 일치를 줄이기 위해
    # 길이 2 이상 토큰과 숫자 토큰만 support 판단에 쓴다.
    return [token for token in tokenize(text) if len(token) >= 2 or token.isdigit()]


def exact_match(gold_text, predicted_text):
    return 1.0 if normalize_text(gold_text) == normalize_text(predicted_text) else 0.0


def token_f1(gold_text, predicted_text):
    gold_tokens = tokenize(gold_text)
    predicted_tokens = tokenize(predicted_text)
    if not gold_tokens and not predicted_tokens:
        return 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0

    gold_counter = Counter(gold_tokens)
    predicted_counter = Counter(predicted_tokens)
    overlap = sum((gold_counter & predicted_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_problem_query_text(row):
    # `v3`는 선택지가 없는 descriptive line이므로,
    # 질문 본문과 쟁점 메타를 같이 써서 problem retrieval 축을 고정한다.
    fields = [
        f"문서유형: {row.get('doc_type_name', '')}",
        f"대분류: {row.get('source_subset', '')}",
        f"정답모드: {row.get('answer_mode', '')}",
        f"문제유형: {row.get('problem_task_type', '')}",
        f"쟁점: {row.get('focus_issue', '')}",
        f"문제: {row.get('generated_problem', '')}",
    ]
    return "\n".join(normalize_text(field) for field in fields if normalize_text(field))


def build_answer_query_text(row):
    # answer reference reranking에서는 split memo와 focus issue를 같이 넣어
    # top-k 안에서 현재 질문과 더 가까운 설명 축을 고르게 한다.
    fields = [
        f"문서유형: {row.get('doc_type_name', '')}",
        f"대분류: {row.get('source_subset', '')}",
        f"쟁점: {row.get('focus_issue', '')}",
        f"분리힌트: {row.get('split_focus_hint', '')}",
        f"문제: {row.get('generated_problem', '')}",
    ]
    return "\n".join(normalize_text(field) for field in fields if normalize_text(field))


def build_answer_reference_text(row):
    # descriptive line은 choice가 없으므로 train row의 정답과 reference explanation을
    # 함께 묶어 retrieval reference로 사용한다.
    fields = [row.get("gold_short_answer", ""), row.get("gold_reference_explanation", "")]
    return "\n".join(normalize_text(field) for field in fields if normalize_text(field))


def build_idf(train_rows):
    document_frequency = Counter()
    for row in train_rows:
        token_set = set(content_tokens(build_problem_query_text(row)) + content_tokens(build_answer_reference_text(row)))
        for token in token_set:
            document_frequency[token] += 1

    total_documents = max(len(train_rows), 1)
    return {
        token: math.log((1 + total_documents) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def weighted_jaccard(query_tokens, candidate_tokens, idf):
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    all_tokens = set(query_counter) | set(candidate_counter)
    if not all_tokens:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for token in all_tokens:
        weight = idf.get(token, 1.0)
        numerator += min(query_counter[token], candidate_counter[token]) * weight
        denominator += max(query_counter[token], candidate_counter[token]) * weight
    return numerator / denominator if denominator else 0.0


def rank_candidates(train_rows, eval_row, idf):
    problem_query_tokens = content_tokens(build_problem_query_text(eval_row))
    answer_query_tokens = content_tokens(build_answer_query_text(eval_row))
    ranked = []

    for candidate in train_rows:
        candidate_problem_tokens = content_tokens(build_problem_query_text(candidate))
        candidate_answer_tokens = content_tokens(build_answer_reference_text(candidate))

        problem_score = weighted_jaccard(problem_query_tokens, candidate_problem_tokens, idf)
        answer_score = weighted_jaccard(answer_query_tokens, candidate_answer_tokens, idf)

        # 문서유형과 source subset을 같이 맞춰 주면,
        # 작은 파일럿에서도 retrieval이 완전히 엉뚱한 축으로 튀는 것을 조금 줄일 수 있다.
        doc_type_bonus = 0.08 if candidate.get("doc_type_name") == eval_row.get("doc_type_name") else 0.0
        subset_bonus = 0.04 if candidate.get("source_subset") == eval_row.get("source_subset") else 0.0
        answer_mode_bonus = 0.03 if candidate.get("answer_mode") == eval_row.get("answer_mode") else 0.0

        combined_score = (0.60 * problem_score) + (0.30 * answer_score) + doc_type_bonus + subset_bonus + answer_mode_bonus

        ranked.append(
            {
                "problem_id": candidate["problem_id"],
                "problem_score": round(problem_score, 6),
                "answer_score": round(answer_score, 6),
                "doc_type_bonus": round(doc_type_bonus, 6),
                "subset_bonus": round(subset_bonus, 6),
                "answer_mode_bonus": round(answer_mode_bonus, 6),
                "combined_score": round(combined_score, 6),
                "row": candidate,
            }
        )

    ranked.sort(key=lambda item: item["combined_score"], reverse=True)
    return ranked


def answer_overlap_count(gold_text, candidate_text):
    gold_tokens = set(content_tokens(gold_text))
    candidate_tokens = set(content_tokens(candidate_text))
    return len(gold_tokens & candidate_tokens)


def summarize_predictions(split_name, rows):
    if not rows:
        return {
            "split": split_name,
            "row_count": 0,
            "normalized_exact_match": 0.0,
            "short_answer_token_f1": 0.0,
            "reference_explanation_token_f1": 0.0,
            "top2_answer_lexical_hit": 0.0,
            "top3_answer_lexical_hit": 0.0,
            "doc_type_breakdown": {},
        }

    summary = {
        "split": split_name,
        "row_count": len(rows),
        "normalized_exact_match": round(mean(row["normalized_exact_match"] for row in rows), 6),
        "short_answer_token_f1": round(mean(row["short_answer_token_f1"] for row in rows), 6),
        "reference_explanation_token_f1": round(mean(row["reference_explanation_token_f1"] for row in rows), 6),
        "top2_answer_lexical_hit": round(mean(row["top2_answer_lexical_hit"] for row in rows), 6),
        "top3_answer_lexical_hit": round(mean(row["top3_answer_lexical_hit"] for row in rows), 6),
        "doc_type_breakdown": {},
    }

    rows_by_doc_type = defaultdict(list)
    for row in rows:
        rows_by_doc_type[row["doc_type_name"]].append(row)

    for doc_type_name, doc_rows in rows_by_doc_type.items():
        summary["doc_type_breakdown"][doc_type_name] = {
            "row_count": len(doc_rows),
            "normalized_exact_match": round(mean(row["normalized_exact_match"] for row in doc_rows), 6),
            "short_answer_token_f1": round(mean(row["short_answer_token_f1"] for row in doc_rows), 6),
            "reference_explanation_token_f1": round(mean(row["reference_explanation_token_f1"] for row in doc_rows), 6),
            "top2_answer_lexical_hit": round(mean(row["top2_answer_lexical_hit"] for row in doc_rows), 6),
            "top3_answer_lexical_hit": round(mean(row["top3_answer_lexical_hit"] for row in doc_rows), 6),
        }

    return summary


def build_markdown_report(version_tag, run_name, run_dir, config, train_summary, dev_summary, test_summary):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {version_tag} problem descriptive consumer baseline 결과",
        "",
        f"- 생성 시각: {generated_at}",
        f"- run 이름: `{run_name}`",
        f"- 출력 경로: `{run_dir}`",
        "- baseline 종류: `lexical_problem_answer_support_ranker`",
        f"- retrieval top-k: `{config['top_k']}`",
        f"- lexical support 최소 겹침 기준: `{config['support_overlap_min']}`",
        "- 목적: `problem_generation v3 split-descriptive line`이 만든 `problem_train/dev/test_v3.jsonl`을 직접 소비하는 첫 lightweight real baseline을 잠그는 것",
        "",
        "## 소비 구조",
        "",
        "- 입력: `problem_train/dev/test_v3.jsonl`의 generated problem, focus issue, split meta",
        "- retrieval query: `doc_type_name + source_subset + answer_mode + problem_task_type + focus_issue + generated_problem`",
        "- answer reference: train split의 `gold_short_answer + gold_reference_explanation`",
        "- 예측 방식: lexical retrieval로 train row를 ranking한 뒤, top-1 train row의 short answer와 reference explanation을 복사한다",
        "- 평가지표: normalized EM, short answer token F1, reference explanation token F1, top-2/top-3 answer lexical hit",
        "",
        "## split별 요약",
        "",
        "| split | rows | normalized EM | short answer token F1 | reference explanation token F1 | top-2 lexical hit | top-3 lexical hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_summary['row_count']} | {train_summary['normalized_exact_match']:.4f} | {train_summary['short_answer_token_f1']:.4f} | {train_summary['reference_explanation_token_f1']:.4f} | {train_summary['top2_answer_lexical_hit']:.4f} | {train_summary['top3_answer_lexical_hit']:.4f} |",
        f"| dev | {dev_summary['row_count']} | {dev_summary['normalized_exact_match']:.4f} | {dev_summary['short_answer_token_f1']:.4f} | {dev_summary['reference_explanation_token_f1']:.4f} | {dev_summary['top2_answer_lexical_hit']:.4f} | {dev_summary['top3_answer_lexical_hit']:.4f} |",
        f"| test | {test_summary['row_count']} | {test_summary['normalized_exact_match']:.4f} | {test_summary['short_answer_token_f1']:.4f} | {test_summary['reference_explanation_token_f1']:.4f} | {test_summary['top2_answer_lexical_hit']:.4f} | {test_summary['top3_answer_lexical_hit']:.4f} |",
        "",
        "## 해석 메모",
        "",
        "- 이 baseline은 `LLM` fine-tuning이나 생성형 추론 모델이 아니라, 현재 `v3` descriptive problem 셋을 실제로 소비하는 경량 retrieval-copy baseline이다.",
        "- train 점수는 self-retrieval이 포함되므로 sanity check 성격으로 읽고, 실제 해석은 dev/test 위주로 본다.",
        "- descriptive split line은 family hold-out이라 EM이 낮을 수 있다. 따라서 이번 결과는 높은 절대 성능보다 `v3` 입력 포맷, retrieval, 예측 산출, 지표 계산 흐름이 실제로 닫혔다는 점에 더 의미가 있다.",
        "- `top-k lexical hit`는 정답 완전 일치가 아니라, retrieved answer들 안에 현재 gold short answer와 lexical support를 일부 공유하는 row가 들어오는지를 보는 보조 지표다.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_split(rows, split_name, train_rows, idf, top_k, support_overlap_min):
    prediction_rows = []
    for row in rows:
        ranked_candidates = rank_candidates(train_rows, row, idf)
        top_candidates = ranked_candidates[:top_k]
        predicted_row = ranked_candidates[0]["row"]

        top_overlap_counts = [
            answer_overlap_count(row["gold_short_answer"], candidate["row"]["gold_short_answer"])
            for candidate in ranked_candidates
        ]

        prediction_rows.append(
            {
                "split": split_name,
                "problem_id": row["problem_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "focus_issue": row["focus_issue"],
                "generated_problem": row["generated_problem"],
                "predicted_train_problem_id": predicted_row["problem_id"],
                "predicted_train_doc_type_name": predicted_row["doc_type_name"],
                "predicted_train_source_subset": predicted_row["source_subset"],
                "gold_short_answer": row["gold_short_answer"],
                "predicted_short_answer": predicted_row["gold_short_answer"],
                "gold_reference_explanation": row["gold_reference_explanation"],
                "predicted_reference_explanation": predicted_row["gold_reference_explanation"],
                "normalized_exact_match": round(exact_match(row["gold_short_answer"], predicted_row["gold_short_answer"]), 6),
                "short_answer_token_f1": round(token_f1(row["gold_short_answer"], predicted_row["gold_short_answer"]), 6),
                "reference_explanation_token_f1": round(
                    token_f1(row["gold_reference_explanation"], predicted_row["gold_reference_explanation"]),
                    6,
                ),
                "top2_answer_lexical_hit": round(
                    1.0 if max(top_overlap_counts[: min(2, len(top_overlap_counts))], default=0) >= support_overlap_min else 0.0,
                    6,
                ),
                "top3_answer_lexical_hit": round(
                    1.0 if max(top_overlap_counts[: min(3, len(top_overlap_counts))], default=0) >= support_overlap_min else 0.0,
                    6,
                ),
                "retrieved_problem_ids": " | ".join(candidate["problem_id"] for candidate in top_candidates),
                "retrieved_combined_scores": " | ".join(str(candidate["combined_score"]) for candidate in top_candidates),
                "retrieved_problem_scores": " | ".join(str(candidate["problem_score"]) for candidate in top_candidates),
                "retrieved_answer_scores": " | ".join(str(candidate["answer_score"]) for candidate in top_candidates),
                "retrieved_answer_overlap_counts": " | ".join(str(count) for count in top_overlap_counts[: len(top_candidates)]),
            }
        )
    return prediction_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3", help="problem_generation descriptive 버전 태그")
    parser.add_argument("--top-k", type=int, default=3, help="retrieval top-k")
    parser.add_argument("--support-overlap-min", type=int, default=1, help="top-k lexical hit를 셀 최소 content-token overlap")
    parser.add_argument("--run-name", default=None, help="override run directory name")
    args = parser.parse_args()

    version_tag = args.version
    train_path, dev_path, test_path = resolve_processed_split_paths(version_tag)

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    test_rows = load_jsonl(test_path)

    idf = build_idf(train_rows)

    train_predictions = evaluate_split(
        rows=train_rows,
        split_name="train",
        train_rows=train_rows,
        idf=idf,
        top_k=args.top_k,
        support_overlap_min=args.support_overlap_min,
    )
    dev_predictions = evaluate_split(
        rows=dev_rows,
        split_name="dev",
        train_rows=train_rows,
        idf=idf,
        top_k=args.top_k,
        support_overlap_min=args.support_overlap_min,
    )
    test_predictions = evaluate_split(
        rows=test_rows,
        split_name="test",
        train_rows=train_rows,
        idf=idf,
        top_k=args.top_k,
        support_overlap_min=args.support_overlap_min,
    )

    train_summary = summarize_predictions("train", train_predictions)
    dev_summary = summarize_predictions("dev", dev_predictions)
    test_summary = summarize_predictions("test", test_predictions)

    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d')}_problem_{version_tag}_consumer_real_descriptive_ranker"
    run_dir = ANALYSIS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        run_dir / "train_predictions.csv",
        train_predictions,
        list(train_predictions[0].keys()) if train_predictions else [],
    )
    write_csv(
        run_dir / "dev_predictions.csv",
        dev_predictions,
        list(dev_predictions[0].keys()) if dev_predictions else [],
    )
    write_csv(
        run_dir / "test_predictions.csv",
        test_predictions,
        list(test_predictions[0].keys()) if test_predictions else [],
    )

    config = {
        "baseline_name": "lexical_problem_answer_support_ranker",
        "version_tag": version_tag,
        "top_k": args.top_k,
        "support_overlap_min": args.support_overlap_min,
        "idf_vocab_size": len(idf),
        "train_problem_ids": [row["problem_id"] for row in train_rows],
    }
    write_json(run_dir / "model_bundle.json", config)

    summary_payload = {
        "run_name": run_name,
        "config": config,
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "test_summary": test_summary,
        "problem_train_count": len(train_rows),
        "problem_dev_count": len(dev_rows),
        "problem_test_count": len(test_rows),
    }
    write_json(run_dir / "summary.json", summary_payload)
    write_text(
        run_dir / "summary.md",
        build_markdown_report(version_tag, run_name, run_dir, config, train_summary, dev_summary, test_summary),
    )


if __name__ == "__main__":
    main()
