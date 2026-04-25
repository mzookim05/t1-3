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
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "aihub" / "dataset_build" / "dryrun"


def resolve_processed_split_paths(version_tag: str) -> tuple[Path, Path, Path]:
    # 파일명이 아니라 버전 폴더로 split을 관리하므로, legacy tag 입력을 현재 폴더명으로 변환한다.
    folder_by_version = {
        "v6": "v6_generalization_full_01_04",
        "v7": "v7_strict_final",
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


def build_idf(train_rows):
    document_frequency = Counter()
    for row in train_rows:
        tokens = set(tokenize(row["transformed_problem"]))
        for token in tokens:
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


def exact_match(gold_text, predicted_text):
    return 1.0 if normalize_text(gold_text) == normalize_text(predicted_text) else 0.0


def retrieve_best(train_rows, query_row, idf):
    if not train_rows:
        raise ValueError("train split이 비어 있어 검색형 dryrun 기준선을 만들 수 없습니다.")

    query_tokens = tokenize(query_row["transformed_problem"])
    best_row = None
    best_score = -1.0
    best_reasons = {}

    for candidate in train_rows:
        candidate_tokens = tokenize(candidate["transformed_problem"])
        lexical_score = weighted_jaccard(query_tokens, candidate_tokens, idf)
        doc_type_bonus = 0.10 if candidate["doc_type_name"] == query_row["doc_type_name"] else 0.0
        answer_mode_bonus = 0.05 if candidate.get("answer_mode") == query_row.get("answer_mode") else 0.0
        lane_bonus = 0.02 if candidate.get("sampling_lane") == query_row.get("sampling_lane") else 0.0
        final_score = lexical_score + doc_type_bonus + answer_mode_bonus + lane_bonus

        if final_score > best_score:
            best_score = final_score
            best_row = candidate
            best_reasons = {
                "lexical_score": lexical_score,
                "doc_type_bonus": doc_type_bonus,
                "answer_mode_bonus": answer_mode_bonus,
                "lane_bonus": lane_bonus,
            }

    # Pylance가 이후 best_row 접근을 안전하게 좁힐 수 있도록 비어 있는 검색 결과를 명시적으로 차단한다.
    if best_row is None:
        raise RuntimeError("검색 후보를 찾지 못했습니다.")
    return best_row, best_score, best_reasons


def evaluate_split(train_rows, eval_rows, idf):
    prediction_rows = []
    for row in eval_rows:
        best_row, retrieval_score, reasons = retrieve_best(train_rows, row, idf)
        predicted_short_answer = best_row["short_answer"]
        predicted_explanation = best_row["generated_explanation"]

        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "family_id": row["family_id"],
                "best_train_sample_id": best_row["sample_id"],
                "best_train_family_id": best_row["family_id"],
                "retrieval_score": round(retrieval_score, 6),
                "lexical_score": round(reasons["lexical_score"], 6),
                "doc_type_bonus": round(reasons["doc_type_bonus"], 6),
                "answer_mode_bonus": round(reasons["answer_mode_bonus"], 6),
                "lane_bonus": round(reasons["lane_bonus"], 6),
                "gold_short_answer": row["short_answer"],
                "predicted_short_answer": predicted_short_answer,
                "gold_generated_explanation": row["generated_explanation"],
                "predicted_generated_explanation": predicted_explanation,
                "short_answer_exact_match": round(exact_match(row["short_answer"], predicted_short_answer), 6),
                "short_answer_token_f1": round(token_f1(row["short_answer"], predicted_short_answer), 6),
                "explanation_token_f1": round(token_f1(row["generated_explanation"], predicted_explanation), 6),
            }
        )
    return prediction_rows


def summarize_predictions(split_name, rows):
    if not rows:
        return {
            "split": split_name,
            "row_count": 0,
            "short_answer_exact_match": 0.0,
            "short_answer_token_f1": 0.0,
            "explanation_token_f1": 0.0,
            "doc_type_breakdown": {},
        }

    summary = {
        "split": split_name,
        "row_count": len(rows),
        "short_answer_exact_match": round(mean(row["short_answer_exact_match"] for row in rows), 6),
        "short_answer_token_f1": round(mean(row["short_answer_token_f1"] for row in rows), 6),
        "explanation_token_f1": round(mean(row["explanation_token_f1"] for row in rows), 6),
        "doc_type_breakdown": {},
    }

    rows_by_doc_type = defaultdict(list)
    for row in rows:
        rows_by_doc_type[row["doc_type_name"]].append(row)

    for doc_type_name, doc_rows in rows_by_doc_type.items():
        summary["doc_type_breakdown"][doc_type_name] = {
            "row_count": len(doc_rows),
            "short_answer_exact_match": round(mean(row["short_answer_exact_match"] for row in doc_rows), 6),
            "short_answer_token_f1": round(mean(row["short_answer_token_f1"] for row in doc_rows), 6),
            "explanation_token_f1": round(mean(row["explanation_token_f1"] for row in doc_rows), 6),
        }

    return summary


def build_markdown_report(version_tag, run_dir, train_rows, dev_summary, test_summary):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {version_tag} 학습/평가 드라이런 결과",
        "",
        f"- 생성 시각: {generated_at}",
        f"- 출력 경로: `{run_dir}`",
        f"- 드라이런 방식: `transformed_problem` 기반 검색형 baseline",
        f"- 목적: 최종셋 handoff 이후 downstream 학습·평가 경로가 실제로 끊기지 않는지 최소 비용으로 검산",
        "",
        "## 데이터 개요",
        "",
        f"- train: {len(train_rows)}",
        f"- dev: {dev_summary['row_count']}",
        f"- test: {test_summary['row_count']}",
        "",
        "## split별 요약",
        "",
        "| split | rows | short_answer EM | short_answer token F1 | explanation token F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| dev | {dev_summary['row_count']} | {dev_summary['short_answer_exact_match']:.4f} | {dev_summary['short_answer_token_f1']:.4f} | {dev_summary['explanation_token_f1']:.4f} |",
        f"| test | {test_summary['row_count']} | {test_summary['short_answer_exact_match']:.4f} | {test_summary['short_answer_token_f1']:.4f} | {test_summary['explanation_token_f1']:.4f} |",
        "",
        "## 해석 메모",
        "",
        "- 이 결과는 최종 모델 성능 주장이 아니라, `train/dev/test` JSONL이 실제 downstream 평가 입력으로 바로 소비되는지 확인하기 위한 검산용 baseline이다.",
        "- 검색형 baseline이므로 점수 자체보다 데이터 로딩, 필드 일관성, split별 예측 산출, 지표 계산이 끝까지 닫혔는지가 더 중요하다.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v7", help="dataset_build 버전 태그")
    args = parser.parse_args()

    version_tag = args.version
    train_path, dev_path, test_path = resolve_processed_split_paths(version_tag)

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    test_rows = load_jsonl(test_path)

    # 현재 단계는 큰 학습 스택을 얹기보다 handoff 품질을 확인하는 것이 목적이므로,
    # 외부 의존성 없이 바로 재현 가능한 검색형 baseline으로 드라이런을 고정한다.
    idf = build_idf(train_rows)
    dev_predictions = evaluate_split(train_rows, dev_rows, idf)
    test_predictions = evaluate_split(train_rows, test_rows, idf)

    dev_summary = summarize_predictions("dev", dev_predictions)
    test_summary = summarize_predictions("test", test_predictions)

    # 같은 날짜에 여러 consumer run이 생겨도 폴더명만으로 실행 순서가 보이도록 초 단위까지 고정한다.
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{version_tag}_learning_eval_dryrun"
    run_dir = ANALYSIS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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

    summary_payload = {
        "version_tag": version_tag,
        "run_name": run_name,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "test_count": len(test_rows),
        "baseline_type": "retrieval_dryrun",
        "dev_summary": dev_summary,
        "test_summary": test_summary,
    }
    write_json(run_dir / "summary.json", summary_payload)
    write_text(run_dir / "summary.md", build_markdown_report(version_tag, run_dir, train_rows, dev_summary, test_summary))

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
