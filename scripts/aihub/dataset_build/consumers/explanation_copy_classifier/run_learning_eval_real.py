import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "aihub" / "dataset_build" / "realrun"


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


def build_model_input(row):
    # 실제 학습에서는 문제 본문만 쓰기보다 문서유형과 해설 목표를 함께 넣어
    # 샘플 수가 작아도 라우팅 경계가 조금 더 선명하게 잡히도록 한다.
    fields = [
        f"문서유형: {row.get('doc_type_name', '')}",
        f"대분류: {row.get('source_subset', '')}",
        f"정답모드: {row.get('answer_mode', '')}",
        f"해설목표: {row.get('explanation_target', '')}",
        f"문제: {row.get('transformed_problem', '')}",
    ]
    return "\n".join(normalize_text(field) for field in fields if normalize_text(field))


def select_margin_score(model, texts):
    decision_values = model.decision_function(texts)
    margin_scores = []

    if len(model.classes_) == 2:
        for value in decision_values:
            margin_scores.append(abs(float(value)))
        return margin_scores

    for row in decision_values:
        ordered_scores = sorted(float(value) for value in row)
        if len(ordered_scores) >= 2:
            margin_scores.append(ordered_scores[-1] - ordered_scores[-2])
        elif ordered_scores:
            margin_scores.append(ordered_scores[0])
        else:
            margin_scores.append(0.0)
    return margin_scores


def evaluate_split(model, train_lookup, split_name, rows):
    if not rows:
        return []

    inputs = [build_model_input(row) for row in rows]
    predicted_sample_ids = model.predict(inputs)
    margin_scores = select_margin_score(model, inputs)

    prediction_rows = []
    for row, predicted_sample_id, margin_score in zip(rows, predicted_sample_ids, margin_scores):
        copied_row = train_lookup[predicted_sample_id]
        predicted_short_answer = copied_row["short_answer"]
        predicted_explanation = copied_row["generated_explanation"]

        prediction_rows.append(
            {
                "split": split_name,
                "sample_id": row["sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "family_id": row["family_id"],
                "predicted_train_sample_id": predicted_sample_id,
                "predicted_train_doc_type_name": copied_row["doc_type_name"],
                "predicted_train_source_subset": copied_row["source_subset"],
                "margin_score": round(float(margin_score), 6),
                "gold_short_answer": row["short_answer"],
                "predicted_short_answer": predicted_short_answer,
                "gold_generated_explanation": row["generated_explanation"],
                "predicted_generated_explanation": predicted_explanation,
                "sample_routing_exact_match": round(exact_match(row["sample_id"], predicted_sample_id), 6),
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
            "sample_routing_exact_match": 0.0,
            "short_answer_exact_match": 0.0,
            "short_answer_token_f1": 0.0,
            "explanation_token_f1": 0.0,
            "doc_type_breakdown": {},
        }

    summary = {
        "split": split_name,
        "row_count": len(rows),
        "sample_routing_exact_match": round(mean(row["sample_routing_exact_match"] for row in rows), 6),
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
            "sample_routing_exact_match": round(mean(row["sample_routing_exact_match"] for row in doc_rows), 6),
            "short_answer_exact_match": round(mean(row["short_answer_exact_match"] for row in doc_rows), 6),
            "short_answer_token_f1": round(mean(row["short_answer_token_f1"] for row in doc_rows), 6),
            "explanation_token_f1": round(mean(row["explanation_token_f1"] for row in doc_rows), 6),
        }

    return summary


def build_markdown_report(version_tag, run_dir, train_rows, train_summary, dev_summary, test_summary):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {version_tag} 실제 학습/평가 baseline 결과",
        "",
        f"- 생성 시각: {generated_at}",
        f"- 출력 경로: `{run_dir}`",
        "- baseline 종류: `tfidf_linear_svc_copy_classifier`",
        "- 학습 방식: `transformed_problem + 문서유형/정답모드/해설목표`를 입력으로 받아 train sample id를 예측한 뒤, 해당 train sample의 short_answer와 generated_explanation을 복원",
        "- 목적: downstream handoff 검산을 넘어, 현재 셋으로 실제 학습 가능한 baseline 1개를 잠그고 첫 real train/eval run을 남기는 것",
        "",
        "## 데이터 개요",
        "",
        f"- train: {len(train_rows)}",
        f"- dev: {dev_summary['row_count']}",
        f"- test: {test_summary['row_count']}",
        "",
        "## split별 요약",
        "",
        "| split | rows | sample routing EM | short_answer EM | short_answer token F1 | explanation token F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_summary['row_count']} | {train_summary['sample_routing_exact_match']:.4f} | {train_summary['short_answer_exact_match']:.4f} | {train_summary['short_answer_token_f1']:.4f} | {train_summary['explanation_token_f1']:.4f} |",
        f"| dev | {dev_summary['row_count']} | {dev_summary['sample_routing_exact_match']:.4f} | {dev_summary['short_answer_exact_match']:.4f} | {dev_summary['short_answer_token_f1']:.4f} | {dev_summary['explanation_token_f1']:.4f} |",
        f"| test | {test_summary['row_count']} | {test_summary['sample_routing_exact_match']:.4f} | {test_summary['short_answer_exact_match']:.4f} | {test_summary['short_answer_token_f1']:.4f} | {test_summary['explanation_token_f1']:.4f} |",
        "",
        "## 해석 메모",
        "",
        "- 이 결과는 LLM 최종 성능 보고가 아니라, 실제 학습 가능한 baseline 1개를 현재 셋에 연결해 end-to-end train/eval 경로를 잠갔다는 의미로 읽는다.",
        "- 샘플 수가 매우 작고 각 sample_id가 사실상 one-shot class에 가까워 점수 자체는 낮을 수 있다.",
        "- 그럼에도 이 런은 `train/dev/test` 소비, 모델 fit, 예측 생성, 지표 계산, 산출물 저장까지 실제 학습 스택이 작동한다는 증거다.",
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

    train_inputs = [build_model_input(row) for row in train_rows]
    train_labels = [row["sample_id"] for row in train_rows]
    train_lookup = {row["sample_id"]: row for row in train_rows}

    # 현재 데이터 규모에서는 대형 생성 모델 미세조정보다,
    # 재현 가능한 경량 실제 학습 baseline을 먼저 잠그는 것이 더 실무적이다.
    model = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=2.0,
                ),
            ),
        ]
    )
    model.fit(train_inputs, train_labels)

    train_predictions = evaluate_split(model, train_lookup, "train", train_rows)
    dev_predictions = evaluate_split(model, train_lookup, "dev", dev_rows)
    test_predictions = evaluate_split(model, train_lookup, "test", test_rows)

    train_summary = summarize_predictions("train", train_predictions)
    dev_summary = summarize_predictions("dev", dev_predictions)
    test_summary = summarize_predictions("test", test_predictions)

    run_name = f"{datetime.now().strftime('%Y-%m-%d')}_{version_tag}_learning_eval_real_copyclf"
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

    joblib.dump(model, run_dir / "model.joblib")

    summary_payload = {
        "version_tag": version_tag,
        "run_name": run_name,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "test_count": len(test_rows),
        "baseline_type": "tfidf_linear_svc_copy_classifier",
        "model_artifact": str(run_dir / "model.joblib"),
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "test_summary": test_summary,
    }
    write_json(run_dir / "summary.json", summary_payload)
    write_text(
        run_dir / "summary.md",
        build_markdown_report(version_tag, run_dir, train_rows, train_summary, dev_summary, test_summary),
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
