import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path


# 이 스크립트는 현재 문서에 적어 둔 'v1' 브리지 정렬 검증을
# 다시 같은 방식으로 재현하기 위한 분석기다.
# 실행 시점마다 표본이 달라지면 결과 비교가 어렵기 때문에 입력 경로, 표본 수,
# 성공 기준, 출력 파일명은 모두 코드 내부 상수로 고정한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

QA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aihub"
    / "03.행정법 LLM 사전학습 및 Instruction Tuning 데이터"
    / "3.개방데이터"
    / "1.데이터"
    / "Training"
    / "02.라벨링데이터"
    / "TL_해석례_QA"
)
TRAINING_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aihub"
    / "03.행정법 LLM 사전학습 및 Instruction Tuning 데이터"
    / "3.개방데이터"
    / "1.데이터"
    / "Training"
    / "01.원천데이터"
    / "TS_해석례"
)
ALL_RAW_GLOB = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aihub"
    / "03.행정법 LLM 사전학습 및 Instruction Tuning 데이터"
)

# 생성 산출물은 학습셋이나 공식 상태 문서가 아니라, 재실행 가능한 분석 결과물이므로
# 로컬 보조 폴더가 아니라 루트 'analysis' 아래에 둬서 스크립트 결과 위치를 명시적으로 맞춘다.
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "aihub"
OUTPUT_STEM = "aihub_03_interpretation_bridge_v1_validation"
OUTPUT_CSV = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
OUTPUT_JSON = OUTPUT_DIR / f"{OUTPUT_STEM}.json"
OUTPUT_MD = OUTPUT_DIR / f"{OUTPUT_STEM}.md"

SAMPLE_SIZE = 20
SUCCESS_THRESHOLD_PERCENT = 85.0
REQUIRED_SECTIONS = ("질의요지", "회답", "이유")

LOGGER = logging.getLogger("aihub_bridge_v1")


def configure_logging():
    # 다른 수집/추출 스크립트와 같은 로그 형식을 유지해야 실행 로그를 한 눈에 읽기 쉽다.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_info(tag, message):
    LOGGER.info("[%s] %s", tag, message)


def build_temp_output_path(output_path):
    # 결과 파일을 바로 덮어쓰지 않고 임시 파일에 먼저 쓰면
    # 분석 중간 실패가 나도 기존 산출물이 깨지지 않는다.
    return output_path.with_suffix(output_path.suffix + ".tmp")


def write_text_atomic(output_path, text):
    temp_path = build_temp_output_path(output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)


def write_json_atomic(output_path, payload):
    temp_path = build_temp_output_path(output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    temp_path.replace(output_path)


def write_csv_atomic(output_path, rows, fieldnames):
    temp_path = build_temp_output_path(output_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_path)


def load_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def list_qa_files(qa_dir):
    # AI Hub 폴더에는 실제 샘플과 별도로 '.extract_complete.json' marker가 함께 있으므로
    # 샘플 집계와 정렬 검증에서는 marker를 반드시 제외해야 문서의 실제 파일 수와 맞는다.
    return sorted(
        path
        for path in qa_dir.glob("*.json")
        if path.name != ".extract_complete.json"
    )


def load_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def build_sample_indices(total_count, sample_size):
    # 표본을 사람이 임의로 고르면 이후 재현성이 떨어진다.
    # 그래서 정렬된 전체 파일 목록에서 균등 간격으로 고정 추출한다.
    if total_count == 0:
        return []
    if sample_size <= 1 or total_count == 1:
        return [0]

    indices = []
    for order in range(sample_size):
        index = round(order * (total_count - 1) / (sample_size - 1))
        if index not in indices:
            indices.append(index)
    return indices


def count_sentences(text):
    # 현재 문서에서 잠근 기준과 동일하게, '이유' 전체를 이어 붙인 뒤
    # 종결 부호 단위로 문장 수를 세어 '2문장 이상' 여부를 판단한다.
    if not text:
        return 0
    sentences = [piece.strip() for piece in re.split(r"[.!?]+\s*", text) if piece.strip()]
    return len(sentences)


def collect_section_text(rows, section_name):
    return " ".join(row["내용"].strip() for row in rows if row["구분"] == section_name)


def build_raw_index(raw_paths):
    raw_index = {}
    for raw_path in raw_paths:
        rows = load_csv_rows(raw_path)
        if not rows:
            continue

        interpre_id = rows[0]["해석례일련번호"].strip()
        raw_index[interpre_id] = {
            "raw_path": raw_path,
            "rows": rows,
        }
    return raw_index


def summarize_match_rate(qa_paths, raw_id_set):
    qa_ids = []
    for qa_path in qa_paths:
        qa_payload = load_json(qa_path)
        qa_ids.append(str(qa_payload["info"]["interpreId"]).strip())

    matched_count = sum(1 for qa_id in qa_ids if qa_id in raw_id_set)
    unmatched_ids = sorted({qa_id for qa_id in qa_ids if qa_id not in raw_id_set})
    matched_rate = round(matched_count / len(qa_ids) * 100, 2) if qa_ids else 0.0

    return {
        "qa_count": len(qa_ids),
        "matched_count": matched_count,
        "matched_rate_percent": matched_rate,
        "unmatched_count": len(qa_ids) - matched_count,
        "unmatched_unique_ids": len(unmatched_ids),
        "sample_unmatched_ids": unmatched_ids[:20],
    }


def build_sample_rows(qa_paths, training_raw_index, all_raw_index, sample_indices):
    sample_rows = []
    valid_count = 0

    for sample_order, sample_index in enumerate(sample_indices, start=1):
        qa_path = qa_paths[sample_index]
        qa_payload = load_json(qa_path)

        interpre_id = str(qa_payload["info"]["interpreId"]).strip()
        agenda_num = qa_payload["info"]["agendaNum"]
        label_input = qa_payload["label"]["input"]
        label_output = qa_payload["label"]["output"]

        training_raw_entry = training_raw_index.get(interpre_id)
        any_raw_entry = all_raw_index.get(interpre_id)

        row = {
            "sample_order": sample_order,
            "sample_index": sample_index,
            "interpre_id": interpre_id,
            "agenda_num": agenda_num,
            "qa_file": qa_path.name,
            "raw_exists_in_training": "예" if training_raw_entry else "아니오",
            "raw_exists_anywhere_in_03": "예" if any_raw_entry else "아니오",
            "has_required_sections": "-",
            "reason_sentence_count": 0,
            "verdict": "실패",
            "failure_reason": "raw 미매칭",
            "qa_input": label_input,
            "qa_output": label_output,
            "raw_file": "",
            "raw_question_preview": "",
            "raw_answer_preview": "",
            "raw_reason_preview": "",
        }

        if not training_raw_entry:
            sample_rows.append(row)
            continue

        raw_rows = training_raw_entry["rows"]
        section_names = {raw_row["구분"] for raw_row in raw_rows}
        question_text = collect_section_text(raw_rows, "질의요지")
        answer_text = collect_section_text(raw_rows, "회답")
        reason_text = collect_section_text(raw_rows, "이유")
        reason_sentence_count = count_sentences(reason_text)

        has_required_sections = all(section in section_names for section in REQUIRED_SECTIONS)
        is_valid = has_required_sections and reason_sentence_count >= 2

        row.update(
            {
                "has_required_sections": "예" if has_required_sections else "아니오",
                "reason_sentence_count": reason_sentence_count,
                "verdict": "유효" if is_valid else "실패",
                "failure_reason": (
                    ""
                    if is_valid
                    else "구간 누락 또는 이유 2문장 미만"
                ),
                "raw_file": training_raw_entry["raw_path"].name,
                "raw_question_preview": question_text[:300],
                "raw_answer_preview": answer_text[:300],
                "raw_reason_preview": reason_text[:300],
            }
        )

        if is_valid:
            valid_count += 1

        sample_rows.append(row)

    sample_success_rate = round(valid_count / len(sample_rows) * 100, 2) if sample_rows else 0.0
    return sample_rows, valid_count, sample_success_rate


def build_markdown_report(summary, sample_rows):
    lines = []
    lines.append(f"# `{OUTPUT_STEM}`")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(
        f"- 표본 기준 유효 수: `{summary['sample_valid_count']}/{summary['sample_count']}`"
    )
    lines.append(
        f"- 표본 기준 브리지 정렬 성공률: `{summary['sample_success_rate_percent']}%`"
    )
    lines.append(
        f"- `v1` 통과 기준: `{SUCCESS_THRESHOLD_PERCENT}%`"
    )
    lines.append(
        f"- fallback 발동 여부: `{summary['fallback_triggered']}`"
    )
    lines.append(
        f"- `Training TL_해석례_QA` 전체 매칭률: `{summary['training_match_rate_percent']}%`"
    )
    lines.append(
        f"- `03` 전체 raw 보조 검산 매칭률: `{summary['all_raw_match_rate_percent']}%`"
    )
    lines.append("")
    lines.append("## 표본 추출 기준")
    lines.append("")
    lines.append(
        f"- 정렬된 `Training TL_해석례_QA` `{summary['qa_count']}`개에서 균등 간격 표본 `{summary['sample_count']}`개를 고정 추출"
    )
    lines.append(
        f"- 사용 인덱스: `{', '.join(str(index) for index in summary['sample_indices'])}`"
    )
    lines.append("")
    lines.append("## 표본 검증 결과")
    lines.append("")
    lines.append("| 번호 | interpreId | agendaNum | training raw 존재 | 03 전체 raw 존재 | 구간 존재 | 이유 문장 수 | 판정 | 실패 이유 |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |")
    for row in sample_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_order"]),
                    f"`{row['interpre_id']}`",
                    f"`{row['agenda_num']}`",
                    row["raw_exists_in_training"],
                    row["raw_exists_anywhere_in_03"],
                    row["has_required_sections"],
                    str(row["reason_sentence_count"]),
                    row["verdict"],
                    row["failure_reason"] or "-",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 참고")
    lines.append("")
    lines.append("- `training raw 존재`는 현재 `v1` 기준 원천인 `03 Training TS_해석례` 매칭 여부다.")
    lines.append("- `03 전체 raw 존재`는 같은 `interpreId`가 `03` 배포셋 전체 raw 어디에도 없는지 보조 검산한 값이다.")
    lines.append("- 상세 미리보기와 전체 메타는 같은 stem의 `csv`와 `json` 결과 파일에 함께 저장한다.")
    lines.append("")
    return "\n".join(lines)


def main():
    configure_logging()

    qa_paths = list_qa_files(QA_DIR)
    training_raw_paths = sorted(TRAINING_RAW_DIR.glob("*.csv"))
    all_raw_paths = sorted(ALL_RAW_GLOB.glob("**/TS_해석례/*.csv"))

    if not qa_paths:
        raise FileNotFoundError(f"QA 파일을 찾지 못했습니다: {QA_DIR}")
    if not training_raw_paths:
        raise FileNotFoundError(f"Training raw 파일을 찾지 못했습니다: {TRAINING_RAW_DIR}")

    log_info("LOAD", f"QA {len(qa_paths)}건, training raw {len(training_raw_paths)}건, all raw {len(all_raw_paths)}건 확인")

    training_raw_index = build_raw_index(training_raw_paths)
    all_raw_index = build_raw_index(all_raw_paths)

    sample_indices = build_sample_indices(len(qa_paths), SAMPLE_SIZE)
    sample_rows, valid_count, sample_success_rate = build_sample_rows(
        qa_paths=qa_paths,
        training_raw_index=training_raw_index,
        all_raw_index=all_raw_index,
        sample_indices=sample_indices,
    )

    training_summary = summarize_match_rate(qa_paths, set(training_raw_index.keys()))
    all_raw_summary = summarize_match_rate(qa_paths, set(all_raw_index.keys()))

    summary_payload = {
        "analysis_name": OUTPUT_STEM,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_count": len(qa_paths),
        "training_raw_count": len(training_raw_paths),
        "all_raw_count": len(all_raw_paths),
        "sample_count": len(sample_rows),
        "sample_indices": sample_indices,
        "sample_valid_count": valid_count,
        "sample_invalid_count": len(sample_rows) - valid_count,
        "sample_success_rate_percent": sample_success_rate,
        "success_threshold_percent": SUCCESS_THRESHOLD_PERCENT,
        "fallback_triggered": sample_success_rate < SUCCESS_THRESHOLD_PERCENT,
        "training_match_count": training_summary["matched_count"],
        "training_match_rate_percent": training_summary["matched_rate_percent"],
        "training_unmatched_count": training_summary["unmatched_count"],
        "all_raw_match_count": all_raw_summary["matched_count"],
        "all_raw_match_rate_percent": all_raw_summary["matched_rate_percent"],
        "all_raw_unmatched_count": all_raw_summary["unmatched_count"],
        "sample_failed_interpre_ids": [
            row["interpre_id"] for row in sample_rows if row["verdict"] != "유효"
        ],
        "sample_rows": sample_rows,
        "training_missing_id_examples": training_summary["sample_unmatched_ids"],
    }

    csv_fieldnames = [
        "sample_order",
        "sample_index",
        "interpre_id",
        "agenda_num",
        "qa_file",
        "raw_file",
        "raw_exists_in_training",
        "raw_exists_anywhere_in_03",
        "has_required_sections",
        "reason_sentence_count",
        "verdict",
        "failure_reason",
        "qa_input",
        "qa_output",
        "raw_question_preview",
        "raw_answer_preview",
        "raw_reason_preview",
    ]

    write_csv_atomic(OUTPUT_CSV, sample_rows, csv_fieldnames)
    write_json_atomic(OUTPUT_JSON, summary_payload)
    write_text_atomic(OUTPUT_MD, build_markdown_report(summary_payload, sample_rows))

    log_info(
        "DONE",
        (
            f"표본 {len(sample_rows)}건 중 {valid_count}건 유효 "
            f"({sample_success_rate}%). 결과 저장: {OUTPUT_CSV}, {OUTPUT_JSON}, {OUTPUT_MD}"
        ),
    )


if __name__ == "__main__":
    main()
