import csv
import io
import logging
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

import fitz


# 이 스크립트는 입력 인자를 받지 않고 바로 실행하는 형태로 고정한다.
# 현재 단계의 목표는 최종 학습용 데이터셋이 아니라, 원문을 최대한 훼손하지 않는 1차 master CSV를 확보하는 것이다.
# 그래서 파일 경로, 출력 파일명, 기대 사례 수 등은 실행 시점마다 바뀌는 값이 아니라 코드 내부 상수로 둔다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_PDF = PROJECT_ROOT / "data" / "raw" / "moleg" / "법제처_2025_법령해석사례집(상).pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "moleg"
OUTPUT_CSV = OUTPUT_DIR / "moleg_master_cases.csv"
EXPECTED_CASE_COUNT = 148

# PDF 본문에는 상단 머리글, 우측 세로 장 제목, 하단 페이지 번호가 섞여 있으므로
# 사례 파싱에는 본문 영역만 사용하고, 구조 파악에 방해가 되는 가장자리 텍스트는 제외한다.
BODY_MIN_Y = 70
BODY_MAX_Y = 680
BODY_MAX_X = 500
LINE_GROUP_Y_TOLERANCE = 3.0

# 본문에는 작은 글씨의 각주 표식이 섞여 있고, 페이지 하단에는 별도 각주 블록이 존재한다.
# 1차 산출물에서는 본문 문장과 각주를 분리해 보존해야 하므로 하단 각주 블록과 본문 안의 작은 표식은 다르게 취급한다.
FOOTNOTE_MIN_Y = 600
FOOTNOTE_MAX_FONT = 8.6
INLINE_FOOTNOTE_MAX_FONT = 8.5

# 표는 완전 구조화보다 존재 여부와 원문 보존이 우선이므로, 감지된 표를 텍스트 블록으로 풀어서 별도 컬럼에 저장한다.
ENABLE_TABLE_EXTRACTION = True

SOURCE_YEAR = "2025"
SOURCE_VOLUME = "상"
FALSE_LITERAL = "false"
TRUE_LITERAL = "true"

QUESTION_HEADING = "질의요지"
ANSWER_HEADING = "회답"
REASON_HEADING = "이유"
RECOMMENDATION_HEADING = "법령정비 권고사항"

CASE_START_PATTERN = re.compile(r"^(?:(\d+)\s+)?▶?\s*안건번호\s+(.+)$")
SECTION_PATTERN = re.compile(r"^제\s*(\d+)\s*절\s*(.+)$")
ONLY_DIGITS_PATTERN = re.compile(r"^\d+$")
AGENDA_NUMBER_PATTERN = re.compile(r"\d{2}-\d{4}")
SUBQUESTION_PATTERN = re.compile(r"(?m)^[가-하]\.")
LAW_CONTENT_PATTERN = re.compile(
    r"^(제\d+조(?:의\d+)?\(|부칙|별표|\d+\.\s|가\.\s|나\.\s|다\.\s|라\.\s|비고)"
)
LAW_TITLE_PATTERN = re.compile(
    r"^(?:\[별표[^\]]*\]\s*)?(?:[가-힣0-9･·「」\-\(\)\[\]\"'., ]+)"
    r"(?:법|법률|령|시행령|시행규칙|규칙|규정|지침|부칙|조례|훈령|고시|예규)"
    r"(?:\([^)]+\))?$"
)
PAGE_PUBLICATION_PATTERN = re.compile(
    r"(편집\s*･\s*발행|법제처\s*법령해석총괄과|제\s*작\s*･\s*인\s*쇄|홈페\s*이\s*지)"
)
LAW_REFERENCE_PATTERN = re.compile(r"제\d+조(?:의\d+)?|제\d+항|별표\s*\d+")

CSV_COLUMNS = [
    "case_id",
    "source_file",
    "source_year",
    "source_volume",
    "case_order_in_book",
    "chapter_no",
    "chapter_title",
    "section_no",
    "section_title",
    "agenda_numbers_raw",
    "title_raw",
    "case_text_raw",
    "question_raw",
    "answer_raw",
    "reason_raw",
    "related_law_raw",
    "recommendation_raw",
    "footnotes_raw",
    "tables_raw",
    "page_start",
    "page_end",
    "has_multiple_agenda_numbers",
    "has_subquestions",
    "has_footnotes",
    "has_tables",
    "parse_notes",
]

LOGGER = logging.getLogger("moleg_master_cases")
EMITTED_LIBRARY_MESSAGES = set()


def configure_logging():
    # 로그 형식은 다른 수집 스크립트와 동일하게 맞춰 두어야 실행 중 상태를 같은 방식으로 읽을 수 있다.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_info(tag, message):
    LOGGER.info("[%s] %s", tag, message)


def log_warning(tag, message):
    LOGGER.warning("[%s] %s", tag, message)


def emit_library_messages(tag, captured_output):
    # 일부 PDF 라이브러리는 logging을 거치지 않고 표준 출력으로 안내 문구를 직접 남긴다.
    # 그대로 두면 현재 스크립트의 로그 형식과 섞여서 읽기 불편하므로,
    # 캡처한 문구를 같은 로그 포맷으로 다시 내보내고 중복 문구는 한 번만 기록한다.
    if not captured_output:
        return

    for raw_line in captured_output.splitlines():
        message = normalize_joined_text(raw_line)
        if not message:
            continue

        dedupe_key = (tag, message)
        if dedupe_key in EMITTED_LIBRARY_MESSAGES:
            continue

        EMITTED_LIBRARY_MESSAGES.add(dedupe_key)
        log_info(tag, message)


def build_temp_output_path(output_path):
    # 최종 CSV를 바로 덮어쓰지 않고 임시 파일을 먼저 만든 뒤 교체해야
    # 중간 실패가 발생해도 기존 결과가 손상되지 않는다.
    return output_path.with_suffix(output_path.suffix + ".tmp")


def normalize_joined_text(text):
    if text is None:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def normalize_main_text(text):
    text = normalize_joined_text(text)
    text = re.sub(r"\s+([,.;:)\]」”])", r"\1", text)
    text = re.sub(
        r"([가-힣A-Za-z0-9」”])\s+(의|에|은|는|이|가|을|를|와|과|로|도|만)(?=\s)",
        r"\1\2",
        text,
    )
    return text


def append_piece_with_gap(parts, previous_x1, current_x0, text):
    if previous_x1 is not None and current_x0 - previous_x1 > 1.5:
        parts.append(" ")
    parts.append(text)


def build_line_text(line, drop_inline_footnote_markers):
    parts = []
    previous_x1 = None

    for span in line.get("spans", []):
        text = span.get("text", "")
        if not text:
            continue

        stripped = text.strip()
        if not stripped:
            continue

        size = span.get("size", 0.0)
        is_inline_footnote_marker = (
            drop_inline_footnote_markers
            and size <= INLINE_FOOTNOTE_MAX_FONT
            and stripped.endswith(")")
            and stripped[:-1].isdigit()
        )
        if is_inline_footnote_marker:
            continue

        x0, _, x1, _ = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
        append_piece_with_gap(parts, previous_x1, x0, text)
        previous_x1 = x1

    joined = "".join(parts)
    if drop_inline_footnote_markers:
        return normalize_main_text(joined)
    return normalize_joined_text(joined)


def build_block_text_and_sizes(block):
    line_texts = []
    span_sizes = []

    for line in block.get("lines", []):
        raw_text = build_line_text(line, drop_inline_footnote_markers=False)
        if raw_text:
            line_texts.append(raw_text)

        for span in line.get("spans", []):
            text = span.get("text", "")
            if text and text.strip():
                span_sizes.append(span.get("size", 0.0))

    return line_texts, span_sizes


def is_footnote_block(block):
    x0, y0, _, _ = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
    if x0 >= BODY_MAX_X or y0 < FOOTNOTE_MIN_Y or y0 > BODY_MAX_Y:
        return False

    line_texts, span_sizes = build_block_text_and_sizes(block)
    if not line_texts or not span_sizes:
        return False

    if max(span_sizes) > FOOTNOTE_MAX_FONT:
        return False

    return bool(re.match(r"^\d+\)", line_texts[0]))


def build_grouped_line(page_number, fragments):
    fragments = sorted(fragments, key=lambda item: item["x0"])
    raw_text = normalize_joined_text(" ".join(item["raw_text"] for item in fragments if item["raw_text"]))
    main_text = normalize_main_text(" ".join(item["main_text"] for item in fragments if item["main_text"]))

    return {
        "page": page_number,
        "y0": min(item["y0"] for item in fragments),
        "x0": min(item["x0"] for item in fragments),
        "raw_text": raw_text,
        "main_text": main_text,
    }


def build_page_lines(page_number, page):
    # 사례 경계, 장/절 제목, 본문 섹션은 줄 순서가 중요하므로
    # 블록에서 바로 문자열만 뽑지 않고, 줄 단위 좌표를 유지한 채 다시 합친다.
    body_fragments = []
    footnote_lines = []

    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        x0, y0, _, _ = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if x0 >= BODY_MAX_X or y0 < BODY_MIN_Y or y0 > BODY_MAX_Y:
            continue

        if is_footnote_block(block):
            for line in block.get("lines", []):
                raw_text = build_line_text(line, drop_inline_footnote_markers=False)
                if raw_text:
                    footnote_lines.append(
                        {
                            "page": page_number,
                            "y0": line["bbox"][1],
                            "x0": line["bbox"][0],
                            "text": raw_text,
                        }
                    )
            continue

        for line in block.get("lines", []):
            raw_text = build_line_text(line, drop_inline_footnote_markers=False)
            main_text = build_line_text(line, drop_inline_footnote_markers=True)
            if not raw_text:
                continue

            body_fragments.append(
                {
                    "page": page_number,
                    "y0": line["bbox"][1],
                    "x0": line["bbox"][0],
                    "raw_text": raw_text,
                    "main_text": main_text,
                }
            )

    body_fragments.sort(key=lambda item: (item["y0"], item["x0"]))
    grouped_lines = []
    current_group = []

    for fragment in body_fragments:
        if not current_group:
            current_group = [fragment]
            continue

        if abs(fragment["y0"] - current_group[0]["y0"]) <= LINE_GROUP_Y_TOLERANCE:
            current_group.append(fragment)
        else:
            grouped_lines.append(build_grouped_line(page_number, current_group))
            current_group = [fragment]

    if current_group:
        grouped_lines.append(build_grouped_line(page_number, current_group))

    footnote_lines.sort(key=lambda item: (item["page"], item["y0"], item["x0"]))
    return grouped_lines, footnote_lines


def build_table_text(page_number, page):
    if not ENABLE_TABLE_EXTRACTION or not hasattr(page, "find_tables"):
        return []

    table_texts = []
    # find_tables()는 표 감지 자체는 정상 수행하면서도, 권장 패키지 안내를 stdout으로 직접 출력한다.
    # 이 안내도 실행 로그의 일부로 읽히도록 캡처한 뒤 현재 스크립트와 같은 형식으로 다시 남긴다.
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        finder = page.find_tables()
    emit_library_messages("표", captured_stdout.getvalue())

    for table_index, table in enumerate(finder.tables, start=1):
        x0, y0, _, _ = table.bbox
        if x0 >= BODY_MAX_X or y0 < BODY_MIN_Y or y0 > BODY_MAX_Y:
            continue

        rows = table.extract()
        if not rows:
            continue

        row_texts = []
        for row in rows:
            cells = []
            for cell in row:
                if cell is None:
                    continue
                cell_text = normalize_joined_text(str(cell).replace("\n", " / "))
                if cell_text:
                    cells.append(cell_text)
            if cells:
                row_texts.append(" | ".join(cells))

        if row_texts:
            table_texts.append(f"[page {page_number} table {table_index}]\n" + "\n".join(row_texts))

    return table_texts


def build_page_cache(doc):
    pages = {}
    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc.load_page(page_index)
        lines, footnotes = build_page_lines(page_number, page)
        tables = build_table_text(page_number, page)
        pages[page_number] = {
            "lines": lines,
            "footnotes": footnotes,
            "tables": tables,
        }

    return pages


def is_case_start_line(text):
    return bool(CASE_START_PATTERN.match(text))


def parse_case_start_line(text):
    match = CASE_START_PATTERN.match(text)
    if not match:
        return "", ""

    return match.group(1) or "", normalize_joined_text(match.group(2))


def is_section_line(text):
    return bool(SECTION_PATTERN.match(text))


def parse_section_line(text):
    match = SECTION_PATTERN.match(text)
    if not match:
        return "", ""

    return match.group(1), normalize_joined_text(match.group(2))


def is_chapter_intro_page(page_lines):
    if any(is_case_start_line(line["main_text"]) for line in page_lines):
        return False

    if not any(ONLY_DIGITS_PATTERN.match(line["main_text"]) for line in page_lines):
        return False

    # 장 표지 페이지는 대체로 줄 수가 적고, 본문 조문 페이지와 달리 짧은 숫자 줄과 제목 줄 조합으로 구성된다.
    return len(page_lines) <= 8


def parse_chapter_intro(page_lines):
    chapter_no = ""
    chapter_title_parts = []
    section_candidates = []

    for line in sorted(page_lines, key=lambda item: (item["y0"], item["x0"])):
        text = line["main_text"]
        if not text:
            continue

        if ONLY_DIGITS_PATTERN.match(text):
            chapter_no = text
            continue

        if is_section_line(text):
            section_candidates.append(parse_section_line(text))
            continue

        chapter_title_parts.append(text)

    chapter_title = normalize_joined_text("".join(chapter_title_parts))
    section_no = ""
    section_title = ""
    if section_candidates:
        section_no, section_title = section_candidates[0]

    return chapter_no, chapter_title, section_no, section_title


def find_publication_start_page(pages, minimum_page_number):
    for page_number in sorted(pages):
        if page_number < minimum_page_number:
            continue

        page_text = "\n".join(
            line["main_text"] for line in pages[page_number]["lines"] if line["main_text"]
        )
        if PAGE_PUBLICATION_PATTERN.search(page_text):
            return page_number

    return None


def looks_like_law_title_line(text):
    if not text:
        return False

    if len(text) > 140:
        return False

    if text in {QUESTION_HEADING, ANSWER_HEADING, REASON_HEADING, RECOMMENDATION_HEADING}:
        return False

    # 관계법령 제목은 보통 법령명 한 줄로 떨어지는데,
    # 이유 문장 중간에서 법령명을 언급한 문장도 끝부분만 보면 제목처럼 보일 수 있다.
    # 조문 번호, 공포 연혁, 개정 문구가 섞인 줄은 본문 설명일 가능성이 높으므로 제목 후보에서 제외한다.
    if LAW_REFERENCE_PATTERN.search(text):
        return False

    if "일부개정된" in text or "개정된" in text:
        return False

    if text.endswith("다.") or text.endswith("습니다.") or text.endswith("바,"):
        return False

    return bool(LAW_TITLE_PATTERN.match(text))


def looks_like_law_content_line(text):
    if not text:
        return False

    # 질의가 여러 개인 사례의 소제목은 법령 조문 구조와 모양이 비슷하지만
    # 실제 관계법령 본문이 아니므로 법령 내용 줄로 간주하면 안 된다.
    if re.match(r"^[가-하]\.\s*질의", text):
        return False

    if LAW_CONTENT_PATTERN.match(text):
        return True

    return text.startswith("[별표")


def find_related_law_start(lines):
    # 관계법령은 보통 이유의 결론 뒤에 법령명 짧은 줄과 조문 줄이 연속해서 시작된다.
    # 문장 중간의 법령명 언급을 관계법령으로 오인하지 않도록, 제목형 짧은 줄과 이어지는 조문 줄 조합을 우선 본다.
    for index, text in enumerate(lines):
        if not looks_like_law_title_line(text):
            continue

        lookahead = lines[index + 1 : index + 5]
        next_law_title_index = next(
            (offset for offset, item in enumerate(lookahead, start=1) if looks_like_law_title_line(item)),
            None,
        )
        next_law_content_index = next(
            (offset for offset, item in enumerate(lookahead, start=1) if looks_like_law_content_line(item)),
            None,
        )

        # 이유 문장의 마지막이 "...법", "...시행령"으로 끝나고 다음 줄에 실제 법령명이 나오는 경우가 있다.
        # 이런 줄은 제목처럼 보여도 관계법령의 진짜 시작점이 아니므로 건너뛴다.
        if next_law_title_index is not None and (
            next_law_content_index is None or next_law_title_index < next_law_content_index
        ):
            continue

        if next_law_content_index is not None:
            return index

    return None


def join_text_slice(lines, start_index, end_index, key):
    if start_index is None or end_index is None or start_index >= end_index:
        return ""

    return normalize_joined_text(
        "\n".join(line[key] for line in lines[start_index:end_index] if line[key])
    )


def split_case_sections(case_lines, parse_notes):
    raw_texts = [line["raw_text"] for line in case_lines]
    main_texts = [line["main_text"] for line in case_lines]

    question_index = next((i for i, text in enumerate(main_texts) if text == QUESTION_HEADING), None)
    answer_index = next((i for i, text in enumerate(main_texts) if text == ANSWER_HEADING), None)
    reason_index = next((i for i, text in enumerate(main_texts) if text == REASON_HEADING), None)

    if question_index is None:
        parse_notes.append("질의요지 경계 미검출")
    if answer_index is None:
        parse_notes.append("회답 경계 미검출")
    if reason_index is None:
        parse_notes.append("이유 경계 미검출")

    title_start = 1
    title_end = question_index if question_index is not None else len(case_lines)
    title_raw = join_text_slice(case_lines, title_start, title_end, "raw_text")

    question_raw = ""
    if question_index is not None and answer_index is not None and question_index < answer_index:
        question_raw = join_text_slice(case_lines, question_index + 1, answer_index, "main_text")

    answer_raw = ""
    if answer_index is not None:
        answer_end = reason_index if reason_index is not None and answer_index < reason_index else len(case_lines)
        answer_raw = join_text_slice(case_lines, answer_index + 1, answer_end, "main_text")

    reason_raw = ""
    related_law_raw = ""
    recommendation_raw = ""

    if reason_index is not None:
        tail_lines = main_texts[reason_index + 1 :]
        recommendation_in_tail = next(
            (i for i, text in enumerate(tail_lines) if text == RECOMMENDATION_HEADING),
            None,
        )
        related_law_in_tail = find_related_law_start(tail_lines)

        if recommendation_in_tail is None and related_law_in_tail is None:
            reason_raw = normalize_joined_text("\n".join(tail_lines))
            if tail_lines:
                parse_notes.append("관계법령 경계 미검출")
        elif recommendation_in_tail is not None and (
            related_law_in_tail is None or recommendation_in_tail < related_law_in_tail
        ):
            reason_raw = normalize_joined_text("\n".join(tail_lines[:recommendation_in_tail]))
            recommendation_candidate_lines = tail_lines[recommendation_in_tail + 1 :]
            related_law_after_recommendation = find_related_law_start(recommendation_candidate_lines)
            if related_law_after_recommendation is None:
                recommendation_raw = normalize_joined_text("\n".join(recommendation_candidate_lines))
            else:
                recommendation_raw = normalize_joined_text(
                    "\n".join(recommendation_candidate_lines[:related_law_after_recommendation])
                )
                related_law_raw = normalize_joined_text(
                    "\n".join(recommendation_candidate_lines[related_law_after_recommendation:])
                )
        else:
            reason_raw = normalize_joined_text("\n".join(tail_lines[:related_law_in_tail]))
            if recommendation_in_tail is None:
                related_law_raw = normalize_joined_text("\n".join(tail_lines[related_law_in_tail:]))
            else:
                related_law_raw = normalize_joined_text(
                    "\n".join(tail_lines[related_law_in_tail:recommendation_in_tail])
                )
                recommendation_raw = normalize_joined_text(
                    "\n".join(tail_lines[recommendation_in_tail + 1 :])
                )

    # case_text_raw는 사례 본문 복구를 위한 중심 컬럼이지만, 완전한 단일 원문 컬럼은 아니다.
    # 하단 각주 블록과 표 텍스트는 각각 footnotes_raw, tables_raw로 따로 보존하므로
    # PDF 사례를 최대한 원문 그대로 다시 보려면 세 컬럼을 함께 읽는 것이 맞다.
    # 또한 사례 시작 줄은 PDF의 번호와 안건번호를 복구용 원문으로 남기기 위해 그대로 유지한다.
    case_text_raw = normalize_joined_text("\n".join(raw_texts))
    return {
        "title_raw": title_raw,
        "case_text_raw": case_text_raw,
        "question_raw": question_raw,
        "answer_raw": answer_raw,
        "reason_raw": reason_raw,
        "related_law_raw": related_law_raw,
        "recommendation_raw": recommendation_raw,
    }


def gather_case_page_data(pages, page_start, page_end):
    case_lines = []
    footnote_lines = []
    table_texts = []

    for page_number in range(page_start, page_end + 1):
        page_data = pages.get(page_number, {})
        case_lines.extend(page_data.get("lines", []))
        footnote_lines.extend(page_data.get("footnotes", []))
        table_texts.extend(page_data.get("tables", []))

    return case_lines, footnote_lines, table_texts


def build_bool_literal(value):
    return TRUE_LITERAL if value else FALSE_LITERAL


def detect_case_boundaries_and_context(pages):
    # 사례 하나를 한 행으로 두는 이유:
    # 지금 단계는 최종 학습 샘플을 만드는 단계가 아니라 원문 복구와 후속 정제의 기준점을 만드는 단계이기 때문이다.
    # 복수 안건번호나 복수 질의를 여기서 성급히 쪼개면 원문 연결 관계를 잃기 쉬우므로
    # 1차 master에서는 책의 사례 경계를 그대로 따라가고, 후속 단계에서 필요한 방식으로 분해하는 편이 안전하다.
    current_chapter_no = ""
    current_chapter_title = ""
    current_section_no = ""
    current_section_title = ""
    cases = []

    for page_number in sorted(pages):
        page_lines = pages[page_number]["lines"]
        if not page_lines:
            continue

        if is_chapter_intro_page(page_lines):
            chapter_no, chapter_title, section_no, section_title = parse_chapter_intro(page_lines)
            if chapter_no:
                current_chapter_no = chapter_no
            if chapter_title:
                current_chapter_title = chapter_title
            current_section_no = section_no
            current_section_title = section_title
            continue

        for line in sorted(page_lines, key=lambda item: (item["y0"], item["x0"])):
            text = line["main_text"]
            if not text:
                continue

            if line["y0"] <= 180 and is_section_line(text):
                current_section_no, current_section_title = parse_section_line(text)
                continue

            if is_case_start_line(text):
                _, agenda_numbers_raw = parse_case_start_line(text)
                cases.append(
                    {
                        "page_start": page_number,
                        "chapter_no": current_chapter_no,
                        "chapter_title": current_chapter_title,
                        "section_no": current_section_no,
                        "section_title": current_section_title,
                        "agenda_numbers_raw": agenda_numbers_raw,
                    }
                )
                break

    return cases


def trim_case_end_page(pages, page_start, page_end):
    # page_end는 다음 사례 시작 직전 페이지를 기계적으로 쓰지 않고,
    # 실제 사례 본문이 남아 있는 마지막 페이지를 기록한다.
    # 따라서 절 구분지, 장 표지, 빈 페이지, 발행 정보처럼 본문이 없는 페이지가 끼어 있으면
    # page_end가 다음 사례 시작 직전 페이지보다 앞에서 끝날 수 있다.
    while page_end >= page_start:
        page_lines = pages.get(page_end, {}).get("lines", [])
        if not page_lines:
            page_end -= 1
            continue

        if is_chapter_intro_page(page_lines):
            page_end -= 1
            continue

        break

    return page_end


def trim_case_lines(case_lines):
    # 사례 시작 전의 장/절 제목 줄은 제거하되, 실제 사례 시작 줄은 남긴다.
    # 이때 PDF 상에서 번호 줄과 안건번호 줄이 한 줄로 합쳐질 수 있으므로
    # case_text_raw 첫 줄에 "1 ▶ 안건번호 ..." 같은 형태가 남는 것은 의도된 보존 결과다.
    start_index = 0
    for index, line in enumerate(case_lines):
        if is_case_start_line(line["main_text"]):
            start_index = index
            break

    return case_lines[start_index:]


def assemble_case_rows(doc, pages):
    boundaries = detect_case_boundaries_and_context(pages)
    last_case_start_page = boundaries[-1]["page_start"] if boundaries else 1
    publication_start_page = find_publication_start_page(pages, last_case_start_page + 1)
    last_content_page = publication_start_page - 1 if publication_start_page else doc.page_count
    rows = []

    for case_index, boundary in enumerate(boundaries, start=1):
        page_start = boundary["page_start"]
        if case_index < len(boundaries):
            page_end = boundaries[case_index]["page_start"] - 1
        else:
            page_end = last_content_page
        page_end = trim_case_end_page(pages, page_start, page_end)

        parse_notes = []
        case_lines, footnote_lines, table_texts = gather_case_page_data(pages, page_start, page_end)
        case_lines = trim_case_lines(case_lines)
        case_start_line = next((line for line in case_lines if is_case_start_line(line["main_text"])), None)
        if not case_start_line:
            parse_notes.append("사례 시작 줄 미검출")
            agenda_numbers_raw = boundary["agenda_numbers_raw"]
        else:
            _, agenda_numbers_raw = parse_case_start_line(case_start_line["main_text"])

        sections = split_case_sections(case_lines, parse_notes)

        # 각주와 표는 1차 단계에서 완전 구조화보다 원문 보존이 우선이다.
        # 각주는 본문에서 분리한 하단 블록을 그대로 모아 footnotes_raw에 저장하고,
        # 표는 표 감지 결과를 셀 텍스트 블록으로 풀어 tables_raw에 넣어 후속 정제에서 다시 다듬을 수 있게 한다.
        footnotes_raw = normalize_joined_text(
            "\n".join(item["text"] for item in footnote_lines if item["text"])
        )
        tables_raw = normalize_joined_text("\n\n".join(table_texts))

        if not boundary["section_no"] and not boundary["section_title"]:
            parse_notes.append("절 없음")
        if table_texts:
            parse_notes.append("표 텍스트는 보조 추출 결과")

        row = {
            "case_id": f"C{case_index:04d}",
            "source_file": INPUT_PDF.name,
            "source_year": SOURCE_YEAR,
            "source_volume": SOURCE_VOLUME,
            "case_order_in_book": str(case_index),
            "chapter_no": boundary["chapter_no"],
            "chapter_title": boundary["chapter_title"],
            "section_no": boundary["section_no"],
            "section_title": boundary["section_title"],
            "agenda_numbers_raw": agenda_numbers_raw,
            "title_raw": sections["title_raw"],
            "case_text_raw": sections["case_text_raw"],
            "question_raw": sections["question_raw"],
            "answer_raw": sections["answer_raw"],
            "reason_raw": sections["reason_raw"],
            "related_law_raw": sections["related_law_raw"],
            "recommendation_raw": sections["recommendation_raw"],
            "footnotes_raw": footnotes_raw,
            "tables_raw": tables_raw,
            "page_start": str(page_start),
            "page_end": str(page_end),
            "has_multiple_agenda_numbers": build_bool_literal(
                len(AGENDA_NUMBER_PATTERN.findall(agenda_numbers_raw)) > 1
            ),
            "has_subquestions": build_bool_literal(
                bool(SUBQUESTION_PATTERN.search(sections["question_raw"]))
                or bool(SUBQUESTION_PATTERN.search(sections["answer_raw"]))
            ),
            "has_footnotes": build_bool_literal(bool(footnotes_raw)),
            "has_tables": build_bool_literal(bool(tables_raw)),
            "parse_notes": "; ".join(dict.fromkeys(parse_notes)),
        }
        rows.append(row)

        log_info(
            "사례",
            (
                f"{case_index}/{len(boundaries)} 사례 처리 중, "
                f"시작 {page_start}쪽 종료 {page_end}쪽, "
                f"누적 저장 예정 {len(rows)}건"
            ),
        )

    return rows


def write_rows(rows):
    temp_output_csv = build_temp_output_path(OUTPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(temp_output_csv):
        os.remove(temp_output_csv)

    with open(temp_output_csv, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        saved_count = 0
        for row in rows:
            writer.writerow(row)
            saved_count += 1

            if saved_count % 10 == 0 or saved_count == len(rows):
                csv_file.flush()
                log_info("저장", f"현재까지 총 {saved_count}건 저장")

    os.replace(temp_output_csv, OUTPUT_CSV)


def main():
    configure_logging()

    if not INPUT_PDF.exists():
        raise FileNotFoundError(f"입력 PDF를 찾을 수 없습니다: {INPUT_PDF}")

    # 왜 PDF를 바로 정규화하지 않고 원문 보존형 CSV를 먼저 만드는가:
    # 법령해석 사례집은 장/절 구조, 복수 안건번호, 복수 질의, 각주, 표, 권고사항이 섞인 문서라서
    # 초기에 의미 단위로 강하게 정규화하면 되돌리기 어려운 손실이 생길 수 있다.
    # 따라서 먼저 사례 단위 master CSV를 만들어 복구 가능한 기준점을 확보하고,
    # 이후 단계에서 학습 목적에 맞는 분해나 정제를 수행하는 편이 안전하다.
    log_info("시작", f"입력 PDF 분석 시작: {INPUT_PDF}")
    document = fitz.open(INPUT_PDF)

    try:
        pages = build_page_cache(document)
        rows = assemble_case_rows(document, pages)
        write_rows(rows)
    finally:
        document.close()

    final_count = len(rows)
    log_info("완료", f"최종 총 행 수: {final_count}")
    log_info("완료", f"CSV 저장 완료: {OUTPUT_CSV}")

    if final_count != EXPECTED_CASE_COUNT:
        log_warning(
            "경고",
            f"기대 사례 수 {EXPECTED_CASE_COUNT}건과 실제 저장 수 {final_count}건이 다릅니다.",
        )


if __name__ == "__main__":
    main()
