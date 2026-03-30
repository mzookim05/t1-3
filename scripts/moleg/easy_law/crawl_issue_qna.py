import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from common import (
    EASYLAW_BASE_URL,
    PROJECT_ROOT,
    build_bool_literal,
    build_session,
    canonicalize_url,
    clean_text,
    configure_logging,
    extract_query_value,
    fetch_html,
    log_info,
    log_warning,
    make_sha1,
    write_csv,
)


# 이슈Q&A는 책자형 leaf node 탭이 아니라 별도 목록형 게시판이므로
# targetRow 기준 페이징을 직접 순회하는 전용 수집 경로를 사용한다.
LIST_URL_TEMPLATE = (
    EASYLAW_BASE_URL
    + "/CSP/IssueQaLstRetrieve.laf?topMenu=serviceUl7&search_put=&txtKeyword="
    + "&REQUEST_DATA_SINGLE_MODE=true&targetRow={target_row}"
)
MOLEG_BASE_URL = "https://www.moleg.go.kr"
LIST_PAGE_SIZE = 20

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "moleg" / "easy_law"
QNA_OUTPUT_CSV = OUTPUT_DIR / "issue_qna.csv"
ATTACHMENTS_OUTPUT_CSV = OUTPUT_DIR / "issue_qna_attachments.csv"

# 기존 생활법령 계열 수집기와 동일하게 디버깅 범위를 코드 상수로만 제한한다.
MAX_LIST_PAGES = None
MAX_ISSUES = None

QNA_COLUMNS = [
    "issue_id",
    "issueqa_seq",
    "list_page_url",
    "list_page_index",
    "page_target_row",
    "list_order_in_page",
    "board_number",
    "detail_url",
    "detail_url_canonical",
    "issue_title",
    "issue_topic",
    "registered_date",
    "source_content_url",
    "source_content_url_canonical",
    "source_content_key_param",
    "thumbnail_url",
    "thumbnail_alt",
    "body_structure_type",
    "question_raw",
    "question_outline_raw",
    "question_outline_count",
    "answer_raw",
    "body_text_raw",
    "body_text_with_markers",
    "body_html_raw",
    "body_tables_raw",
    "has_table",
    "body_table_count",
    "has_inline_qna",
    "has_attachment",
    "attachment_count",
    "attachment_notice_raw",
    "crawled_at",
]
ATTACHMENT_COLUMNS = [
    "issue_id",
    "issueqa_seq",
    "attachment_order",
    "attachment_role",
    "attachment_label",
    "attachment_title",
    "attachment_url",
    "attachment_url_canonical",
    "attachment_bid_param",
    "attachment_list_key_param",
    "attachment_seq_param",
    "crawled_at",
]

INLINE_QNA_PATTERN = re.compile(
    r"(?:^|\s)Q\s*[.．:：]\s*(?P<question>.+?)\s+A\s*[.．:：]\s*(?P<answer>.+)",
    flags=re.DOTALL,
)
BOARD_COUNT_PATTERN = re.compile(r"전체게시물\s*:\s*([0-9,]+)")
PAGE_INDEX_PATTERN = re.compile(r"\[([0-9]+)\s*/\s*([0-9]+)\]")
ISSUE_TOPIC_PATTERN = re.compile(r"^\[([^\]]+)\]")
OUTLINE_QUESTION_PATTERN = re.compile(r"^\d+\.\s*")


def build_list_url(target_row):
    # 첫 페이지도 targetRow=1 형식으로 고정하면
    # 목록 페이지 이동 규칙을 한 함수 안에서 일관되게 유지할 수 있다.
    return canonicalize_url(
        LIST_URL_TEMPLATE.format(target_row=target_row),
        EASYLAW_BASE_URL,
    )


def parse_issue_topic(issue_title):
    match = ISSUE_TOPIC_PATTERN.match(issue_title or "")
    return clean_text(match.group(1)) if match else ""


def parse_list_head_info(soup):
    # 게시물 수와 현재 페이지 정보를 같이 읽어 두면
    # 중간 실패 시 어느 목록 페이지까지 순회했는지 로그만으로 복원하기 쉽다.
    head = soup.select_one("div.vote_list div.head div.fL")
    raw_text = clean_text(head.get_text(" ", strip=True)) if head else ""

    total_posts = 0
    current_page = 0
    total_pages = 0

    count_match = BOARD_COUNT_PATTERN.search(raw_text)
    if count_match:
        total_posts = int(count_match.group(1).replace(",", ""))

    page_match = PAGE_INDEX_PATTERN.search(raw_text)
    if page_match:
        current_page = int(page_match.group(1))
        total_pages = int(page_match.group(2))

    return total_posts, current_page, total_pages


def parse_list_page(html, list_page_final_url, page_index, page_target_row):
    # 이슈Q&A는 목록 한 행이 상세 1건과 1:1 대응하므로
    # 목록 메타를 먼저 정리해 두고 상세 파서가 이를 이어받도록 만든다.
    soup = BeautifulSoup(html, "html.parser")
    total_posts, current_page, total_pages = parse_list_head_info(soup)
    rows = []

    for row_index, row in enumerate(soup.select("table.normal tbody tr"), start=1):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        detail_anchor = row.select_one("td.txtL.st2 a[href]")
        if detail_anchor is None:
            continue

        detail_url = urljoin(list_page_final_url, detail_anchor["href"])
        detail_url_canonical = canonicalize_url(detail_anchor["href"], list_page_final_url)
        issueqa_seq = extract_query_value(detail_url_canonical, "issueqaSeq")
        if not issueqa_seq:
            continue

        source_content_anchor = row.select_one("td.csmImg a[href]")
        source_content_url = (
            urljoin(MOLEG_BASE_URL, source_content_anchor["href"])
            if source_content_anchor is not None
            else ""
        )
        source_content_url_canonical = (
            canonicalize_url(source_content_anchor["href"], MOLEG_BASE_URL)
            if source_content_anchor is not None
            else ""
        )

        thumbnail_image = row.select_one("td.csmImg img")
        thumbnail_url = (
            canonicalize_url(thumbnail_image["src"], list_page_final_url)
            if thumbnail_image is not None and thumbnail_image.get("src")
            else ""
        )
        thumbnail_alt = (
            clean_text(thumbnail_image.get("alt", ""))
            if thumbnail_image is not None
            else ""
        )

        issue_title = clean_text(detail_anchor.get_text(" ", strip=True))
        rows.append(
            {
                "issue_id": make_sha1(f"issueqa:{issueqa_seq}")[:16],
                "issueqa_seq": issueqa_seq,
                "list_page_url": list_page_final_url,
                "list_page_index": str(page_index),
                "page_target_row": str(page_target_row),
                "list_order_in_page": str(row_index),
                "board_number": clean_text(cells[0].get_text(" ", strip=True)),
                "detail_url": detail_url,
                "detail_url_canonical": detail_url_canonical,
                "issue_title": issue_title,
                "issue_topic": parse_issue_topic(issue_title),
                "registered_date": clean_text(cells[3].get_text(" ", strip=True)),
                "source_content_url": source_content_url,
                "source_content_url_canonical": source_content_url_canonical,
                "source_content_key_param": extract_query_value(
                    source_content_url_canonical, "leg_nl_pst_seq"
                ),
                "thumbnail_url": thumbnail_url,
                "thumbnail_alt": thumbnail_alt,
            }
        )

    return rows, total_posts, current_page, total_pages


def build_html_table_text(table):
    # 표가 레이아웃용인지 내용용인지 사전에 완전히 구분하기 어려워
    # raw 단계에서는 모든 표를 셀 경계를 남긴 문자열로 함께 보존한다.
    row_texts = []
    for row in table.select("tr"):
        cells = []
        for cell in row.select("th, td"):
            cell_text = clean_text(cell.get_text("\n", strip=True))
            if cell_text:
                cells.append(cell_text.replace("\n", " / "))
        if cells:
            row_texts.append(" | ".join(cells))
    return "\n".join(row_texts)


def extract_body_variants(body_cell):
    # 이슈Q&A 본문은 핵심 Q/A가 표 안에 들어가는 경우가 많아서
    # Q/A 추출용 전체 텍스트는 따로 보되, 저장 컬럼 자체는 기존 수집기와 맞춰
    # 1) 표 제외 본문 2) 표 위치 marker 포함 본문 3) 원문 HTML 4) 표 덤프를 분리한다.
    body_html_raw = str(body_cell) if body_cell else ""
    if not body_html_raw:
        return "", "", "", "", False, 0, ""

    raw_soup = BeautifulSoup(body_html_raw, "html.parser")
    raw_root = raw_soup.find("td")
    if raw_root is None:
        return "", "", body_html_raw, "", False, 0, ""

    full_body_text_compact = clean_text(raw_root.get_text(" ", strip=True))

    marker_soup = BeautifulSoup(body_html_raw, "html.parser")
    marker_root = marker_soup.find("td")
    if marker_root is None:
        return "", "", body_html_raw, "", False, 0, full_body_text_compact

    table_blocks = []
    table_count = 0
    for table_index, table in enumerate(marker_root.select("table"), start=1):
        table_count += 1
        table_text = build_html_table_text(table)
        marker_text = f"[table {table_index}]"
        if table_text:
            table_blocks.append(f"{marker_text}\n{table_text}")
            table.replace_with(f"\n{marker_text}\n")
        else:
            table_blocks.append(marker_text)
            table.replace_with(f"\n{marker_text}\n")

    body_text_with_markers = clean_text(marker_root.get_text("\n", strip=True))
    plain_soup = BeautifulSoup(body_html_raw, "html.parser")
    plain_root = plain_soup.find("td")
    if plain_root is None:
        return (
            "",
            body_text_with_markers,
            body_html_raw,
            "\n\n".join(table_blocks),
            bool(table_blocks),
            table_count,
            full_body_text_compact,
        )

    # 다른 생활법령 수집기와 마찬가지로 plain 본문은 표를 제거한 버전으로 둔다.
    for table in plain_root.select("table"):
        table.decompose()

    body_text_raw = clean_text(plain_root.get_text("\n", strip=True))
    return (
        body_text_raw,
        body_text_with_markers,
        body_html_raw,
        "\n\n".join(table_blocks),
        bool(table_blocks),
        table_count,
        full_body_text_compact,
    )


def extract_inline_qna(body_text_compact):
    # 최신 이슈Q&A 상당수는 본문 시작부에 Q./A.가 직접 들어가므로
    # 별도 질문/답변 필드를 같이 남겨 두면 후속 QA 전환이 쉬워진다.
    if not body_text_compact:
        return "", "", False

    match = INLINE_QNA_PATTERN.search(body_text_compact)
    if not match:
        return "", "", False

    question_text = clean_text(match.group("question"))
    answer_text = clean_text(match.group("answer"))
    if not question_text or not answer_text:
        return "", "", False

    return question_text, answer_text, True


def extract_outline_questions(body_text_raw, has_inline_qna):
    # 구형 이슈Q&A는 월간 묶음 소개문과 번호형 질문 목록만 두고
    # 상세 답변은 첨부파일에 의존하는 경우가 있어 질문 목록을 따로 보존한다.
    if has_inline_qna or not body_text_raw:
        return "", 0

    outline_lines = []
    for line in body_text_raw.split("\n"):
        if OUTLINE_QUESTION_PATTERN.match(line):
            outline_lines.append(line)

    return "\n".join(outline_lines), len(outline_lines)


def build_body_structure_type(has_inline_qna, outline_question_count, body_text_raw):
    if has_inline_qna:
        return "inline_qna"
    if outline_question_count > 0:
        return "question_outline_only"
    if body_text_raw:
        return "body_only"
    return "empty_body"


def parse_attachment_role(anchor):
    class_names = set(anchor.get("class", []))
    if "bl-down" in class_names:
        return "download"
    if "bl-view" in class_names:
        return "view"
    if "issueTtl" in class_names:
        return "title_link"
    return "other"


def parse_attachment_rows(attachment_cell, issue_id, issueqa_seq, crawled_at):
    # 일부 시기에는 첨부파일이 없거나 첨부 셀 자체가 빠져 있으므로
    # 링크 row와 안내 문구를 분리해 두어 "첨부 없음"과 "파싱 실패"를 구분한다.
    if attachment_cell is None:
        return [], ""

    anchors = attachment_cell.select("a[href]")
    if not anchors:
        return [], clean_text(attachment_cell.get_text(" ", strip=True))

    rows = []
    for attachment_order, anchor in enumerate(anchors, start=1):
        attachment_url = urljoin(MOLEG_BASE_URL, anchor["href"])
        attachment_url_canonical = canonicalize_url(anchor["href"], MOLEG_BASE_URL)
        rows.append(
            {
                "issue_id": issue_id,
                "issueqa_seq": issueqa_seq,
                "attachment_order": str(attachment_order),
                "attachment_role": parse_attachment_role(anchor),
                "attachment_label": clean_text(anchor.get_text(" ", strip=True)),
                "attachment_title": clean_text(anchor.get("title", "")),
                "attachment_url": attachment_url,
                "attachment_url_canonical": attachment_url_canonical,
                "attachment_bid_param": extract_query_value(attachment_url_canonical, "bid"),
                "attachment_list_key_param": extract_query_value(
                    attachment_url_canonical, "list_key"
                ),
                "attachment_seq_param": extract_query_value(attachment_url_canonical, "seq"),
                "crawled_at": crawled_at,
            }
        )

    return rows, ""


def build_empty_issue_row(list_item, attachment_notice_raw, crawled_at):
    # 상세 페이지 구조가 예상과 달라도 목록 메타 자체는 남겨 두어야
    # 나중에 어떤 issueqaSeq가 비정상 응답이었는지 다시 확인할 수 있다.
    return {
        "issue_id": list_item["issue_id"],
        "issueqa_seq": list_item["issueqa_seq"],
        "list_page_url": list_item["list_page_url"],
        "list_page_index": list_item["list_page_index"],
        "page_target_row": list_item["page_target_row"],
        "list_order_in_page": list_item["list_order_in_page"],
        "board_number": list_item["board_number"],
        "detail_url": list_item["detail_url"],
        "detail_url_canonical": list_item["detail_url_canonical"],
        "issue_title": list_item["issue_title"],
        "issue_topic": list_item["issue_topic"],
        "registered_date": list_item["registered_date"],
        "source_content_url": list_item["source_content_url"],
        "source_content_url_canonical": list_item["source_content_url_canonical"],
        "source_content_key_param": list_item["source_content_key_param"],
        "thumbnail_url": list_item["thumbnail_url"],
        "thumbnail_alt": list_item["thumbnail_alt"],
        "body_structure_type": "empty_body",
        "question_raw": "",
        "question_outline_raw": "",
        "question_outline_count": "0",
        "answer_raw": "",
        "body_text_raw": "",
        "body_text_with_markers": "",
        "body_html_raw": "",
        "body_tables_raw": "",
        "has_table": build_bool_literal(False),
        "body_table_count": "0",
        "has_inline_qna": build_bool_literal(False),
        "has_attachment": build_bool_literal(False),
        "attachment_count": "0",
        "attachment_notice_raw": attachment_notice_raw,
        "crawled_at": crawled_at,
    }


def parse_detail_page(html, detail_final_url, list_item):
    # 상세 페이지는 제목 링크, 본문 셀, 첨부 셀 순서가 비교적 안정적이어서
    # 셀 단위로 나눠 본문과 첨부 링크를 함께 보존한다.
    soup = BeautifulSoup(html, "html.parser")
    crawled_at = time.strftime("%Y-%m-%d %H:%M:%S")
    cells = soup.select("table.normal tbody tr td.file.st4")

    if not cells:
        log_warning("issue", f"missing_detail_cells issueqa_seq={list_item['issueqa_seq']}")
        return build_empty_issue_row(list_item, "missing_detail_cells", crawled_at), []

    title_anchor = cells[0].select_one("a[href]") if len(cells) >= 1 else None
    body_cell = cells[1] if len(cells) >= 2 else None
    attachment_cell = cells[2] if len(cells) >= 3 else None

    (
        body_text_raw,
        body_text_with_markers,
        body_html_raw,
        body_tables_raw,
        has_table,
        body_table_count,
        compact_body_text,
    ) = extract_body_variants(body_cell)
    question_raw, answer_raw, has_inline_qna = extract_inline_qna(compact_body_text)
    question_outline_raw, question_outline_count = extract_outline_questions(
        body_text_raw,
        has_inline_qna,
    )
    attachment_rows, attachment_notice_raw = parse_attachment_rows(
        attachment_cell=attachment_cell,
        issue_id=list_item["issue_id"],
        issueqa_seq=list_item["issueqa_seq"],
        crawled_at=crawled_at,
    )

    issue_title = (
        clean_text(title_anchor.get_text(" ", strip=True))
        if title_anchor is not None
        else list_item["issue_title"]
    )
    source_content_url = (
        urljoin(MOLEG_BASE_URL, title_anchor["href"])
        if title_anchor is not None and title_anchor.get("href")
        else list_item["source_content_url"]
    )
    source_content_url_canonical = (
        canonicalize_url(title_anchor["href"], MOLEG_BASE_URL)
        if title_anchor is not None and title_anchor.get("href")
        else list_item["source_content_url_canonical"]
    )

    issue_row = {
        "issue_id": list_item["issue_id"],
        "issueqa_seq": list_item["issueqa_seq"],
        "list_page_url": list_item["list_page_url"],
        "list_page_index": list_item["list_page_index"],
        "page_target_row": list_item["page_target_row"],
        "list_order_in_page": list_item["list_order_in_page"],
        "board_number": list_item["board_number"],
        "detail_url": detail_final_url,
        "detail_url_canonical": canonicalize_url(detail_final_url, EASYLAW_BASE_URL),
        "issue_title": issue_title,
        "issue_topic": parse_issue_topic(issue_title),
        "registered_date": list_item["registered_date"],
        "source_content_url": source_content_url,
        "source_content_url_canonical": source_content_url_canonical,
        "source_content_key_param": extract_query_value(
            source_content_url_canonical, "leg_nl_pst_seq"
        ),
        "thumbnail_url": list_item["thumbnail_url"],
        "thumbnail_alt": list_item["thumbnail_alt"],
        "body_structure_type": build_body_structure_type(
            has_inline_qna=has_inline_qna,
            outline_question_count=question_outline_count,
            body_text_raw=body_text_raw,
        ),
        "question_raw": question_raw,
        "question_outline_raw": question_outline_raw,
        "question_outline_count": str(question_outline_count),
        "answer_raw": answer_raw,
        "body_text_raw": body_text_raw,
        "body_text_with_markers": body_text_with_markers,
        "body_html_raw": body_html_raw,
        "body_tables_raw": body_tables_raw,
        "has_table": build_bool_literal(has_table),
        "body_table_count": str(body_table_count),
        "has_inline_qna": build_bool_literal(has_inline_qna),
        "has_attachment": build_bool_literal(bool(attachment_rows)),
        "attachment_count": str(len(attachment_rows)),
        "attachment_notice_raw": attachment_notice_raw,
        "crawled_at": crawled_at,
    }
    return issue_row, attachment_rows


def flush_outputs(issue_rows, attachment_rows):
    # 목록 페이지를 여러 번 도는 구조라서
    # 페이지 하나가 끝날 때마다 checkpoint 저장을 남겨야 중간 실패 복구가 쉽다.
    write_csv(QNA_OUTPUT_CSV, QNA_COLUMNS, issue_rows)
    write_csv(ATTACHMENTS_OUTPUT_CSV, ATTACHMENT_COLUMNS, attachment_rows)


def crawl_issue_qna():
    configure_logging()
    session = build_session()

    seen_issueqa_seqs = set()
    issue_rows = []
    attachment_rows = []

    target_row = 1
    page_index = 1
    total_pages = None

    while True:
        if MAX_LIST_PAGES is not None and page_index > MAX_LIST_PAGES:
            break

        list_url = build_list_url(target_row)
        list_html, list_final_url = fetch_html(session, list_url)
        list_items, total_posts, current_page, parsed_total_pages = parse_list_page(
            html=list_html,
            list_page_final_url=list_final_url,
            page_index=page_index,
            page_target_row=target_row,
        )
        if not list_items:
            log_warning("list", f"empty_page target_row={target_row}")
            break

        if parsed_total_pages:
            total_pages = parsed_total_pages

        log_info(
            "list",
            (
                f"page_index={page_index} "
                f"target_row={target_row} "
                f"item_count={len(list_items)} "
                f"current_page={current_page or page_index} "
                f"total_pages={total_pages or 0} "
                f"total_posts={total_posts}"
            ),
        )

        for list_item in list_items:
            issueqa_seq = list_item["issueqa_seq"]
            if issueqa_seq in seen_issueqa_seqs:
                continue

            seen_issueqa_seqs.add(issueqa_seq)
            try:
                detail_html, detail_final_url = fetch_html(
                    session,
                    list_item["detail_url_canonical"],
                    referer=list_final_url,
                )
                issue_row, page_attachment_rows = parse_detail_page(
                    html=detail_html,
                    detail_final_url=detail_final_url,
                    list_item=list_item,
                )
                issue_rows.append(issue_row)
                attachment_rows.extend(page_attachment_rows)
            except requests.RequestException as error:
                # 목록 row를 이미 확보한 상태에서 상세 요청 하나가 실패했다고
                # 전체 수집을 중단하면 나머지 issueqaSeq를 잃게 되므로 경고 후 계속 진행한다.
                log_warning("issue", f"skip issueqa_seq={issueqa_seq} error={error}")

            if MAX_ISSUES is not None and len(issue_rows) >= MAX_ISSUES:
                break

        flush_outputs(issue_rows, attachment_rows)
        log_info(
            "checkpoint",
            (
                f"page_index={page_index} saved "
                f"issue_rows={len(issue_rows)} "
                f"attachment_rows={len(attachment_rows)}"
            ),
        )

        if MAX_ISSUES is not None and len(issue_rows) >= MAX_ISSUES:
            break
        if len(list_items) < LIST_PAGE_SIZE:
            break
        if total_pages is not None and page_index >= total_pages:
            break

        page_index += 1
        target_row += LIST_PAGE_SIZE

    flush_outputs(issue_rows, attachment_rows)
    log_info("done", f"issue_rows={len(issue_rows)}")
    log_info("done", f"attachment_rows={len(attachment_rows)}")


if __name__ == "__main__":
    crawl_issue_qna()
