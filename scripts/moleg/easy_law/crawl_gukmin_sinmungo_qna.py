import re
import time
from collections import Counter

from bs4 import BeautifulSoup

from common import (
    CATEGORY_NAME_BY_SEQ,
    CATEGORY_SEQS,
    EASYLAW_BASE_URL,
    PROJECT_ROOT,
    build_bool_literal,
    build_session,
    canonicalize_url,
    clean_label,
    clean_text,
    configure_logging,
    extract_query_value,
    fetch_html,
    log_info,
    make_sha1,
    parse_category_page,
    parse_leaf_nodes,
    write_csv,
)


# 국민신문고는 같은 책자형 노드 안에서도 별도 탭(menuType=qna)으로 제공되므로
# 기존 백문백답과 분리된 source 파일명으로 저장한다.
QNA_URL_TEMPLATE = (
    EASYLAW_BASE_URL
    + "/CSP/CnpClsMain.laf?popMenu=ov&csmSeq={csm_seq}&ccfNo={ccf_no}"
    + "&cciNo={cci_no}&cnpClsNo={cnp_cls_no}&menuType=qna"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "moleg" / "easy_law"
BOOKS_OUTPUT_CSV = OUTPUT_DIR / "gukmin_sinmungo_books.csv"
QNA_OUTPUT_CSV = OUTPUT_DIR / "gukmin_sinmungo_qna.csv"

MAX_BOOKS = None
MAX_LEAF_PAGES_PER_BOOK = None

BOOKS_COLUMNS = [
    "category_seq",
    "category_name",
    "csm_seq",
    "book_title",
    "book_url",
    "leaf_node_count",
    "qna_page_count",
    "qna_item_count",
    "has_qna_rows",
    "qna_status",
    "qna_status_note",
    "crawled_at",
]
QNA_COLUMNS = [
    "qa_id",
    "category_seq",
    "category_name",
    "csm_seq",
    "book_title",
    "ccf_no",
    "cci_no",
    "cnp_cls_no",
    "qna_page_url",
    "page_title",
    "qa_order_in_page",
    "anchor_name",
    "question_title",
    "question_raw",
    "answer_raw",
    "answer_raw_with_markers",
    "answer_html_raw",
    "answer_tables_raw",
    "has_table",
    "content_classification",
    "government_agency",
    "department_raw",
    "department_phone",
    "crawled_at",
]


def build_qna_url(csm_seq, ccf_no, cci_no, cnp_cls_no):
    # 국민신문고는 상세 노드에 qna 탭을 붙여 접근하므로
    # leaf node의 공통 파라미터만 유지한 채 menuType만 바꾼다.
    return canonicalize_url(
        QNA_URL_TEMPLATE.format(
            csm_seq=csm_seq,
            ccf_no=ccf_no,
            cci_no=cci_no,
            cnp_cls_no=cnp_cls_no,
        ),
        EASYLAW_BASE_URL,
    )


def build_html_table_text(table):
    # 답변 안에 표가 들어가도 위치 복원 단서를 남길 수 있게
    # 셀 경계를 유지한 문자열로 별도 보존한다.
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


def extract_answer_variants(answer_tag):
    # 국민신문고 답변도 p/br/table이 섞일 수 있으므로
    # 본문, 위치 마커 포함 본문, 원문 HTML, 표 텍스트를 함께 남긴다.
    answer_html_raw = str(answer_tag) if answer_tag else ""
    if not answer_html_raw:
        return "", "", "", "", False

    table_blocks = []

    marker_soup = BeautifulSoup(answer_html_raw, "html.parser")
    marker_root = marker_soup.find("li")
    if marker_root is None:
        return "", "", answer_html_raw, "", False

    for table_index, table in enumerate(marker_root.select("table"), start=1):
        table_text = build_html_table_text(table)
        marker_text = f"[table {table_index}]"
        if table_text:
            table_blocks.append(f"{marker_text}\n{table_text}")
        table.replace_with(f"\n{marker_text}\n")

    answer_text_with_markers = clean_text(marker_root.get_text("\n", strip=True))

    plain_soup = BeautifulSoup(answer_html_raw, "html.parser")
    plain_root = plain_soup.find("li")
    if plain_root is None:
        return "", answer_text_with_markers, answer_html_raw, "\n\n".join(table_blocks), bool(
            table_blocks
        )

    for table in plain_root.select("table"):
        table.decompose()

    answer_text = clean_text(plain_root.get_text("\n", strip=True))
    answer_tables_raw = "\n\n".join(table_blocks)
    return (
        answer_text,
        answer_text_with_markers,
        answer_html_raw,
        answer_tables_raw,
        bool(table_blocks),
    )


def parse_department_phone(text):
    # 담당부서 라벨에는 전화번호가 괄호 안에 붙어 나오므로
    # 후속 정제에서 기관/부서/전화번호를 분리할 수 있게 전화번호만 따로 뽑는다.
    match = re.search(r"☏\s*([0-9-]+)", text or "")
    return match.group(1).strip() if match else ""


def parse_ci_metadata(ci_tag):
    # 기관 메타는 ci 블록 내부의 li들에 라벨-값 형태로 들어가므로
    # 최소한 콘텐츠 분류, 기관명, 담당부서를 분리해 둔다.
    metadata = {
        "content_classification": "",
        "government_agency": "",
        "department_raw": "",
        "department_phone": "",
    }
    if ci_tag is None:
        return metadata

    for item in ci_tag.select("ul.ci_list > li"):
        raw_text = clean_text(item.get_text(" ", strip=True))
        if not raw_text:
            continue

        if ":" in raw_text:
            label, value = raw_text.split(":", 1)
            label = clean_label(label)
            value = clean_text(value)
        else:
            label = ""
            value = raw_text

        if "콘텐츠 분류" in label:
            metadata["content_classification"] = value
        elif "정부기관" in label:
            metadata["government_agency"] = value
        elif "담당부서" in label:
            metadata["department_raw"] = value
            metadata["department_phone"] = parse_department_phone(value)

    return metadata


def build_qna_page_status(soup, qna_page_final_url, page_title, item_count, qa_row_count):
    # 국민신문고 탭은 epp_list 항목이 실제 수집 단위라서
    # 항목 수와 최종 페이지 상태를 기준으로 탭 존재 여부를 요약한다.
    if qa_row_count > 0:
        return "has_qna"
    if item_count > 0:
        return "qna_tab_empty_or_unparsed"
    if extract_query_value(qna_page_final_url, "menuType") != "qna" or "본문" in page_title:
        return "no_qna_tab"
    if "국민신문고" in page_title:
        return "qna_tab_empty_or_unparsed"
    return "no_qna_detected"


def build_book_qna_status(leaf_node_count, qna_page_count, qna_item_count, page_status_counter):
    # 국민신문고도 book 인벤토리는 raw로 남기고
    # 실제 탭 상태만 별도 컬럼으로 요약한다.
    if leaf_node_count == 0:
        return "no_leaf_nodes"
    if qna_item_count > 0:
        return "has_qna_all_leaf_nodes" if qna_page_count == leaf_node_count else "has_qna_partial"

    nonempty_statuses = [
        status
        for status in ("no_qna_tab", "qna_tab_empty_or_unparsed", "no_qna_detected")
        if page_status_counter[status] > 0
    ]
    if len(nonempty_statuses) == 1:
        return {
            "no_qna_tab": "no_qna_tab_all_leaf_nodes",
            "qna_tab_empty_or_unparsed": "qna_tab_empty_or_unparsed",
            "no_qna_detected": "no_qna_detected",
        }[nonempty_statuses[0]]
    if len(nonempty_statuses) > 1:
        return "no_qna_mixed_status"
    return "no_qna_detected"


def build_book_qna_status_note(leaf_node_count, qna_page_count, qna_item_count, page_status_counter):
    # book 상태 메모는 백문백답과 동일한 형식으로 남겨야
    # source별 book 상태 CSV를 나중에 비교하기 쉽다.
    parts = [
        f"leaf_nodes={leaf_node_count}",
        f"qna_pages={qna_page_count}",
        f"qna_items={qna_item_count}",
    ]
    for status in ("has_qna", "no_qna_tab", "qna_tab_empty_or_unparsed", "no_qna_detected"):
        count = page_status_counter[status]
        if count > 0:
            parts.append(f"{status}={count}")
    return "; ".join(parts)


def parse_qna_page(
    html,
    qna_page_url,
    qna_page_final_url,
    category_seq,
    category_name,
    csm_seq,
    book_title,
    ccf_no,
    cci_no,
    cnp_cls_no,
):
    # 국민신문고는 한 페이지 안에 여러 상담 항목이 들어가는 구조라
    # epp_list의 각 li를 독립 QA row로 만든다.
    soup = BeautifulSoup(html, "html.parser")
    page_title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    crawled_at = time.strftime("%Y-%m-%d %H:%M:%S")

    qa_rows = []
    items = soup.select("ul.epp_list > li")
    for block_index, item in enumerate(items, start=1):
        title_anchor = item.select_one("p.part a[href]")
        question_tag = item.select_one("li.q")
        answer_tag = item.select_one("li.a")
        ci_tag = item.select_one("li.ci")
        if not (title_anchor and question_tag and answer_tag):
            continue

        (
            answer_text,
            answer_text_with_markers,
            answer_html_raw,
            answer_tables_raw,
            has_table,
        ) = extract_answer_variants(answer_tag)
        metadata = parse_ci_metadata(ci_tag)
        anchor_name_tag = item.select_one("div[id^='smg_q'] a[name]")
        qa_id = make_sha1(f"{qna_page_url}#{block_index}")[:16]

        qa_rows.append(
            {
                "qa_id": qa_id,
                "category_seq": str(category_seq),
                "category_name": category_name,
                "csm_seq": str(csm_seq),
                "book_title": book_title,
                "ccf_no": str(ccf_no),
                "cci_no": str(cci_no),
                "cnp_cls_no": str(cnp_cls_no),
                "qna_page_url": qna_page_url,
                "page_title": page_title,
                "qa_order_in_page": str(block_index),
                "anchor_name": clean_text(anchor_name_tag.get("name", "")) if anchor_name_tag else "",
                "question_title": clean_text(title_anchor.get_text(" ", strip=True)),
                "question_raw": clean_text(question_tag.get_text("\n", strip=True)),
                "answer_raw": answer_text,
                "answer_raw_with_markers": answer_text_with_markers,
                "answer_html_raw": answer_html_raw,
                "answer_tables_raw": answer_tables_raw,
                "has_table": build_bool_literal(has_table),
                "content_classification": metadata["content_classification"],
                "government_agency": metadata["government_agency"],
                "department_raw": metadata["department_raw"],
                "department_phone": metadata["department_phone"],
                "crawled_at": crawled_at,
            }
        )

    page_status = build_qna_page_status(
        soup=soup,
        qna_page_final_url=qna_page_final_url,
        page_title=page_title,
        item_count=len(items),
        qa_row_count=len(qa_rows),
    )
    return qa_rows, page_status


def flush_outputs(book_rows, qa_rows):
    # 긴 순회 도중에도 현재 source 결과를 바로 확인할 수 있게
    # book 인벤토리와 QA 산출물을 함께 checkpoint 저장한다.
    write_csv(BOOKS_OUTPUT_CSV, BOOKS_COLUMNS, book_rows)
    write_csv(QNA_OUTPUT_CSV, QNA_COLUMNS, qa_rows)


def crawl_gukmin_sinmungo_qna():
    configure_logging()
    session = build_session()

    seen_books = set()
    book_rows = []
    qa_rows = []

    crawled_book_count = 0
    for category_seq in CATEGORY_SEQS:
        category_books = parse_category_page(session, category_seq)
        category_name = (
            category_books[0]["category_name"]
            if category_books
            else CATEGORY_NAME_BY_SEQ.get(category_seq, "")
        )
        category_log_parts = [f"category_seq={category_seq}"]
        if category_name:
            category_log_parts.append(f"category_name={category_name}")
        category_log_parts.append(f"book_count={len(category_books)}")
        log_info("category", " ".join(category_log_parts))

        for category_book in category_books:
            csm_seq = category_book["csm_seq"]
            if csm_seq in seen_books:
                continue

            seen_books.add(csm_seq)
            crawled_book_count += 1
            if MAX_BOOKS is not None and crawled_book_count > MAX_BOOKS:
                break

            book_html, book_final_url = fetch_html(
                session,
                category_book["book_url"],
                referer=category_book["category_url"],
            )
            leaf_nodes = parse_leaf_nodes(book_html, csm_seq)
            if MAX_LEAF_PAGES_PER_BOOK is not None:
                leaf_nodes = leaf_nodes[:MAX_LEAF_PAGES_PER_BOOK]

            log_info(
                "book",
                (
                    f"category_name={category_book['category_name']} "
                    f"csm_seq={csm_seq} "
                    f"book_title={category_book['book_title']} "
                    f"leaf_node_count={len(leaf_nodes)}"
                ),
            )

            book_qna_page_count = 0
            book_qna_item_count = 0
            page_status_counter = Counter()
            for leaf_node in leaf_nodes:
                qna_url = build_qna_url(
                    csm_seq=csm_seq,
                    ccf_no=leaf_node["ccf_no"],
                    cci_no=leaf_node["cci_no"],
                    cnp_cls_no=leaf_node["cnp_cls_no"],
                )
                qna_html, qna_final_url = fetch_html(session, qna_url, referer=book_final_url)
                page_qa_rows, page_status = parse_qna_page(
                    html=qna_html,
                    qna_page_url=qna_url,
                    qna_page_final_url=qna_final_url,
                    category_seq=category_book["category_seq"],
                    category_name=category_book["category_name"],
                    csm_seq=csm_seq,
                    book_title=category_book["book_title"],
                    ccf_no=leaf_node["ccf_no"],
                    cci_no=leaf_node["cci_no"],
                    cnp_cls_no=leaf_node["cnp_cls_no"],
                )
                page_status_counter[page_status] += 1

                if page_qa_rows:
                    book_qna_page_count += 1
                    book_qna_item_count += len(page_qa_rows)
                    qa_rows.extend(page_qa_rows)

            book_rows.append(
                {
                    "category_seq": category_book["category_seq"],
                    "category_name": category_book["category_name"],
                    "csm_seq": csm_seq,
                    "book_title": category_book["book_title"],
                    "book_url": category_book["book_url"],
                    "leaf_node_count": str(len(leaf_nodes)),
                    "qna_page_count": str(book_qna_page_count),
                    "qna_item_count": str(book_qna_item_count),
                    "has_qna_rows": build_bool_literal(book_qna_item_count > 0),
                    "qna_status": build_book_qna_status(
                        leaf_node_count=len(leaf_nodes),
                        qna_page_count=book_qna_page_count,
                        qna_item_count=book_qna_item_count,
                        page_status_counter=page_status_counter,
                    ),
                    "qna_status_note": build_book_qna_status_note(
                        leaf_node_count=len(leaf_nodes),
                        qna_page_count=book_qna_page_count,
                        qna_item_count=book_qna_item_count,
                        page_status_counter=page_status_counter,
                    ),
                    "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        flush_outputs(book_rows, qa_rows)
        log_info(
            "checkpoint",
            f"category={category_seq} saved books={len(book_rows)} qa_rows={len(qa_rows)}",
        )

        if MAX_BOOKS is not None and crawled_book_count >= MAX_BOOKS:
            break

    flush_outputs(book_rows, qa_rows)
    log_info("done", f"books={len(book_rows)}")
    log_info("done", f"qa_rows={len(qa_rows)}")


if __name__ == "__main__":
    crawl_gukmin_sinmungo_qna()
