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
    parse_category_page,
    parse_leaf_nodes,
    write_csv,
)


# 법령해석례는 같은 상세 노드에서 menuType=expc 탭으로 접근하므로
# 백문백답/국민신문고와 분리된 사례형 source 파일명을 사용한다.
EXPC_URL_TEMPLATE = (
    EASYLAW_BASE_URL
    + "/CSP/CnpClsMain.laf?popMenu=ov&csmSeq={csm_seq}&ccfNo={ccf_no}"
    + "&cciNo={cci_no}&cnpClsNo={cnp_cls_no}&menuType=expc"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "moleg" / "easy_law"
BOOKS_OUTPUT_CSV = OUTPUT_DIR / "expc_books.csv"
CASES_OUTPUT_CSV = OUTPUT_DIR / "expc_cases.csv"

MAX_BOOKS = None
MAX_LEAF_PAGES_PER_BOOK = None

BOOKS_COLUMNS = [
    "category_seq",
    "category_name",
    "csm_seq",
    "book_title",
    "book_url",
    "leaf_node_count",
    "case_page_count",
    "case_item_count",
    "has_case_rows",
    "case_status",
    "case_status_note",
    "crawled_at",
]
CASE_COLUMNS = [
    "case_id",
    "category_seq",
    "category_name",
    "csm_seq",
    "book_title",
    "ccf_no",
    "cci_no",
    "cnp_cls_no",
    "case_page_url",
    "page_title",
    "case_order_in_page",
    "case_anchor_name",
    "case_caption",
    "case_name",
    "question_raw",
    "answer_raw",
    "interpreting_agency_source_raw",
    "crawled_at",
]


def build_expc_url(csm_seq, ccf_no, cci_no, cnp_cls_no):
    # 법령해석례도 책자형 공통 파라미터를 그대로 쓰고
    # 탭 구분자만 expc로 바꿔 접근한다.
    return canonicalize_url(
        EXPC_URL_TEMPLATE.format(
            csm_seq=csm_seq,
            ccf_no=ccf_no,
            cci_no=cci_no,
            cnp_cls_no=cnp_cls_no,
        ),
        EASYLAW_BASE_URL,
    )


def parse_case_table(table):
    # 법령해석례는 표 자체가 사례의 구조이므로
    # 안건명, 질의, 회답, 해석기관 및 출처를 라벨별로 분리 저장한다.
    case_data = {
        "case_anchor_name": "",
        "case_caption": clean_text(table.find("caption").get_text(" ", strip=True))
        if table.find("caption")
        else "",
        "case_name": "",
        "question_raw": "",
        "answer_raw": "",
        "interpreting_agency_source_raw": "",
    }

    anchor = table.select_one("a[name]")
    if anchor:
        case_data["case_anchor_name"] = clean_text(anchor.get("name", ""))

    for row in table.select("tr"):
        header = row.find("th")
        value = row.find("td")
        if not (header and value):
            continue

        label = clean_label(header.get_text(" ", strip=True))
        value_text = clean_text(value.get_text("\n", strip=True))
        if "안건명" in label:
            case_data["case_name"] = value_text
        elif "질의" in label:
            case_data["question_raw"] = value_text
        elif "회답" in label:
            case_data["answer_raw"] = value_text
        elif "해석기관" in label or "출처" in label:
            case_data["interpreting_agency_source_raw"] = value_text

    return case_data


def build_case_page_status(expc_page_final_url, page_title, table_count, case_row_count):
    # 법령해석례는 사례 테이블이 실제 수집 단위이므로
    # 표 개수와 최종 탭 상태를 기준으로 탭 존재 여부를 판정한다.
    if case_row_count > 0:
        return "has_cases"
    if table_count > 0:
        return "case_tab_empty_or_unparsed"
    if extract_query_value(expc_page_final_url, "menuType") != "expc" or "본문" in page_title:
        return "no_case_tab"
    if "법령해석례" in page_title:
        return "case_tab_empty_or_unparsed"
    return "no_case_detected"


def build_book_case_status(leaf_node_count, case_page_count, case_item_count, page_status_counter):
    # 사례형 탭도 book 인벤토리는 그대로 남기고
    # 사례 row 존재 여부만 상태 컬럼으로 구분한다.
    if leaf_node_count == 0:
        return "no_leaf_nodes"
    if case_item_count > 0:
        return "has_cases_all_leaf_nodes" if case_page_count == leaf_node_count else "has_cases_partial"

    nonempty_statuses = [
        status
        for status in ("no_case_tab", "case_tab_empty_or_unparsed", "no_case_detected")
        if page_status_counter[status] > 0
    ]
    if len(nonempty_statuses) == 1:
        return {
            "no_case_tab": "no_case_tab_all_leaf_nodes",
            "case_tab_empty_or_unparsed": "case_tab_empty_or_unparsed",
            "no_case_detected": "no_case_detected",
        }[nonempty_statuses[0]]
    if len(nonempty_statuses) > 1:
        return "no_case_mixed_status"
    return "no_case_detected"


def build_book_case_status_note(leaf_node_count, case_page_count, case_item_count, page_status_counter):
    # book 메모 형식은 다른 탭과 맞춰 두어야
    # source별 status CSV를 나란히 비교하기 쉽다.
    parts = [
        f"leaf_nodes={leaf_node_count}",
        f"case_pages={case_page_count}",
        f"case_items={case_item_count}",
    ]
    for status in ("has_cases", "no_case_tab", "case_tab_empty_or_unparsed", "no_case_detected"):
        count = page_status_counter[status]
        if count > 0:
            parts.append(f"{status}={count}")
    return "; ".join(parts)


def parse_expc_page(
    html,
    expc_page_url,
    expc_page_final_url,
    category_seq,
    category_name,
    csm_seq,
    book_title,
    ccf_no,
    cci_no,
    cnp_cls_no,
):
    # 법령해석례는 한 페이지에 사례 테이블이 여러 개 있을 수도 있으므로
    # table.prcd.w760를 모두 순회해 사례 row를 만든다.
    soup = BeautifulSoup(html, "html.parser")
    page_title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    crawled_at = time.strftime("%Y-%m-%d %H:%M:%S")

    case_rows = []
    tables = soup.select("table.prcd.w760")
    for table_index, table in enumerate(tables, start=1):
        case_data = parse_case_table(table)
        if not any(
            [
                case_data["case_name"],
                case_data["question_raw"],
                case_data["answer_raw"],
                case_data["interpreting_agency_source_raw"],
            ]
        ):
            continue

        case_rows.append(
            {
                "case_id": extract_query_value(expc_page_url, "csmSeq")
                + "_"
                + extract_query_value(expc_page_url, "ccfNo")
                + "_"
                + extract_query_value(expc_page_url, "cciNo")
                + "_"
                + extract_query_value(expc_page_url, "cnpClsNo")
                + "_"
                + str(table_index),
                "category_seq": str(category_seq),
                "category_name": category_name,
                "csm_seq": str(csm_seq),
                "book_title": book_title,
                "ccf_no": str(ccf_no),
                "cci_no": str(cci_no),
                "cnp_cls_no": str(cnp_cls_no),
                "case_page_url": expc_page_url,
                "page_title": page_title,
                "case_order_in_page": str(table_index),
                "case_anchor_name": case_data["case_anchor_name"],
                "case_caption": case_data["case_caption"],
                "case_name": case_data["case_name"],
                "question_raw": case_data["question_raw"],
                "answer_raw": case_data["answer_raw"],
                "interpreting_agency_source_raw": case_data["interpreting_agency_source_raw"],
                "crawled_at": crawled_at,
            }
        )

    page_status = build_case_page_status(
        expc_page_final_url=expc_page_final_url,
        page_title=page_title,
        table_count=len(tables),
        case_row_count=len(case_rows),
    )
    return case_rows, page_status


def flush_outputs(book_rows, case_rows):
    # 사례형 탭도 checkpoint 저장 방식을 동일하게 유지해야
    # 긴 순회 도중 어느 단계까지 저장됐는지 바로 알 수 있다.
    write_csv(BOOKS_OUTPUT_CSV, BOOKS_COLUMNS, book_rows)
    write_csv(CASES_OUTPUT_CSV, CASE_COLUMNS, case_rows)


def crawl_expc_cases():
    configure_logging()
    session = build_session()

    seen_books = set()
    book_rows = []
    case_rows = []

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

            book_case_page_count = 0
            book_case_item_count = 0
            page_status_counter = Counter()
            for leaf_node in leaf_nodes:
                expc_url = build_expc_url(
                    csm_seq=csm_seq,
                    ccf_no=leaf_node["ccf_no"],
                    cci_no=leaf_node["cci_no"],
                    cnp_cls_no=leaf_node["cnp_cls_no"],
                )
                expc_html, expc_final_url = fetch_html(session, expc_url, referer=book_final_url)
                page_case_rows, page_status = parse_expc_page(
                    html=expc_html,
                    expc_page_url=expc_url,
                    expc_page_final_url=expc_final_url,
                    category_seq=category_book["category_seq"],
                    category_name=category_book["category_name"],
                    csm_seq=csm_seq,
                    book_title=category_book["book_title"],
                    ccf_no=leaf_node["ccf_no"],
                    cci_no=leaf_node["cci_no"],
                    cnp_cls_no=leaf_node["cnp_cls_no"],
                )
                page_status_counter[page_status] += 1

                if page_case_rows:
                    book_case_page_count += 1
                    book_case_item_count += len(page_case_rows)
                    case_rows.extend(page_case_rows)

            book_rows.append(
                {
                    "category_seq": category_book["category_seq"],
                    "category_name": category_book["category_name"],
                    "csm_seq": csm_seq,
                    "book_title": category_book["book_title"],
                    "book_url": category_book["book_url"],
                    "leaf_node_count": str(len(leaf_nodes)),
                    "case_page_count": str(book_case_page_count),
                    "case_item_count": str(book_case_item_count),
                    "has_case_rows": build_bool_literal(book_case_item_count > 0),
                    "case_status": build_book_case_status(
                        leaf_node_count=len(leaf_nodes),
                        case_page_count=book_case_page_count,
                        case_item_count=book_case_item_count,
                        page_status_counter=page_status_counter,
                    ),
                    "case_status_note": build_book_case_status_note(
                        leaf_node_count=len(leaf_nodes),
                        case_page_count=book_case_page_count,
                        case_item_count=book_case_item_count,
                        page_status_counter=page_status_counter,
                    ),
                    "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        flush_outputs(book_rows, case_rows)
        log_info(
            "checkpoint",
            f"category={category_seq} saved books={len(book_rows)} case_rows={len(case_rows)}",
        )

        if MAX_BOOKS is not None and crawled_book_count >= MAX_BOOKS:
            break

    flush_outputs(book_rows, case_rows)
    log_info("done", f"books={len(book_rows)}")
    log_info("done", f"case_rows={len(case_rows)}")


if __name__ == "__main__":
    crawl_expc_cases()
