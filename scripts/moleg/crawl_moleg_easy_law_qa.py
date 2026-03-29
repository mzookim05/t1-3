import csv
import hashlib
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# 이 스크립트는 인자를 받지 않고 바로 실행하는 형태로 고정한다.
# 현재 목표는 책자형 생활법령 -> 백문백답 -> 관련법령 링크를 따라가며
# 원문 사이트에서 바로 받은 1차 수집본을 확보하는 것이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

EASYLAW_BASE_URL = "https://www.easylaw.go.kr"
LAW_BASE_URL = "https://www.law.go.kr"
CATEGORY_URL_TEMPLATE = (
    EASYLAW_BASE_URL + "/CSP/CsmSortRetrieveLst.laf?sortType=cate&csmAstSeq={category_seq}"
)
BOOK_URL_TEMPLATE = EASYLAW_BASE_URL + "/CSP/CsmMain.laf?csmSeq={csm_seq}"
QNA_URL_TEMPLATE = (
    EASYLAW_BASE_URL
    + "/CSP/CnpClsMain.laf?popMenu=ov&csmSeq={csm_seq}&ccfNo={ccf_no}"
    + "&cciNo={cci_no}&cnpClsNo={cnp_cls_no}&menuType=onhunqna"
)
LAW_URL_PREFIXES = ("http://www.law.go.kr", "https://www.law.go.kr", "/LSW/")

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "moleg"
BOOKS_OUTPUT_CSV = OUTPUT_DIR / "moleg_easy_law_books.csv"
QA_OUTPUT_CSV = OUTPUT_DIR / "moleg_easy_law_qa.csv"
QA_LAW_LINKS_OUTPUT_CSV = OUTPUT_DIR / "moleg_easy_law_qa_law_links.csv"
LAW_PAGES_OUTPUT_CSV = OUTPUT_DIR / "moleg_easy_law_law_pages.csv"

REQUEST_TIMEOUT = 30
REQUEST_RETRY_TOTAL = 5
REQUEST_BACKOFF_FACTOR = 1.0
REQUEST_SLEEP_SECONDS = 0.15

# 운영 범위는 코드 내부 상수로만 제어한다.
# 디버깅할 때는 CATEGORY_SEQS를 (1,)처럼 줄여서 시작하면 된다.
CATEGORY_SEQS = tuple(range(1, 19))
CATEGORY_NAME_BY_SEQ = {
    1: "가정법률",
    2: "아동·청소년/교육",
    3: "부동산/임대",
    4: "금융/금전",
    5: "사업",
    6: "창업",
    7: "무역/출입국",
    8: "소비자",
    9: "문화/여가생활",
    10: "민형사/소송",
    11: "교통/운전",
    12: "근로/노동",
    13: "복지",
    14: "국방/보훈",
    15: "정보통신/기술",
    16: "환경/에너지",
    17: "사회안전/범죄",
    18: "국가 및 지자체",
}
MAX_BOOKS = None
MAX_LEAF_PAGES_PER_BOOK = None
# 국가법령정보센터 본문은 requests 1회 요청만으로는 빈 본문이 내려오는 경우가 있어
# 기본값은 꺼 두고, 관련법령 링크를 먼저 안정적으로 확보하는 쪽을 우선한다.
FETCH_LAW_PAGES = False

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
QA_COLUMNS = [
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
    "title_section",
    "title_book",
    "title_subject",
    "question",
    "answer_raw",
    "answer_raw_with_markers",
    "answer_html_raw",
    "answer_tables_raw",
    "has_table",
    "view_count",
    "recommend_count",
    "related_life_areas_raw",
    "related_life_area_urls_raw",
    "related_life_area_count",
    "related_laws_raw",
    "related_law_count",
    "crawled_at",
]
QA_LAW_LINK_COLUMNS = [
    "qa_id",
    "law_ref_order",
    "law_ref_raw",
    "law_anchor_text",
    "law_suffix_text",
    "law_name",
    "article_label",
    "law_url",
    "law_url_canonical",
    "ls_name_param",
    "jo_no_param",
    "doc_type_param",
    "crawled_at",
]
LAW_PAGES_COLUMNS = [
    "law_page_id",
    "law_url",
    "law_url_canonical",
    "page_title",
    "header_text",
    "body_text_raw",
    "ls_name_param",
    "jo_no_param",
    "doc_type_param",
    "crawled_at",
]

LOGGER = logging.getLogger("moleg_easy_law_crawler")


def configure_logging():
    # 다른 수집 스크립트와 같은 형식으로 남겨야 진행 상황을 한 눈에 비교하기 쉽다.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_info(tag, message):
    LOGGER.info("[%s] %s", tag, message)


def log_warning(tag, message):
    LOGGER.warning("[%s] %s", tag, message)


def build_temp_output_path(output_path):
    # 최종 CSV를 바로 덮어쓰지 않고 임시 파일을 거쳐 교체해야
    # 중간 실패가 나도 기존 결과를 보존할 수 있다.
    return output_path.with_suffix(output_path.suffix + ".tmp")


def build_session():
    # 생활법령 사이트는 서버 렌더링 HTML이므로 브라우저 자동화까지는 필요 없다.
    # 대신 재시도와 연결 재사용을 켜서 긴 순회에서도 비교적 안정적으로 수집한다.
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
        backoff_factor=REQUEST_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Referer": EASYLAW_BASE_URL,
        }
    )
    return session


def fetch_html(session, url, referer=None):
    # 카테고리 -> 책자 -> 백문백답으로 따라 들어가는 구조라 referer를 넘겨 두는 편이 안전하다.
    headers = {}
    if referer:
        headers["Referer"] = referer

    response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.text, response.url


def clean_text(text):
    # 저장 단계에서는 원문을 최대한 보존하되, CSV를 망가뜨리는 과한 공백만 정리한다.
    if text is None:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        normalized = re.sub(r"[ \t]+", " ", line).strip()
        if normalized:
            cleaned_lines.append(normalized)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_label(text):
    return clean_text(text).rstrip(":").strip()


def parse_number(text):
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else 0


def build_bool_literal(value):
    # 다른 수집 스크립트와 마찬가지로 bool 컬럼은 소문자 literal로 고정해 두면
    # CSV를 후속 정제 파이프라인에 바로 넘길 때 해석이 단순해진다.
    return "true" if value else "false"


def has_inline_qna_candidate(soup):
    # 100문 100답 탭이 없어도 본문 안에 Q./A. 박스가 들어간 페이지가 있어
    # strong 태그의 짧은 라벨 패턴으로 inline Q/A 후보만 보수적으로 감지한다.
    has_question_label = False
    has_answer_label = False

    for strong_tag in soup.select("strong"):
        label = clean_text(strong_tag.get_text(" ", strip=True))
        if re.fullmatch(r"Q\s*[.．:：]?", label, flags=re.IGNORECASE):
            has_question_label = True
        elif re.fullmatch(r"A\s*[.．:：]?", label, flags=re.IGNORECASE):
            has_answer_label = True

        if has_question_label and has_answer_label:
            return True

    return False


def build_qna_page_status(soup, qna_page_final_url, page_title, question_block_count, qa_row_count):
    # book 단위 상태를 만들기 전에 leaf 페이지 하나하나가 어떤 종류였는지 기록해 두면
    # "진짜 무문답인지 / 탭이 없는지 / 본문형 Q/A 후보인지"를 사실 기반으로 요약할 수 있다.
    if qa_row_count > 0:
        return "has_qna"
    if question_block_count > 0:
        return "qna_tab_empty_or_unparsed"
    if has_inline_qna_candidate(soup):
        return "inline_qna_candidate"
    if extract_query_value(qna_page_final_url, "menuType") != "onhunqna" or "본문" in page_title:
        return "no_qna_tab"
    if "100문 100답" in page_title:
        return "qna_tab_empty_or_unparsed"
    return "no_qna_detected"


def build_book_qna_status(leaf_node_count, qna_page_count, qna_item_count, page_status_counter):
    # raw 단계에서는 book을 버리지 않고 남기되, 지금 파서가 본 사실을 상태값으로 요약해 두어야
    # 나중에 0건 book과 inline Q/A 후보를 다시 추적하기 쉬워진다.
    if leaf_node_count == 0:
        return "no_leaf_nodes"
    if qna_item_count > 0:
        if qna_page_count == leaf_node_count:
            return "has_qna_all_leaf_nodes"
        if page_status_counter["inline_qna_candidate"] > 0:
            return "has_qna_partial_with_inline_candidates"
        return "has_qna_partial"
    if page_status_counter["inline_qna_candidate"] > 0:
        return "no_qna_inline_candidate_only"

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
    # 상태값만으로는 왜 그렇게 분류됐는지 바로 안 보일 수 있어
    # leaf 수와 페이지별 상태 집계를 짧은 메모 문자열로 같이 남긴다.
    parts = [
        f"leaf_nodes={leaf_node_count}",
        f"qna_pages={qna_page_count}",
        f"qna_items={qna_item_count}",
    ]
    for status in (
        "has_qna",
        "inline_qna_candidate",
        "no_qna_tab",
        "qna_tab_empty_or_unparsed",
        "no_qna_detected",
    ):
        count = page_status_counter[status]
        if count > 0:
            parts.append(f"{status}={count}")
    return "; ".join(parts)


def make_sha1(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def canonicalize_url(url, default_base):
    # 같은 페이지가 쿼리 순서만 다른 URL로 중복 저장되지 않도록
    # 절대경로화 + 쿼리 정렬까지 같이 처리한다.
    absolute_url = urljoin(default_base, url)
    split_result = urlsplit(absolute_url)
    query_pairs = parse_qs(split_result.query, keep_blank_values=True)
    flattened_pairs = []
    for key in sorted(query_pairs):
        for value in sorted(query_pairs[key]):
            flattened_pairs.append((key, value))
    canonical_query = urlencode(flattened_pairs, doseq=True)
    return urlunsplit(
        (
            split_result.scheme,
            split_result.netloc,
            split_result.path,
            canonical_query,
            "",
        )
    )


def extract_query_value(url, key):
    query = parse_qs(urlsplit(url).query)
    values = query.get(key, [])
    return values[0] if values else ""


def parse_category_name(soup):
    # 카테고리 페이지 상단의 현재 분야명을 먼저 읽고,
    # 실패하면 아래 fallback 사전을 사용한다.
    current_heading = soup.select_one("div.real h5 span")
    if current_heading:
        return clean_text(current_heading.get_text(" ", strip=True))
    return ""


def parse_csm_seq(url):
    return extract_query_value(url, "csmSeq")


def parse_category_page(session, category_seq):
    # 18개 상위 분야 페이지에서 책자형 생활법령 목록만 먼저 모은다.
    # 여기 단계에서는 csmSeq와 책자 제목만 안정적으로 확보하면 충분하다.
    url = CATEGORY_URL_TEMPLATE.format(category_seq=category_seq)
    html, final_url = fetch_html(session, url)
    soup = BeautifulSoup(html, "html.parser")
    category_name = parse_category_name(soup) or CATEGORY_NAME_BY_SEQ.get(category_seq, "")
    rows = []

    for item in soup.select("ul.ganaList li"):
        anchor = item.select_one("a[href]")
        if not anchor:
            continue

        book_url = canonicalize_url(anchor["href"], EASYLAW_BASE_URL)
        csm_seq = parse_csm_seq(book_url)
        if not csm_seq:
            continue

        book_title = clean_text(
            (
                item.select_one("h6").get_text(" ", strip=True)
                if item.select_one("h6")
                else anchor.get_text(" ", strip=True)
            )
        )

        rows.append(
            {
                "category_seq": str(category_seq),
                "category_name": category_name,
                "csm_seq": csm_seq,
                "book_title": book_title,
                "book_url": book_url,
                "category_url": final_url,
            }
        )

    return rows


def parse_leaf_nodes(book_html, csm_seq):
    # 책자 메인 페이지의 좌측 트리에서 실제 leaf만 골라야 같은 책의 하위 문답 페이지를 순회할 수 있다.
    # menuType이 붙은 링크는 본문/백문백답/관련법령 탭 링크이므로 leaf 후보에서 제외한다.
    soup = BeautifulSoup(book_html, "html.parser")
    seen = {}

    for node in soup.select("li[id] a[href]"):
        href = node["href"]
        absolute_url = canonicalize_url(href, EASYLAW_BASE_URL)
        if "/CSP/CnpClsMain.laf" not in absolute_url:
            continue

        current_csm_seq = extract_query_value(absolute_url, "csmSeq")
        if current_csm_seq != str(csm_seq):
            continue

        if extract_query_value(absolute_url, "menuType"):
            continue

        ccf_no = extract_query_value(absolute_url, "ccfNo")
        cci_no = extract_query_value(absolute_url, "cciNo")
        cnp_cls_no = extract_query_value(absolute_url, "cnpClsNo")
        if not (ccf_no and cci_no and cnp_cls_no):
            continue

        key = (ccf_no, cci_no, cnp_cls_no)
        if key in seen:
            continue

        seen[key] = {
            "ccf_no": ccf_no,
            "cci_no": cci_no,
            "cnp_cls_no": cnp_cls_no,
            "leaf_title": clean_text(node.get_text(" ", strip=True)),
            "content_url": absolute_url,
        }

    return list(seen.values())


def build_qna_url(csm_seq, ccf_no, cci_no, cnp_cls_no):
    # 실제 수집 대상은 책자 본문이 아니라 같은 노드의 menuType=onhunqna 탭이다.
    return canonicalize_url(
        QNA_URL_TEMPLATE.format(
            csm_seq=csm_seq,
            ccf_no=ccf_no,
            cci_no=cci_no,
            cnp_cls_no=cnp_cls_no,
        ),
        EASYLAW_BASE_URL,
    )


def find_immediate_metadata_table(question_block):
    # 저장본과 실서비스 응답을 보면 관련생활분야/관련법령 표는
    # 해당 ul.question 바로 뒤 sibling table로 붙는 경우가 가장 안정적이었다.
    sibling = question_block.next_sibling
    while sibling is not None:
        if getattr(sibling, "name", None) == "table":
            class_names = sibling.get("class", [])
            if "normal" in class_names and "st5" in class_names:
                return sibling
            return None

        if getattr(sibling, "name", None):
            return None

        sibling = sibling.next_sibling

    return None


def split_law_reference(anchor_text, suffix_text):
    # 관련법령 셀은 앵커에 법령명+조문이 같이 들어가고,
    # 뒤쪽 텍스트에 제1항 같은 꼬리정보가 붙는 경우가 있어서 둘을 분리해 둔다.
    anchor_text = clean_text(anchor_text)
    suffix_text = clean_text(suffix_text)
    law_name = anchor_text
    article_from_anchor = ""

    if anchor_text.startswith("「") and "」" in anchor_text:
        closing_index = anchor_text.find("」")
        law_name = anchor_text[1:closing_index]
        article_from_anchor = anchor_text[closing_index + 1 :].strip()
    elif anchor_text.startswith("『") and "』" in anchor_text:
        closing_index = anchor_text.find("』")
        law_name = anchor_text[1:closing_index]
        article_from_anchor = anchor_text[closing_index + 1 :].strip()
    elif " 제" in anchor_text:
        split_name, split_article = anchor_text.split(" 제", 1)
        law_name = split_name.strip()
        article_from_anchor = "제" + split_article.strip()

    article_label = clean_text(" ".join(part for part in [article_from_anchor, suffix_text] if part))
    return law_name, article_label


def parse_related_law_rows(cell):
    # 관련법령은 나중에 조문 본문을 별도로 붙일 수 있도록
    # 한 줄 텍스트뿐 아니라 법령명, 조문표기, 원본 URL 파라미터까지 함께 보존한다.
    rows = []
    blocks = cell.find_all(["p", "li"], recursive=False)
    if not blocks:
        blocks = [cell]

    for block in blocks:
        raw_text = clean_text(block.get_text(" ", strip=True))
        if not raw_text:
            continue

        anchor = block.find("a", href=True)
        if anchor:
            law_url = canonicalize_url(anchor["href"], LAW_BASE_URL)
            anchor_text = clean_text(anchor.get_text(" ", strip=True))
            if raw_text.startswith(anchor_text):
                suffix_text = raw_text[len(anchor_text) :].strip()
            else:
                suffix_text = raw_text.replace(anchor_text, "", 1).strip()
        else:
            law_url = ""
            anchor_text = ""
            suffix_text = raw_text

        law_name, article_label = split_law_reference(anchor_text, suffix_text)

        rows.append(
            {
                "law_ref_raw": raw_text,
                "law_anchor_text": anchor_text,
                "law_suffix_text": suffix_text,
                "law_name": law_name,
                "article_label": article_label,
                "law_url": law_url,
                "law_url_canonical": law_url,
                "ls_name_param": extract_query_value(law_url, "lsNm"),
                "jo_no_param": extract_query_value(law_url, "joNo"),
                "doc_type_param": extract_query_value(law_url, "docType"),
            }
        )

    return rows


def parse_metadata_table(metadata_table):
    # 모든 QA 블록이 같은 메타 테이블을 갖지는 않으므로,
    # 테이블이 없더라도 빈 값으로 정상 row를 만들 수 있게 처리한다.
    related_life_area_texts = []
    related_life_area_urls = []
    related_law_rows = []

    if metadata_table is None:
        return {
            "related_life_areas_raw": "",
            "related_life_area_urls_raw": "",
            "related_life_area_count": 0,
            "related_laws_raw": "",
            "related_law_count": 0,
            "related_law_rows": [],
        }

    for row in metadata_table.select("tr"):
        heading = row.find("th")
        value = row.find("td")
        if not (heading and value):
            continue

        heading_text = clean_label(heading.get_text(" ", strip=True))
        if "관련생활분야" in heading_text:
            anchors = value.select("a[href]")
            if anchors:
                for anchor in anchors:
                    related_life_area_texts.append(clean_text(anchor.get_text(" ", strip=True)))
                    related_life_area_urls.append(canonicalize_url(anchor["href"], EASYLAW_BASE_URL))
            else:
                raw_text = clean_text(value.get_text("\n", strip=True))
                if raw_text:
                    related_life_area_texts.append(raw_text)
        elif "관련법령" in heading_text:
            related_law_rows.extend(parse_related_law_rows(value))

    return {
        "related_life_areas_raw": "\n".join(related_life_area_texts),
        "related_life_area_urls_raw": "\n".join(related_life_area_urls),
        "related_life_area_count": len(related_life_area_texts),
        "related_laws_raw": "\n".join(row["law_ref_raw"] for row in related_law_rows),
        "related_law_count": len(related_law_rows),
        "related_law_rows": related_law_rows,
    }


def parse_title_block(title_block):
    # title li에는 분야, 책자명, 소제목과 조회/추천수가 섞여 있으므로
    # 바로 학습 데이터에 쓸 수 있게 최소 필드로 분리한다.
    direct_spans = title_block.find_all("span", recursive=False) if title_block else []
    span_texts = [clean_label(span.get_text(" ", strip=True)) for span in direct_spans]

    count_spans = title_block.select("p span") if title_block else []
    view_count = parse_number(count_spans[0].get_text(" ", strip=True)) if len(count_spans) >= 1 else 0
    recommend_count = (
        parse_number(count_spans[1].get_text(" ", strip=True)) if len(count_spans) >= 2 else 0
    )

    return {
        "title_section": span_texts[0] if len(span_texts) >= 1 else "",
        "title_book": span_texts[1] if len(span_texts) >= 2 else "",
        "title_subject": span_texts[2] if len(span_texts) >= 3 else "",
        "view_count": view_count,
        "recommend_count": recommend_count,
    }


def build_html_table_text(table):
    # HTML 표는 본문 텍스트에 섞어 버리면 행/열 구조를 잃어버리므로
    # 셀 경계만이라도 남도록 "A | B | C" 형태로 풀어서 별도 컬럼에 저장한다.
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


def build_answer_html_raw(answer_tag):
    # raw 단계에서는 사람이 추후에 구조를 다시 해석할 수 있도록
    # dd 전체 HTML을 함께 남겨 두는 편이 가장 안전하다.
    return str(answer_tag) if answer_tag else ""


def extract_answer_variants(answer_tag):
    # 표를 단순히 떼어 내면 위치 정보가 사라지므로,
    # 1) 표 제외 본문
    # 2) 표 위치 마커가 남아 있는 본문
    # 3) 표 블록 텍스트
    # 4) 원문 HTML
    # 을 함께 저장해 raw 단계의 정보 손실을 최소화한다.
    answer_html_raw = build_answer_html_raw(answer_tag)
    if not answer_html_raw:
        return "", "", "", "", False

    table_blocks = []

    marker_soup = BeautifulSoup(answer_html_raw, "html.parser")
    marker_root = marker_soup.find("dd")
    if marker_root is None:
        return "", "", answer_html_raw, "", False

    # 본문 위치 복원을 위해 table 태그를 지우지 않고 표 순서에 대응하는 마커로 치환한다.
    for table_index, table in enumerate(marker_root.select("table"), start=1):
        table_text = build_html_table_text(table)
        marker_text = f"[table {table_index}]"
        if table_text:
            table_blocks.append(f"{marker_text}\n{table_text}")
        table.replace_with(f"\n{marker_text}\n")

    answer_text_with_markers = clean_text(marker_root.get_text("\n", strip=True))

    plain_soup = BeautifulSoup(answer_html_raw, "html.parser")
    plain_root = plain_soup.find("dd")
    if plain_root is None:
        return "", answer_text_with_markers, answer_html_raw, "\n\n".join(table_blocks), bool(
            table_blocks
        )

    # 서술형 본문만 따로 보고 싶을 때를 위해 표가 제거된 버전도 함께 남긴다.
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
    # 백문백답 한 페이지에 QA가 여러 개 들어 있으므로
    # 페이지 단위가 아니라 ul.question 단위로 row를 만들어야 한다.
    soup = BeautifulSoup(html, "html.parser")
    page_title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    crawled_at = time.strftime("%Y-%m-%d %H:%M:%S")

    qa_rows = []
    law_link_rows = []
    question_blocks = soup.select("ul.question")

    for block_index, question_block in enumerate(question_blocks, start=1):
        question_tag = question_block.select_one("li.qa dt")
        answer_tag = question_block.select_one("li.qa dd")
        if not (question_tag and answer_tag):
            continue

        # 메타 테이블은 각 QA 바로 뒤에 붙어 있는 경우가 있어
        # 질문/답변 블록과 별도로 찾아서 합친다.
        title_fields = parse_title_block(question_block.select_one("li.title"))
        metadata = parse_metadata_table(find_immediate_metadata_table(question_block))
        (
            answer_text,
            answer_text_with_markers,
            answer_html_raw,
            answer_tables_raw,
            has_table,
        ) = extract_answer_variants(answer_tag)

        qa_id = make_sha1(f"{qna_page_url}#{block_index}")[:16]
        qa_row = {
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
            "title_section": title_fields["title_section"],
            "title_book": title_fields["title_book"],
            "title_subject": title_fields["title_subject"],
            "question": clean_text(question_tag.get_text("\n", strip=True)),
            "answer_raw": answer_text,
            "answer_raw_with_markers": answer_text_with_markers,
            "answer_html_raw": answer_html_raw,
            "answer_tables_raw": answer_tables_raw,
            "has_table": build_bool_literal(has_table),
            "view_count": str(title_fields["view_count"]),
            "recommend_count": str(title_fields["recommend_count"]),
            "related_life_areas_raw": metadata["related_life_areas_raw"],
            "related_life_area_urls_raw": metadata["related_life_area_urls_raw"],
            "related_life_area_count": str(metadata["related_life_area_count"]),
            "related_laws_raw": metadata["related_laws_raw"],
            "related_law_count": str(metadata["related_law_count"]),
            "crawled_at": crawled_at,
        }
        qa_rows.append(qa_row)

        for law_ref_order, law_row in enumerate(metadata["related_law_rows"], start=1):
            law_link_rows.append(
                {
                    "qa_id": qa_id,
                    "law_ref_order": str(law_ref_order),
                    "law_ref_raw": law_row["law_ref_raw"],
                    "law_anchor_text": law_row["law_anchor_text"],
                    "law_suffix_text": law_row["law_suffix_text"],
                    "law_name": law_row["law_name"],
                    "article_label": law_row["article_label"],
                    "law_url": law_row["law_url"],
                    "law_url_canonical": law_row["law_url_canonical"],
                    "ls_name_param": law_row["ls_name_param"],
                    "jo_no_param": law_row["jo_no_param"],
                    "doc_type_param": law_row["doc_type_param"],
                    "crawled_at": crawled_at,
                }
            )

    page_status = build_qna_page_status(
        soup=soup,
        qna_page_final_url=qna_page_final_url,
        page_title=page_title,
        question_block_count=len(question_blocks),
        qa_row_count=len(qa_rows),
    )
    return qa_rows, law_link_rows, page_status


def parse_law_page(html, law_url):
    # 국가법령정보센터는 저장본 HTML과 실응답 구조가 조금 달라서
    # 여러 후보 컨테이너를 순서대로 시도해 본문 텍스트를 최대한 건져낸다.
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    header_node = soup.select_one("#leftContent")
    body_node = (
        soup.select_one("#bodyContent")
        or soup.select_one("#bodyContentTOP")
        or soup.select_one(".viewwrap")
        or soup.select_one("#container")
    )

    header_text = clean_text(header_node.get_text(" ", strip=True)) if header_node else ""
    body_text = clean_text(body_node.get_text("\n", strip=True)) if body_node else clean_text(
        soup.get_text("\n", strip=True)
    )

    canonical_url = canonicalize_url(law_url, LAW_BASE_URL)
    return {
        "law_page_id": make_sha1(canonical_url)[:16],
        "law_url": law_url,
        "law_url_canonical": canonical_url,
        "page_title": title,
        "header_text": header_text,
        "body_text_raw": body_text,
        "ls_name_param": extract_query_value(canonical_url, "lsNm"),
        "jo_no_param": extract_query_value(canonical_url, "joNo"),
        "doc_type_param": extract_query_value(canonical_url, "docType"),
        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_csv(output_path, fieldnames, rows):
    # 카테고리 단위 checkpoint 저장을 자주 하기 때문에
    # 이 함수는 언제든 재호출해도 안전한 원자적 쓰기 형태를 유지한다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = build_temp_output_path(output_path)
    with open(temp_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, output_path)


def flush_outputs(book_rows, qa_rows, law_link_rows, law_page_rows):
    # 디버깅 중간에도 결과를 바로 확인할 수 있도록
    # 현재까지 누적된 네 종류 산출물을 한 번에 저장한다.
    write_csv(BOOKS_OUTPUT_CSV, BOOKS_COLUMNS, book_rows)
    write_csv(QA_OUTPUT_CSV, QA_COLUMNS, qa_rows)
    write_csv(QA_LAW_LINKS_OUTPUT_CSV, QA_LAW_LINK_COLUMNS, law_link_rows)
    write_csv(LAW_PAGES_OUTPUT_CSV, LAW_PAGES_COLUMNS, law_page_rows)


def crawl_easy_law():
    configure_logging()
    session = build_session()

    # 지금 단계는 raw 수집이 목적이라 메모리에 누적하면서도
    # 카테고리 하나가 끝날 때마다 checkpoint를 남기는 방식으로 운영한다.
    seen_books = set()
    book_rows = []
    qa_rows = []
    law_link_rows = []
    law_page_rows = []

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

            # 같은 책자가 다른 경로에서 다시 보일 수 있어 csmSeq 기준으로 한 번만 수집한다.
            seen_books.add(csm_seq)
            crawled_book_count += 1
            if MAX_BOOKS is not None and crawled_book_count > MAX_BOOKS:
                break

            book_html, book_final_url = fetch_html(session, category_book["book_url"], referer=category_book["category_url"])
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
                # leaf 하나가 책자 트리의 세부 설명 페이지 하나에 해당하고,
                # 그 설명 페이지의 100문 100답 탭에서 실제 QA를 수집한다.
                qna_url = build_qna_url(
                    csm_seq=csm_seq,
                    ccf_no=leaf_node["ccf_no"],
                    cci_no=leaf_node["cci_no"],
                    cnp_cls_no=leaf_node["cnp_cls_no"],
                )
                qna_html, qna_final_url = fetch_html(session, qna_url, referer=book_final_url)
                page_qa_rows, page_law_link_rows, page_status = parse_qna_page(
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
                    law_link_rows.extend(page_law_link_rows)

            book_qna_status = build_book_qna_status(
                leaf_node_count=len(leaf_nodes),
                qna_page_count=book_qna_page_count,
                qna_item_count=book_qna_item_count,
                page_status_counter=page_status_counter,
            )
            book_qna_status_note = build_book_qna_status_note(
                leaf_node_count=len(leaf_nodes),
                qna_page_count=book_qna_page_count,
                qna_item_count=book_qna_item_count,
                page_status_counter=page_status_counter,
            )
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
                    # Q&A가 0건이어도 book 자체는 raw 인벤토리로 남겨 두고,
                    # 상태 컬럼으로만 왜 0건인지 추적 가능하게 만든다.
                    "has_qna_rows": build_bool_literal(book_qna_item_count > 0),
                    "qna_status": book_qna_status,
                    "qna_status_note": book_qna_status_note,
                    "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        flush_outputs(book_rows, qa_rows, law_link_rows, law_page_rows)
        log_info(
            "checkpoint",
            (
                # 중간 실패가 나도 어느 카테고리까지 수집됐는지 바로 읽히도록
                # checkpoint 로그에는 누적 row 수를 같이 남긴다.
                f"category={category_seq} saved "
                f"books={len(book_rows)} qa_rows={len(qa_rows)} "
                f"law_link_rows={len(law_link_rows)}"
            ),
        )

        if MAX_BOOKS is not None and crawled_book_count >= MAX_BOOKS:
            break

    if FETCH_LAW_PAGES:
        # 관련법령은 QA마다 중복으로 등장하므로, 원문 본문 수집을 켤 때는
        # canonical URL 기준으로 unique law URL만 추려서 별도 순회한다.
        unique_law_urls = []
        seen_law_urls = set()
        for row in law_link_rows:
            canonical_url = row["law_url_canonical"]
            if not canonical_url or canonical_url in seen_law_urls:
                continue
            seen_law_urls.add(canonical_url)
            unique_law_urls.append(row["law_url"])

        log_info("law", f"unique_law_urls={len(unique_law_urls)}")
        for index, law_url in enumerate(unique_law_urls, start=1):
            try:
                law_html, _ = fetch_html(session, law_url, referer=EASYLAW_BASE_URL)
                law_page_row = parse_law_page(law_html, law_url)
                if not law_page_row["body_text_raw"]:
                    log_warning("law", f"empty_body_text url={law_url}")
                law_page_rows.append(law_page_row)
            except requests.RequestException as error:
                log_warning("law", f"skip url={law_url} error={error}")
            if index % 50 == 0:
                log_info("law", f"crawled={index}")

    flush_outputs(book_rows, qa_rows, law_link_rows, law_page_rows)

    log_info("done", f"books={len(book_rows)}")
    log_info("done", f"qa_rows={len(qa_rows)}")
    log_info("done", f"law_link_rows={len(law_link_rows)}")
    log_info("done", f"law_pages={len(law_page_rows)}")


if __name__ == "__main__":
    crawl_easy_law()
