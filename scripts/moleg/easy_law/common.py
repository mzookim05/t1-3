import csv
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# 찾기 쉬운 생활법령 계열 수집기는 같은 사이트 구조를 공유하므로
# 카테고리 탐색, 세션 설정, 텍스트 정리, CSV 저장 같은 공통부를 한 파일로 모은다.
SCRIPT_DIR = Path(__file__).resolve().parent
# source별 하위 폴더로 스크립트를 나누면서 디렉토리 깊이가 한 단계 늘어났으므로
# 저장소 루트는 scripts/moleg/easy_law 기준에서 두 단계 위로 계산한다.
PROJECT_ROOT = SCRIPT_DIR.parents[2]

EASYLAW_BASE_URL = "https://www.easylaw.go.kr"
LAW_BASE_URL = "https://www.law.go.kr"
CATEGORY_URL_TEMPLATE = (
    EASYLAW_BASE_URL + "/CSP/CsmSortRetrieveLst.laf?sortType=cate&csmAstSeq={category_seq}"
)
LAW_URL_PREFIXES = ("http://www.law.go.kr", "https://www.law.go.kr", "/LSW/")

REQUEST_TIMEOUT = 30
REQUEST_RETRY_TOTAL = 5
REQUEST_BACKOFF_FACTOR = 1.0
REQUEST_SLEEP_SECONDS = 0.15

# 생활법령 책자형 카테고리 순회는 모든 탭 수집기의 공통 입력이므로
# 상위 18개 분야 정보도 공통 모듈에 둔다.
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

LOGGER = logging.getLogger("moleg_easy_law_common")


def configure_logging():
    # 같은 생활법령 계열 수집기끼리는 로그 형식을 통일해야
    # 탭별 실행 로그를 한 화면에서 비교할 때 헷갈리지 않는다.
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
    # 긴 수집 도중 중간 실패가 나더라도 기존 CSV를 보존하려면
    # 항상 임시 파일에 먼저 쓰고 최종 경로로 교체해야 안전하다.
    return output_path.with_suffix(output_path.suffix + ".tmp")


def build_session():
    # 생활법령 사이트는 서버 렌더링 HTML이라 브라우저 자동화가 필수는 아니다.
    # 대신 연결 재사용과 재시도를 켜 두어 긴 순회에서도 안정성을 높인다.
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
    # 카테고리 -> 책자 -> 탭 페이지 순으로 이동하는 구조라 referer를 같이 넘겨 둔다.
    headers = {}
    if referer:
        headers["Referer"] = referer

    response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.text, response.url


def clean_text(text):
    # raw 수집본은 원문을 최대한 보존하되
    # CSV 저장과 후속 파이프라인을 망가뜨리는 과한 공백만 정리한다.
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
    # 생활법령 계열 수집기에서 bool 컬럼 형식을 통일해야
    # 후속 정제에서 별도 형변환 규칙을 줄일 수 있다.
    return "true" if value else "false"


def make_sha1(value):
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def canonicalize_url(url, default_base):
    # 같은 페이지가 쿼리 순서만 다르게 중복 저장되지 않도록
    # 절대경로화와 쿼리 정렬을 함께 처리한다.
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
    current_heading = soup.select_one("div.real h5 span")
    if current_heading:
        return clean_text(current_heading.get_text(" ", strip=True))
    return ""


def parse_csm_seq(url):
    return extract_query_value(url, "csmSeq")


def parse_category_page(session, category_seq):
    # 상위 카테고리 페이지에서는 책자 목록과 csmSeq만 안정적으로 확보하면
    # 이후 탭별 수집기들이 같은 book 인벤토리를 공유할 수 있다.
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
    # 책자 공통 트리 구조는 탭과 무관하게 동일하므로
    # leaf 탐색도 공통 모듈에서 처리해 두는 편이 재사용성이 높다.
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


def write_csv(output_path, fieldnames, rows):
    # 여러 탭 수집기가 같은 방식으로 checkpoint 저장을 할 수 있게
    # 원자적 쓰기 로직을 공통 함수로 둔다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = build_temp_output_path(output_path)
    with open(temp_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, output_path)
