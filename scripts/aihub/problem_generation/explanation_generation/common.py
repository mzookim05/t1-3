import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from settings import (
    ACTIVE_GENERATION_VARIANT,
    ALLOWED_ERROR_TAGS,
    ANSWER_LOG_PATH,
    AUDIT_QUEUE_PATH,
    DATASET_SPECS,
    DATASET_MANIFEST_PATH,
    DEV_PATH,
    EVIDENCE_CARDS_PATH,
    GENERATOR_MAX_TOKENS,
    GENERATOR_MODEL_CANDIDATES,
    GENERATION_INPUT_VARIANTS,
    GENERATOR_TEMPERATURE,
    GENERATIONS_PATH,
    GENERATOR_API_TIMEOUT_SECONDS,
    GROUNDING_LOG_PATH,
    INTERIM_DIR,
    JUDGE_MODEL_CANDIDATES,
    JUDGE_READY_SAMPLES_PATH,
    JUDGE_API_TIMEOUT_SECONDS,
    JUDGE_TEMPERATURE,
    MEETING_EXAMPLES_CSV_PATH,
    MEETING_EXAMPLES_MD_PATH,
    MERGED_SCORES_PATH,
    PEDAGOGY_LOG_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    PROMPT_DIR,
    RUN_DIR,
    RUN_EXPORTS_DIR,
    RUN_GENERATIONS_DIR,
    RUN_INPUTS_DIR,
    RUN_JUDGE_LOGS_DIR,
    RUN_MERGED_DIR,
    RUN_NAME,
    RUN_PROMPTS_DIR,
    SAMPLE_REGISTRY_PATH,
    TEST_PATH,
    TRAIN_PATH,
    TRANSFORMED_SAMPLES_PATH,
    VERSION_TAG,
)


ID_FIELD_BY_DOC_TYPE = {
    "법령_QA": "lawId",
    "해석례_QA": "interpreId",
    "결정례_QA": "determintId",
    "판결문_QA": "precedId",
}

TITLE_FIELD_BY_DOC_TYPE = {
    "법령_QA": "title",
    "해석례_QA": "agenda",
    "결정례_QA": "caseName",
    "판결문_QA": "caseName",
}

RAW_DIR_KEYWORD_BY_DOC_TYPE = {
    "법령_QA": "TS_법령",
    "해석례_QA": "TS_해석례",
    "결정례_QA": "TS_결정례",
    "판결문_QA": "TS_판결문",
}

RAW_ID_COLUMN_BY_DOC_TYPE = {
    "법령_QA": "법령일련번호",
    "해석례_QA": "해석례일련번호",
    "결정례_QA": "결정례일련번호",
    "판결문_QA": "판례일련번호",
}

CANONICAL_TITLE_FIELD_BY_DOC_TYPE = {
    "법령_QA": "title",
    "해석례_QA": "agenda",
    "결정례_QA": "caseName",
    "판결문_QA": "caseName",
}


def load_root_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_run_dirs():
    for path in (
        INTERIM_DIR,
        PROCESSED_DIR,
        RUN_DIR,
        RUN_PROMPTS_DIR,
        RUN_INPUTS_DIR,
        RUN_GENERATIONS_DIR,
        RUN_JUDGE_LOGS_DIR,
        RUN_MERGED_DIR,
        RUN_EXPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_temp_output_path(output_path):
    return Path(output_path).with_suffix(Path(output_path).suffix + ".tmp")


def write_text_atomic(output_path, text):
    output_path = Path(output_path)
    temp_path = build_temp_output_path(output_path)
    ensure_parent(temp_path)
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)


def write_json_atomic(output_path, payload):
    output_path = Path(output_path)
    temp_path = build_temp_output_path(output_path)
    ensure_parent(temp_path)
    with open(temp_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    temp_path.replace(output_path)


def write_jsonl_atomic(output_path, rows):
    output_path = Path(output_path)
    temp_path = build_temp_output_path(output_path)
    ensure_parent(temp_path)
    with open(temp_path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)


def write_csv_atomic(output_path, rows, fieldnames):
    output_path = Path(output_path)
    temp_path = build_temp_output_path(output_path)
    ensure_parent(temp_path)
    with open(temp_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def text_or_blank(value):
    if value is None:
        return ""
    return str(value).strip()


def normalized_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def collapse_json_sentence_items(sentences):
    if isinstance(sentences, str):
        return [sentences]

    cleaned = [text_or_blank(sentence) for sentence in sentences if text_or_blank(sentence)]
    if not cleaned:
        return []

    single_char_ratio = sum(1 for sentence in cleaned if len(sentence) <= 1) / len(cleaned)
    if len(cleaned) >= 20 and single_char_ratio >= 0.7:
        return ["".join(cleaned)]
    return cleaned


def split_structured_sections(text, doc_type_name):
    normalized = normalized_text(text)
    if not normalized:
        return []

    if doc_type_name != "해석례_QA":
        return [("본문", normalized)]

    pattern = re.compile(r"(질의요지|질의배경|회답|이유)\s*:")
    matches = list(pattern.finditer(normalized))
    if not matches:
        return [("본문", normalized)]

    sections = []
    for index, match in enumerate(matches):
        header = match.group(1)
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        content = normalized_text(normalized[content_start:content_end])
        if content:
            sections.append((header, content))
    return sections or [("본문", normalized)]


def extract_structured_section_text(text, doc_type_name, section_name):
    for header, content in split_structured_sections(text, doc_type_name):
        if header == section_name and content:
            return content
    return ""


def classify_law_question(original_input):
    normalized = normalized_text(original_input)

    if any(keyword in normalized for keyword in ("요건", "성립", "조건", "갖추어야", "요소")):
        return "requirement"
    if any(keyword in normalized for keyword in ("의미", "뜻", "정의", "일컫", "란 무엇", "무엇을 말", "말하나요")):
        return "definition"
    if any(keyword in normalized for keyword in ("범위", "어떤", "어느", "대상", "해당하는")):
        return "scope"
    return "criteria"


def extract_law_subject(original_input):
    question = normalized_text(original_input).rstrip("?")
    patterns = [
        r".+에서 일컫는 (.+?)란 무엇인가요$",
        r"(.+?)란 무엇인가요$",
        r"(.+?)는 무엇인가요$",
        r"(.+?)는 어떤 .+?를 말하나요$",
        r"(.+?)은 어떤 .+?인가요$",
        r"(.+?)는 어떤 .+?인가요$",
        r"(.+?)의 요건은 무엇인가요$",
        r"(.+?)의 범위는 무엇인가요$",
    ]
    for pattern in patterns:
        match = re.match(pattern, question)
        if match:
            return normalized_text(match.group(1))
    return question


def split_sentences(text):
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", normalized_text(text))
        if piece.strip()
    ]
    if pieces:
        return pieces

    fallback = [piece.strip() for piece in re.split(r"(?<=다)\s+", normalized_text(text)) if piece.strip()]
    return fallback if fallback else [normalized_text(text)] if normalized_text(text) else []


def count_words(text):
    return len([token for token in normalized_text(text).split(" ") if token])


def looks_like_complete_sentence(text):
    normalized = normalized_text(text)
    if not normalized:
        return False
    if re.search(r"[.!?]$", normalized):
        return True
    return bool(
        re.search(
            r"(다|요|니다|이다|한다|된다|없다|있다|였다|하였다|합니다|입니다)$",
            normalized,
        )
    )


def pick_short_answer(label_output):
    sentences = split_sentences(label_output)
    if not sentences:
        return ""

    normalized_sentences = [normalized_text(sentence) for sentence in sentences if normalized_text(sentence)]
    if not normalized_sentences:
        return ""

    # `v3`에서는 어절 상한보다 문장 완결성을 우선한다.
    # 이전처럼 25어절에서 기계적으로 자르면 정답 문장이 중간에서 끊겨 학습셋 품질이 흔들린다.
    first_sentence = normalized_sentences[0]
    if 8 <= count_words(first_sentence) <= 25 and looks_like_complete_sentence(first_sentence):
        return first_sentence

    if len(normalized_sentences) > 1:
        combined = f"{normalized_sentences[0]} {normalized_sentences[1]}".strip()
        if 8 <= count_words(combined) <= 25 and looks_like_complete_sentence(combined):
            return combined

    for sentence in normalized_sentences:
        if 8 <= count_words(sentence) and looks_like_complete_sentence(sentence):
            return sentence

    if len(normalized_sentences) > 1:
        combined = f"{normalized_sentences[0]} {normalized_sentences[1]}".strip()
        if looks_like_complete_sentence(combined):
            return combined

    return first_sentence


def pick_long_answer(label_output):
    sentences = split_sentences(label_output)
    if not sentences:
        return ""

    chosen = []
    total_words = 0
    for sentence in sentences[:4]:
        word_count = count_words(sentence)
        if chosen and total_words + word_count > 80:
            break
        chosen.append(sentence)
        total_words += word_count
        if len(chosen) >= 2 and total_words >= 25:
            break
    return " ".join(chosen).strip()


def strip_statute_lead(text):
    stripped = normalized_text(text)
    stripped = re.sub(r"^제\d+조(?:의\d+)?\([^)]+\)\s*", "", stripped)
    return stripped


def is_statute_heading_only(text):
    stripped = normalized_text(text)
    return bool(re.fullmatch(r"제\d+조(?:의\d+)?\([^)]+\)", stripped))


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_sample_indices(total_count, sample_count):
    if total_count <= 0 or sample_count <= 0:
        return []

    if sample_count >= total_count:
        return list(range(total_count))

    indices = []
    for order in range(sample_count):
        fraction = (order + 1) / (sample_count + 1)
        index = round(fraction * (total_count - 1))
        if index not in indices:
            indices.append(index)

    while len(indices) < sample_count:
        candidate = indices[-1] + 1
        if candidate >= total_count:
            break
        indices.append(candidate)

    return indices[:sample_count]


def list_label_files(pattern):
    return [path for path in sorted(PROJECT_ROOT.glob(pattern)) if path.name != ".extract_complete.json"]


def list_raw_files(pattern):
    return [path for path in sorted(PROJECT_ROOT.glob(pattern)) if path.name != ".extract_complete.json"]


def normalize_label_payload(label_path, payload, doc_type_name):
    info = dict(payload.get("info", {}))

    if "label" in payload:
        info.setdefault("source_schema", "aihub_03_04_label")
        return {
            "info": info,
            "label": dict(payload["label"]),
        }

    if "taskinfo" in payload:
        stem_parts = Path(label_path).stem.split("_")
        # `01·02` 라벨 파일명은 `{도메인}_{유형}_질의응답_{번호}` 구조라,
        # 한글 정규화 차이에 흔들리지 않도록 penultimate token을 위치 기준으로 제거한다.
        if len(stem_parts) >= 4 and not stem_parts[-2].isdigit():
            raw_match_stem = "_".join(stem_parts[:-2] + [stem_parts[-1]])
        else:
            raw_match_stem = Path(label_path).stem
        info.update(
            {
                "source_schema": "aihub_01_02_taskinfo",
                "raw_match_stem": raw_match_stem,
                # `01·02`는 `03·04`처럼 stable numeric id가 없으므로,
                # 실제 원천 파일 stem을 family 기준으로 삼아 같은 원문 중복 유입을 막는다.
                "family_id_hint": f"{doc_type_name}::{raw_match_stem}",
            }
        )

        if doc_type_name == "법령_QA":
            info.setdefault("lawId", text_or_blank(info.get("statute_name")) or raw_match_stem)
            info.setdefault("title", text_or_blank(info.get("statute_name")))
            info.setdefault("smClass", "")
        elif doc_type_name == "해석례_QA":
            info.setdefault("interpreId", text_or_blank(info.get("doc_id")) or raw_match_stem)
            info.setdefault("agenda", text_or_blank(info.get("title")))
        elif doc_type_name == "결정례_QA":
            info.setdefault("determintId", text_or_blank(info.get("doc_id")) or raw_match_stem)
            info.setdefault(
                "caseName",
                text_or_blank(info.get("title")) or text_or_blank(info.get("document_type")) or raw_match_stem,
            )
        elif doc_type_name == "판결문_QA":
            info.setdefault("precedId", text_or_blank(info.get("doc_id")) or raw_match_stem)
            info.setdefault("caseName", text_or_blank(info.get("casenames")) or raw_match_stem)

        taskinfo = payload.get("taskinfo", {})
        taskinfo_sentences = text_or_blank(taskinfo.get("sentences"))
        taskinfo_output = text_or_blank(taskinfo.get("output"))
        if doc_type_name == "해석례_QA":
            authoritative_reply = extract_structured_section_text(
                taskinfo_sentences,
                doc_type_name,
                "회답",
            )
            if authoritative_reply:
                info["taskinfo_output_original"] = taskinfo_output
                taskinfo_output = authoritative_reply
        return {
            "info": info,
            "label": {
                "instruction": text_or_blank(taskinfo.get("instruction")),
                "input": text_or_blank(taskinfo.get("input")),
                "output": taskinfo_output,
            },
        }

    raise ValueError(f"지원하지 않는 라벨 스키마입니다: {label_path}")


def make_family_id(doc_type_name, info):
    if info.get("family_id_hint"):
        return str(info["family_id_hint"])
    if doc_type_name == "법령_QA":
        return f"{info['lawId']}::{info.get('smClass', '').strip()}"
    if doc_type_name == "해석례_QA":
        return str(info["interpreId"])
    if doc_type_name == "결정례_QA":
        return str(info["determintId"])
    return str(info["precedId"])


def tokenize(text):
    return [token for token in re.findall(r"[가-힣A-Za-z0-9]+", normalized_text(text).lower()) if len(token) > 1]


def lexical_overlap_score(text, query_tokens):
    if not query_tokens:
        return 0
    text_tokens = set(tokenize(text))
    return sum(1 for token in query_tokens if token in text_tokens)


def locate_raw_path(raw_paths, doc_type_name, info):
    raw_match_stem = text_or_blank(info.get("raw_match_stem"))
    if raw_match_stem:
        exact_matches = [path for path in raw_paths if path.stem == raw_match_stem]
        if exact_matches:
            return exact_matches[0]

    wanted_id = str(info[ID_FIELD_BY_DOC_TYPE[doc_type_name]]).strip()
    exact_matches = [path for path in raw_paths if path.stem.split("_")[-1] == wanted_id]
    if exact_matches:
        return exact_matches[0]

    raise FileNotFoundError(f"{doc_type_name} raw 파일을 찾지 못했습니다: {wanted_id}")


def build_title(record):
    info = record["info"]
    doc_type_name = record["doc_type_name"]
    title_value = info.get(TITLE_FIELD_BY_DOC_TYPE[doc_type_name], "") or info.get(
        CANONICAL_TITLE_FIELD_BY_DOC_TYPE[doc_type_name],
        "",
    )
    if doc_type_name == "법령_QA":
        return f"{title_value} {info.get('smClass', '').strip()}".strip()
    return normalized_text(title_value)


def infer_json_section(doc_type_name, text, current_section):
    stripped = normalized_text(text)
    if not stripped:
        return current_section or ""

    if doc_type_name == "법령_QA":
        if re.match(r"^제\d+조(?:의\d+)?", stripped):
            return "조문"
        if re.match(r"^\d+\.\s*", stripped):
            return "호"
        if re.match(r"^[가-하]\.\s*", stripped):
            return "목"
        return current_section or "조문"

    if doc_type_name == "해석례_QA":
        for section_name in ("질의요지", "회답", "이유"):
            if stripped.startswith(section_name):
                return section_name
        return current_section or "본문"

    section_keywords = (
        "판시사항",
        "판결요지",
        "참조조문",
        "주문",
        "이유",
        "판단",
        "기초사실",
        "절차의 경위",
        "청구취지",
        "신청취지",
        "범죄사실",
        "쟁점",
    )
    for keyword in section_keywords:
        if stripped.startswith(keyword) or stripped.startswith(f"【{keyword}】"):
            return keyword

    return current_section or "전문"


def load_raw_rows(raw_path, doc_type_name):
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() == ".csv":
        return load_csv_rows(raw_path)

    payload = load_json(raw_path)
    sentences = collapse_json_sentence_items(payload.get("sentences", []))

    rows = []
    current_section = ""
    sentence_number = 1
    for sentence in sentences:
        for header, content in split_structured_sections(sentence, doc_type_name):
            sentence_parts = split_sentences(content) if len(content) >= 240 else [content]
            for sentence_part in sentence_parts:
                text = normalized_text(sentence_part)
                if not text:
                    continue
                if header != "본문" and not text.startswith(header):
                    text = f"{header} {text}"
                current_section = infer_json_section(doc_type_name, text, current_section)
                rows.append(
                    {
                        "문장번호": str(sentence_number),
                        "구분": current_section or "본문",
                        "내용": text,
                    }
                )
                sentence_number += 1
    return rows


def prompt_path(name):
    return PROMPT_DIR / name


def load_prompt(name):
    return prompt_path(name).read_text(encoding="utf-8")


def snapshot_prompts(prompt_names):
    for name in prompt_names:
        source_path = prompt_path(name)
        target_path = RUN_PROMPTS_DIR / name
        ensure_parent(target_path)
        shutil.copy2(source_path, target_path)


def render_prompt(template_text, variables):
    rendered = template_text
    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def safe_parse_json(text):
    stripped = normalized_text(text)
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def call_openai_json(messages, response_label):
    load_root_env()
    api_key = os.environ.get("GENERATOR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GENERATOR_API_KEY가 없습니다.")

    errors = []
    for model_name in GENERATOR_MODEL_CANDIDATES:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": GENERATOR_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        # `gpt-5` 계열은 `max_tokens` 대신 `max_completion_tokens`를 요구한다.
        # 구형 호환 모델은 기존 `max_tokens`를 유지해 fallback 경로를 보존한다.
        if model_name.startswith("gpt-5"):
            payload["max_completion_tokens"] = GENERATOR_MAX_TOKENS
        else:
            payload["max_tokens"] = GENERATOR_MAX_TOKENS
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with request.urlopen(req, timeout=GENERATOR_API_TIMEOUT_SECONDS) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            return {
                "model": model_name,
                "response_label": response_label,
                "payload": response_payload,
                "json": safe_parse_json(content),
            }
        except error.HTTPError as exc:
            errors.append({"model": model_name, "status_code": exc.code, "text": exc.read().decode("utf-8", errors="ignore")[:400]})
        except Exception as exc:  # noqa: BLE001
            errors.append({"model": model_name, "status_code": "client_error", "text": str(exc)[:400]})

    raise RuntimeError(f"OpenAI 호출 실패: {errors}")


def call_gemini_json(prompt_text, response_label):
    load_root_env()
    api_key = os.environ.get("JUDGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY가 없습니다.")

    errors = []
    for model_name in JUDGE_MODEL_CANDIDATES:
        for attempt in range(2):
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": JUDGE_TEMPERATURE,
                    "responseMimeType": "application/json",
                },
            }
            req = request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                with request.urlopen(req, timeout=JUDGE_API_TIMEOUT_SECONDS) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                parts = response_payload["candidates"][0]["content"]["parts"]
                content_text = "".join(part.get("text", "") for part in parts)
                parsed = safe_parse_json(content_text)
                error_tags = parsed.get("error_tags", [])
                parsed["error_tags"] = [tag for tag in error_tags if tag in ALLOWED_ERROR_TAGS]
                return {
                    "model": model_name,
                    "response_label": response_label,
                    "payload": response_payload,
                    "json": parsed,
                }
            except error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="ignore")[:400]
                if exc.code in (429, 503) and attempt < 1:
                    # `v6` 메인 런에서는 judge가 전체 파이프라인 병목이 되기 쉬워,
                    # 장시간 재시도로 런이 멈춘 것처럼 보이는 상황을 줄이기 위해
                    # 짧은 1회 재시도 뒤에는 local fallback으로 넘긴다.
                    time.sleep(3 * (attempt + 1))
                    continue
                errors.append({"model": model_name, "status_code": exc.code, "text": error_text})
                break
            except Exception as exc:  # noqa: BLE001
                errors.append({"model": model_name, "status_code": "client_error", "text": str(exc)[:400]})
                break

    raise RuntimeError(f"Gemini 호출 실패: {errors}")


def copy_file_to_run_inputs(source_path, target_name=None):
    source_path = Path(source_path)
    target_path = RUN_INPUTS_DIR / (target_name or source_path.name)
    ensure_parent(target_path)
    shutil.copy2(source_path, target_path)


def build_run_manifest():
    return {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "generated_at_utc": utc_now_iso(),
        "run_dir": str(RUN_DIR),
        "interim_dir": str(INTERIM_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "generator_model_candidates": list(GENERATOR_MODEL_CANDIDATES),
        "judge_model_candidates": list(JUDGE_MODEL_CANDIDATES),
        "active_generation_variant": ACTIVE_GENERATION_VARIANT,
        "generation_input_variants": list(GENERATION_INPUT_VARIANTS),
        "dataset_specs": DATASET_SPECS,
        "artifact_paths": {
            "sample_registry": str(SAMPLE_REGISTRY_PATH),
            "evidence_cards": str(EVIDENCE_CARDS_PATH),
            "transformed_samples": str(TRANSFORMED_SAMPLES_PATH),
            "judge_ready_samples": str(JUDGE_READY_SAMPLES_PATH),
            "generated_explanations": str(GENERATIONS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_answer_log": str(ANSWER_LOG_PATH),
            "judge_pedagogy_log": str(PEDAGOGY_LOG_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "meeting_examples_md": str(MEETING_EXAMPLES_MD_PATH),
            "meeting_examples_csv": str(MEETING_EXAMPLES_CSV_PATH),
            "train": str(TRAIN_PATH),
            "dev": str(DEV_PATH),
            "test": str(TEST_PATH),
            "dataset_manifest": str(DATASET_MANIFEST_PATH),
            "audit_queue": str(AUDIT_QUEUE_PATH),
        },
    }
