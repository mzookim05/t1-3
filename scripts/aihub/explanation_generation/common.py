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
    DATASET_SPECS,
    DATASET_MANIFEST_PATH,
    DEV_PATH,
    EVIDENCE_CARDS_PATH,
    GENERATOR_MAX_TOKENS,
    GENERATOR_MODEL_CANDIDATES,
    GENERATION_INPUT_VARIANTS,
    GENERATOR_TEMPERATURE,
    GENERATIONS_PATH,
    GROUNDING_LOG_PATH,
    INTERIM_DIR,
    JUDGE_MODEL_CANDIDATES,
    JUDGE_READY_SAMPLES_PATH,
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


def normalized_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


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
    return sorted(PROJECT_ROOT.glob(pattern))


def make_family_id(doc_type_name, info):
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
    wanted_id = str(info[ID_FIELD_BY_DOC_TYPE[doc_type_name]]).strip()
    exact_matches = [path for path in raw_paths if path.stem.split("_")[-1] == wanted_id]
    if exact_matches:
        return exact_matches[0]

    raise FileNotFoundError(f"{doc_type_name} raw 파일을 찾지 못했습니다: {wanted_id}")


def build_title(record):
    info = record["info"]
    doc_type_name = record["doc_type_name"]
    title_value = info.get(TITLE_FIELD_BY_DOC_TYPE[doc_type_name], "")
    if doc_type_name == "법령_QA":
        return f"{title_value} {info.get('smClass', '').strip()}".strip()
    return normalized_text(title_value)


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
            with request.urlopen(req, timeout=120) as response:
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
        for attempt in range(4):
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
                with request.urlopen(req, timeout=120) as response:
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
                if exc.code == 429 and attempt < 3:
                    # 무료/저비용 등급에서는 분당 호출량에 민감해 `429`가 자주 난다.
                    # 같은 모델로 잠시 기다렸다가 다시 치면 성공하는 경우가 많아서
                    # `v4`부터는 즉시 fallback으로 내리지 않고 짧게 재시도한다.
                    time.sleep(12 * (attempt + 1))
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
        },
    }
