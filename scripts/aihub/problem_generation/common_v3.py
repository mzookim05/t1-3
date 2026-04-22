import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from settings_v3 import (
    ALLOWED_ERROR_TAGS,
    GENERATED_PROBLEMS_PATH,
    GENERATOR_API_TIMEOUT_SECONDS,
    GENERATOR_MAIN_CHECKPOINT_EVERY,
    GENERATOR_MAX_TOKENS,
    GENERATOR_MODEL_CANDIDATES,
    GENERATOR_TEMPERATURE,
    INTERIM_DIR,
    JUDGE_API_TIMEOUT_SECONDS,
    JUDGE_MODEL_CANDIDATES,
    JUDGE_TEMPERATURE,
    PROMPT_DIR,
    PROJECT_ROOT,
    PROCESSED_DIR,
    REFERENCE_TRAIN_PATH,
    RUN_DIR,
    RUN_EXPORTS_DIR,
    RUN_GENERATIONS_DIR,
    RUN_INPUTS_DIR,
    RUN_JUDGE_LOGS_DIR,
    RUN_MANIFEST_PATH,
    RUN_MERGED_DIR,
    RUN_NAME,
    RUN_PROMPTS_DIR,
    SEED_READY_PATH,
    SEED_REGISTRY_PATH,
    VERSION_TAG,
)


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
    # `v3` 산출물은 기존 `v1/v2`와 완전히 분리된 경로로 관리해 비교 기준선을 흔들지 않는다.
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


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text_or_blank(value):
    if value is None:
        return ""
    return str(value).strip()


def normalized_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text):
    return re.findall(r"[A-Za-z0-9가-힣]+", normalized_text(text).lower())


def split_sentences(text):
    normalized = normalized_text(text)
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?。]|다\.)\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def prompt_path(name):
    return PROMPT_DIR / name


def load_prompt(name):
    return prompt_path(name).read_text(encoding="utf-8")


def snapshot_prompts(prompt_names):
    ensure_run_dirs()
    for name in prompt_names:
        source_path = prompt_path(name)
        target_path = RUN_PROMPTS_DIR / name
        ensure_parent(target_path)
        shutil.copy2(source_path, target_path)


def copy_file_to_run_inputs(source_path, target_name=None):
    source_path = Path(source_path)
    target_path = RUN_INPUTS_DIR / (target_name or source_path.name)
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
            errors.append(
                {
                    "model": model_name,
                    "status_code": exc.code,
                    "text": exc.read().decode("utf-8", errors="ignore")[:400],
                }
            )
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
                content_text = parts[0]["text"]
                return {
                    "model": model_name,
                    "response_label": response_label,
                    "payload": response_payload,
                    "json": safe_parse_json(content_text),
                }
            except error.HTTPError as exc:
                errors.append(
                    {
                        "model": model_name,
                        "attempt": attempt + 1,
                        "status_code": exc.code,
                        "text": exc.read().decode("utf-8", errors="ignore")[:400],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "model": model_name,
                        "attempt": attempt + 1,
                        "status_code": "client_error",
                        "text": str(exc)[:400],
                    }
                )
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Gemini 호출 실패: {errors}")


def build_run_manifest():
    return {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "reference_train_path": str(REFERENCE_TRAIN_PATH),
        "seed_registry_path": str(SEED_REGISTRY_PATH),
        "seed_ready_path": str(SEED_READY_PATH),
        "generated_problems_path": str(GENERATED_PROBLEMS_PATH),
        "allowed_error_tags": ALLOWED_ERROR_TAGS,
        "generator_main_checkpoint_every": GENERATOR_MAIN_CHECKPOINT_EVERY,
        "created_at_utc": utc_now_iso(),
    }
