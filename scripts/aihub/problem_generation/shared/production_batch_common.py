import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
# production batch 공용 prompt는 shared가 아니라 batch line 하위에 두어,
# 공용 코드와 실행별 prompt snapshot의 역할을 분리한다.
PROMPT_DIR = SCRIPT_DIR.parent / "production_batches" / "prompts"


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


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_temp_output_path(output_path):
    return Path(output_path).with_suffix(Path(output_path).suffix + ".tmp")


def write_text_atomic(output_path, text):
    # review/export markdown도 중간 실패 시 깨진 파일을 남기지 않도록 CSV/JSON과 같은 atomic write 규칙을 쓴다.
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


def prompt_path(name, prompt_dir=None):
    # difficulty patch처럼 별도 prompt set을 쓰는 line도 같은 helper를 재사용할 수 있게 한다.
    return Path(prompt_dir or PROMPT_DIR) / name


def load_prompt(name, prompt_dir=None):
    return prompt_path(name, prompt_dir).read_text(encoding="utf-8")


def snapshot_prompts(prompt_names, run_prompts_dir, prompt_dir=None):
    ensure_dirs(run_prompts_dir)
    for name in prompt_names:
        source_path = prompt_path(name, prompt_dir)
        target_path = Path(run_prompts_dir) / name
        ensure_parent(target_path)
        shutil.copy2(source_path, target_path)


def copy_file_to_run_inputs(source_path, run_inputs_dir, target_name=None):
    source_path = Path(source_path)
    target_path = Path(run_inputs_dir) / (target_name or source_path.name)
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


def call_openai_json(
    messages,
    response_label,
    model_candidates,
    temperature,
    max_tokens,
    timeout_seconds,
):
    load_root_env()
    api_key = os.environ.get("GENERATOR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GENERATOR_API_KEY가 없습니다.")

    errors = []
    for model_name in model_candidates:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if model_name.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
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


def call_gemini_json(
    prompt_text,
    response_label,
    model_candidates,
    temperature,
    timeout_seconds,
    allowed_error_tags=None,
):
    load_root_env()
    api_key = os.environ.get("JUDGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JUDGE_API_KEY가 없습니다.")

    errors = []
    for model_name in model_candidates:
        for attempt in range(2):
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "responseMimeType": "application/json",
                },
            }
            req = request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                with request.urlopen(req, timeout=timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                parts = response_payload["candidates"][0]["content"]["parts"]
                content_text = "".join(part.get("text", "") for part in parts)
                parsed = safe_parse_json(content_text)
                if allowed_error_tags is not None:
                    error_tags = parsed.get("error_tags", [])
                    parsed["error_tags"] = [tag for tag in error_tags if tag in allowed_error_tags]
                return {
                    "model": model_name,
                    "response_label": response_label,
                    "payload": response_payload,
                    "json": parsed,
                }
            except error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="ignore")[:400]
                if exc.code in (429, 503) and attempt < 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                errors.append({"model": model_name, "status_code": exc.code, "text": error_text})
                break
            except Exception as exc:  # noqa: BLE001
                errors.append({"model": model_name, "status_code": "client_error", "text": str(exc)[:400]})
                break

    raise RuntimeError(f"Gemini 호출 실패: {errors}")


def load_selected_seed_ids(merged_path):
    selected_seed_ids = set()
    for row in load_csv_rows(merged_path):
        if row.get("selected_for_seed") == "예":
            selected_seed_ids.add(row["seed_sample_id"])
    return selected_seed_ids


def json_dumps_stable(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_jsonl_count(path):
    with open(path, encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def load_csv_count(path):
    with open(path, encoding="utf-8-sig") as input_file:
        return max(0, sum(1 for _ in input_file) - 1)
