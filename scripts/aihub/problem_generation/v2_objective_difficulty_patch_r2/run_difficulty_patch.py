import re
import sys
from pathlib import Path

# `r2`는 새 subtype이 아니라 기존 difficulty patch를 좁게 보강하는 실행선이므로,
# 검증된 `r1` runner를 그대로 재사용하되 설정과 guardrail만 명시적으로 덮어쓴다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.v2_objective_difficulty_patch import run_difficulty_patch as base


ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM = base.build_local_fallback_problem
ORIGINAL_BUILD_RUN_MANIFEST = base.build_run_manifest
ORIGINAL_LOAD_REFERENCE_V2_ROWS = base.load_reference_v2_rows

VERSION_TAG = "v2_difficulty_patch_r2"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-24_012725"
RUN_PURPOSE = "objective_nearmiss_refinement"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROMPT_DIR = SCRIPT_DIR / "prompts"
INTERIM_DIR = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / VERSION_TAG
PROCESSED_DIR = base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / VERSION_TAG
RUN_DIR = base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

REFERENCE_SEED_READY_PATH = base.REFERENCE_SEED_READY_PATH
REFERENCE_V2_MERGED_PATH = base.REFERENCE_V2_MERGED_PATH
REFERENCE_PATCH_MERGED_PATH = base.MERGED_SCORES_PATH
REFERENCE_PATCH_ROWS_CACHE = None

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

SIDE_BY_SIDE_MD_PATH = RUN_EXPORTS_DIR / f"side_by_side_examples_{VERSION_TAG}.md"
SIDE_BY_SIDE_CSV_PATH = RUN_EXPORTS_DIR / f"side_by_side_examples_{VERSION_TAG}.csv"

LAW_MULTI_ISSUE_PATTERNS = [
    r"요건과 방식",
    r"기준과 정함 방식",
    r"구비요건과 지정 경로",
    r"가치와 그 산정기준의 정함 방식",
    r"요건과 절차",
    r"기준과 절차",
]

ENDING_NOISE_PATTERNS = [
    r"\s*옳은 설명을 고르시오\.?$",
    r"\s*가장 적절한 설명을 고르시오\.?$",
    r"\s*올바른 적용 범위를 고르시오\.?$",
    r"\s*옳은 것은 무엇인가\??$",
    r"\s*가장 옳은 것은 무엇인가\??$",
    r"\s*가장 적절한 것은 무엇인가\??$",
    r"\s*가장 적절한 것은$",
    r"\s*가장 정확하게 고르시오\.?$",
    r"\s*가장 정확하게 선택하시오\.?$",
    r"\s*판단하시오\.?$",
    r"\s*무엇인가요\??$",
    r"\s*무엇입니까\??$",
]


def configure_base():
    # `r2` 산출물이 `r1`과 섞이지 않도록 path와 tag를 한 번에 재배선한다.
    base.VERSION_TAG = VERSION_TAG
    base.RUN_DATE = RUN_DATE
    base.RUN_PURPOSE = RUN_PURPOSE
    base.RUN_NAME = RUN_NAME
    base.PROMPT_DIR = PROMPT_DIR
    base.INTERIM_DIR = INTERIM_DIR
    base.PROCESSED_DIR = PROCESSED_DIR
    base.RUN_DIR = RUN_DIR
    base.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    base.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    base.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    base.RUN_MERGED_DIR = RUN_MERGED_DIR
    base.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    base.SEED_READY_PATH = SEED_READY_PATH
    base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    base.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    base.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    base.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    base.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    base.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    base.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    base.SIDE_BY_SIDE_MD_PATH = SIDE_BY_SIDE_MD_PATH
    base.SIDE_BY_SIDE_CSV_PATH = SIDE_BY_SIDE_CSV_PATH
    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Keyedness": KEYEDNESS_LOG_PATH,
        "DistractorFit": DISTRACTORFIT_LOG_PATH,
        "NearMiss": NEARMISS_LOG_PATH,
    }
    # `r2`는 same comparator를 유지하되 법령형 stem 단일 쟁점화를 더 강하게 요구한다.
    base.DOC_TYPE_NEARMISS_HINTS = {
        "법령_QA": "같은 조문 또는 같은 요건 체계를 공유하되 필수 요건, 주체, 기간, 효과 중 하나만 틀린 오답을 만든다. stem에는 요건, 기준, 절차, 방식, 경로 중 하나의 predicate만 남기고 둘 이상을 함께 묻지 않는다.",
        "해석례_QA": base.DOC_TYPE_NEARMISS_HINTS["해석례_QA"],
        "결정례_QA": base.DOC_TYPE_NEARMISS_HINTS["결정례_QA"],
        "판결문_QA": base.DOC_TYPE_NEARMISS_HINTS["판결문_QA"],
    }


def trim_to_first_sentence(text):
    normalized = base.normalized_text(text)
    if not normalized:
        return normalized
    return re.split(r"(?<=[.?!])\s+|(?<=다\.)\s+", normalized)[0].strip()


def normalize_stem_ending(text, mode_name):
    # `r1`에서 보였던 "무엇인가 ... 옳은 설명을 고르시오" 중복 ending을 한 번에 접어낸다.
    stem = base.normalized_text(text)
    for pattern in ENDING_NOISE_PATTERNS:
        stem = re.sub(pattern, "", stem)
    stem = shape_stem_core_for_choice(stem.rstrip(" .?"))
    return f"{stem} {base.MODE_TO_STEM_ENDING[mode_name]}"


def normalize_seed_stem_core(seed):
    # source seed를 다시 쓰더라도 법령형 predicate를 하나만 남긴다는 원칙이 보이도록 짧게 다듬는다.
    core = base.normalized_text(seed["transformed_problem"])
    specialized_replacements = [
        (r"(.+?)은 어떻게 정해지나요의 판단 기준은 무엇인가요\??$", r"\1 정함 기준"),
        (r"(.+?)은 어떻게 정해지나요\??$", r"\1 정함 기준"),
        (r"(.+?)어떤 조건을 갖추어야 하나요의 요건은 무엇인가요\??$", r"\1갖추어야 할 요건"),
        (r"(.+?)어떤 조건을 갖추어야 하나요\??$", r"\1갖추어야 할 요건"),
    ]
    for pattern, replacement in specialized_replacements:
        core = re.sub(pattern, replacement, core)

    replacements = [
        (r"\s*기준으로,\s*", "에 따른 "),
        (r"\s*기준으로\s*", "에 따른 "),
        (r"와 관련된 핵심 판단 기준은 무엇인가요\??$", ""),
        (r"의 판단 기준은 무엇인가요\??$", " 판단 기준"),
        (r"의 요건은 무엇인가요\??$", " 요건"),
        (r"의 이유는 무엇인가요\??$", " 이유"),
        (r"신고해야 하는 서류의 판단 기준은 무엇인가요\??$", "신고 첨부서류"),
        (r"무엇인가요\??$", ""),
        (r"무엇입니까\??$", ""),
    ]
    for pattern, replacement in replacements:
        core = re.sub(pattern, replacement, core)
    return core.rstrip(" ,.?")


def shape_stem_core_for_choice(stem):
    # command phrase를 제거한 뒤에도 문장이 "핵심을", "이유를"처럼 어색하게 끝나지 않도록 선택지형 문장으로 마감한다.
    normalized = base.normalized_text(stem)
    tail_replacements = [
        (r"핵심을$", "핵심에 관하여"),
        (r"판단 기준$", "판단 기준에 관하여"),
        (r"정함 기준$", "정함 기준에 관하여"),
        (r"신고 첨부서류$", "신고 첨부서류에 관하여"),
        (r"요건$", "요건에 관하여"),
        (r"이유$", "이유에 관하여"),
    ]
    for pattern, replacement in tail_replacements:
        if re.search(pattern, normalized):
            return re.sub(pattern, replacement, normalized)
    return normalized


def looks_multi_issue_law_stem(stem):
    normalized = base.normalized_text(stem)
    if any(re.search(pattern, normalized) for pattern in LAW_MULTI_ISSUE_PATTERNS):
        return True
    if len(re.split(r"(?<=[.?!])\s+|(?<=다\.)\s+", normalized)) > 1:
        return True
    issue_keywords = ("요건", "기준", "방식", "절차", "경로", "효과")
    keyword_hit_count = sum(keyword in normalized for keyword in issue_keywords)
    return keyword_hit_count >= 3 and ("과" in normalized or "및" in normalized)


def tighten_law_stem(seed, stem):
    # 법령형은 generation이 좋아도 stem이 넓어지면 hard fail이 되므로, 후처리에서 단일 predicate를 강제한다.
    tightened = trim_to_first_sentence(stem)
    replacements = [
        (r"요건과 방식", "요건"),
        (r"기준과 정함 방식", "기준"),
        (r"구비요건과 지정 경로", "지정 요건"),
        (r"지정 요건과 방식", "지정 요건"),
        (r"가치와 그 산정기준의 정함 방식", "보험가액의 정함 기준"),
        (r"을 중심으로 판단하시오\.?", ""),
        (r"조문은 .*?함께 두고 있다", ""),
        (r"판단의 초점은 .*", ""),
    ]
    for pattern, replacement in replacements:
        tightened = re.sub(pattern, replacement, tightened)
    if looks_multi_issue_law_stem(tightened):
        tightened = normalize_seed_stem_core(seed)
    return normalize_stem_ending(tightened, seed["problem_generation_mode"])


def contains_token_phrase(source_text, target_text, min_tokens):
    source_tokens = base.tokenize(source_text)
    target_tokens_text = " ".join(base.tokenize(target_text))
    if len(source_tokens) < min_tokens:
        return False
    for index in range(len(source_tokens) - min_tokens + 1):
        phrase = " ".join(source_tokens[index : index + min_tokens])
        if phrase and phrase in target_tokens_text:
            return True
    return False


def exposes_answer_too_directly(seed, payload):
    correct_choice = payload["correct_choice"]
    correct_text = payload[f"choice_{correct_choice.lower()}"]
    stem = payload["generated_stem"]
    is_law_row = seed["doc_type_name"] == "법령_QA"
    stem_threshold = 0.58 if is_law_row else 0.68
    choice_threshold = 0.88 if is_law_row else 0.92

    if base.overlap_ratio(seed["short_answer"], stem) >= stem_threshold:
        return True
    if contains_token_phrase(seed["short_answer"], stem, min_tokens=4):
        return True
    if base.overlap_ratio(seed["short_answer"], correct_text) >= choice_threshold and contains_token_phrase(
        seed["short_answer"], correct_text, min_tokens=5
    ):
        return True
    return False


def normalize_generated_stem(seed, payload):
    stem = trim_to_first_sentence(payload["generated_stem"])
    if seed["doc_type_name"] == "법령_QA":
        return tighten_law_stem(seed, stem)
    return normalize_stem_ending(stem, seed["problem_generation_mode"])


def build_local_fallback_problem(seed):
    # severe leakage나 stem widening이 감지되면 `r1` template fallback을 그대로 쓰되 stem만 `r2` 규칙으로 정리한다.
    payload = ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM(seed)
    if seed["doc_type_name"] == "법령_QA":
        payload["generated_stem"] = normalize_stem_ending(normalize_seed_stem_core(seed), seed["problem_generation_mode"])
    else:
        payload["generated_stem"] = normalize_generated_stem(seed, payload)
    return payload


def postprocess_problem(seed, payload):
    payload = base.validate_generated_payload(payload)
    payload["generated_stem"] = normalize_generated_stem(seed, payload)
    if seed["doc_type_name"] == "법령_QA":
        # `r2`의 핵심은 법령형 stem만 좁게 보정하는 것이므로, 우선 stem만 다시 쓰고 choice는 최대한 살린다.
        if exposes_answer_too_directly(seed, payload):
            payload["generated_stem"] = normalize_stem_ending(normalize_seed_stem_core(seed), seed["problem_generation_mode"])
            if base.overlap_ratio(seed["short_answer"], payload["generated_stem"]) >= 0.68:
                return build_local_fallback_problem(seed)
        return payload

    # 비법령 row는 `r1`이 이미 안정적이었으므로, source 질문을 거의 그대로 옮긴 stem만 `r1` stem 기준으로 접는다.
    if base.overlap_ratio(seed["short_answer"], payload["generated_stem"]) >= 0.72 and contains_token_phrase(
        seed["short_answer"], payload["generated_stem"], min_tokens=4
    ):
        reference_patch = load_reference_patch_rows().get(seed["seed_sample_id"], {})
        payload["generated_stem"] = normalize_stem_ending(
            reference_patch.get("generated_stem", normalize_seed_stem_core(seed)),
            seed["problem_generation_mode"],
        )
    return payload


def build_generation_messages(seed, reference_v2):
    system_prompt = base.load_prompt("generator_system.txt", PROMPT_DIR)
    user_template = base.load_prompt("generator_user_template.md", PROMPT_DIR)
    user_prompt = base.render_prompt(
        user_template,
        {
            "doc_type_name": seed["doc_type_name"],
            "source_subset": seed["source_subset"],
            "problem_generation_mode": seed["problem_generation_mode"],
            "doc_type_nearmiss_hint": base.DOC_TYPE_NEARMISS_HINTS[seed["doc_type_name"]],
            "transformed_problem": seed["transformed_problem"],
            "short_answer": seed["short_answer"],
            "generated_explanation": seed["generated_explanation"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
            "problem_v2_generated_stem": reference_v2.get("generated_stem", ""),
            "problem_v2_choice_a": reference_v2.get("choice_a", ""),
            "problem_v2_choice_b": reference_v2.get("choice_b", ""),
            "problem_v2_choice_c": reference_v2.get("choice_c", ""),
            "problem_v2_choice_d": reference_v2.get("choice_d", ""),
            "problem_v2_correct_choice": reference_v2.get("correct_choice", ""),
        },
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_reference_patch_rows():
    global REFERENCE_PATCH_ROWS_CACHE
    if REFERENCE_PATCH_ROWS_CACHE is not None:
        return REFERENCE_PATCH_ROWS_CACHE
    rows = base.load_csv_rows(REFERENCE_PATCH_MERGED_PATH)
    REFERENCE_PATCH_ROWS_CACHE = {row["seed_sample_id"]: row for row in rows if row.get("selected_for_seed") == "예"}
    return REFERENCE_PATCH_ROWS_CACHE


def build_side_by_side_examples(r2_rows):
    reference_v2_map = ORIGINAL_LOAD_REFERENCE_V2_ROWS()
    reference_patch_map = load_reference_patch_rows()
    selected_r2_rows = [row for row in r2_rows if row["selected_for_seed"] == "예"]
    selected_r2_rows.sort(
        key=lambda row: (
            row["final_status"] == reference_patch_map.get(row["seed_sample_id"], {}).get("final_status", row["final_status"])
            and row["final_status"] == reference_v2_map.get(row["seed_sample_id"], {}).get("final_status", row["final_status"]),
            row["doc_type_name"],
            row["seed_sample_id"],
        )
    )

    example_rows = []
    for row in selected_r2_rows[:8]:
        reference_v2 = reference_v2_map.get(row["seed_sample_id"], {})
        reference_patch = reference_patch_map.get(row["seed_sample_id"], {})
        example_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "v2_status": reference_v2.get("final_status", ""),
                "patch_status": reference_patch.get("final_status", ""),
                "r2_status": row["final_status"],
                "v2_weighted_score": reference_v2.get("weighted_score", ""),
                "patch_weighted_score": reference_patch.get("weighted_score", ""),
                "r2_weighted_score": row["weighted_score"],
                "r2_nearmiss_score": row.get("nearmiss_score", ""),
                "v2_generated_stem": reference_v2.get("generated_stem", ""),
                "patch_generated_stem": reference_patch.get("generated_stem", ""),
                "r2_generated_stem": row["generated_stem"],
                "v2_correct_choice": reference_v2.get("correct_choice", ""),
                "patch_correct_choice": reference_patch.get("correct_choice", ""),
                "r2_correct_choice": row["correct_choice"],
                "v2_choices": " | ".join(
                    [
                        f"A. {reference_v2.get('choice_a', '')}",
                        f"B. {reference_v2.get('choice_b', '')}",
                        f"C. {reference_v2.get('choice_c', '')}",
                        f"D. {reference_v2.get('choice_d', '')}",
                    ]
                ),
                "patch_choices": " | ".join(
                    [
                        f"A. {reference_patch.get('choice_a', '')}",
                        f"B. {reference_patch.get('choice_b', '')}",
                        f"C. {reference_patch.get('choice_c', '')}",
                        f"D. {reference_patch.get('choice_d', '')}",
                    ]
                ),
                "r2_choices": " | ".join(
                    [
                        f"A. {row['choice_a']}",
                        f"B. {row['choice_b']}",
                        f"C. {row['choice_c']}",
                        f"D. {row['choice_d']}",
                    ]
                ),
                "r2_error_tags": row.get("error_tags", ""),
                "r2_nearmiss_reason": row.get("nearmiss_reason", ""),
            }
        )

    markdown_blocks = [f"# side-by-side examples `{RUN_NAME}`", ""]
    for index, row in enumerate(example_rows, start=1):
        markdown_blocks.extend(
            [
                f"## example {index}: `{row['seed_sample_id']}` / `{row['doc_type_name']}`",
                "",
                "### reference `v2`",
                f"- status: `{row['v2_status']}`",
                f"- weighted_score: `{row['v2_weighted_score']}`",
                f"- stem: {row['v2_generated_stem']}",
                f"- choices: {row['v2_choices']}",
                f"- correct_choice: `{row['v2_correct_choice']}`",
                "",
                "### reference `v2_difficulty_patch`",
                f"- status: `{row['patch_status']}`",
                f"- weighted_score: `{row['patch_weighted_score']}`",
                f"- stem: {row['patch_generated_stem']}",
                f"- choices: {row['patch_choices']}",
                f"- correct_choice: `{row['patch_correct_choice']}`",
                "",
                "### `r2`",
                f"- status: `{row['r2_status']}`",
                f"- weighted_score: `{row['r2_weighted_score']}`",
                f"- nearmiss_score: `{row['r2_nearmiss_score']}`",
                f"- stem: {row['r2_generated_stem']}",
                f"- choices: {row['r2_choices']}",
                f"- correct_choice: `{row['r2_correct_choice']}`",
                f"- error_tags: `{row['r2_error_tags']}`",
                f"- nearmiss_reason: {row['r2_nearmiss_reason']}",
                "",
            ]
        )

    base.write_csv_atomic(SIDE_BY_SIDE_CSV_PATH, example_rows, list(example_rows[0].keys()) if example_rows else ["seed_sample_id"])
    base.write_text_atomic(SIDE_BY_SIDE_MD_PATH, "\n".join(markdown_blocks) + "\n")
    return example_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, side_by_side_rows):
    # 실행 메타에 `r1` comparator 경로와 `r2` focus를 같이 남겨야 다음 리뷰에서 해석이 흔들리지 않는다.
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, side_by_side_rows)
    manifest["reference_patch_merged_path"] = str(REFERENCE_PATCH_MERGED_PATH)
    manifest["seed_registry_strategy"] = "reuse_v2_16_family_comparator_seed_set_for_r2"
    manifest["r2_focus"] = [
        "법령_QA single-stem guardrail",
        "정답 누설 차단 강화",
        "stem ending 정규화",
    ]
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def main():
    configure_base()
    base.build_local_fallback_problem = build_local_fallback_problem
    base.postprocess_problem = postprocess_problem
    base.build_generation_messages = build_generation_messages
    base.build_side_by_side_examples = build_side_by_side_examples
    base.build_run_manifest = build_run_manifest
    return base.main()


if __name__ == "__main__":
    main()
