import json
import re
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 count 확대용 production batch가 아니라,
# `pb4` 이후 법령형 정답 유일성/선택지 중복 guardrail을 검증하는 targeted pilot이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb4 as pb4


VERSION_TAG = "objective_law_guardrail_targeted_pilot"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-24_233205"
RUN_PURPOSE = "law_guardrail_targeted_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = pb4.pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb4.pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb4.pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
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

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
TAIL_MEMO_MD_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.md"

REFERENCE_PB4_SEED_REGISTRY_PATH = (
    pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb4_objective_current_r2"
    / "seed_registry.csv"
)

# `법령_QA only`, lane `8/8`을 보장하기 위해 법령 source만 별도 quota로 둔다.
TARGET_LAW_SPECS = [
    {
        "source_subset": "01_TL_법령_QA",
        "domain": "01",
        "doc_type_name": "법령_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/01.*/*/1.데이터/Training/02.라벨링데이터/TL_01. 민사법_002. 법령_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/01.*/*/1.데이터/Training/01.원천데이터/TS_01. 민사법_002. 법령/*.json",
        "sample_count": 4,
    },
    {
        "source_subset": "02_TL_법령_QA",
        "domain": "02",
        "doc_type_name": "법령_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_002. 법령_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_002. 법령/*.json",
        "sample_count": 4,
    },
    {
        "source_subset": "03_TL_법령_QA",
        "domain": "03",
        "doc_type_name": "법령_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 4,
    },
    {
        "source_subset": "04_TL_법령_QA",
        "domain": "04",
        "doc_type_name": "법령_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 4,
    },
]

ORIGINAL_BUILD_RUN_MANIFEST = pb4.pb3.build_run_manifest
ORIGINAL_BUILD_GENERATION_MESSAGES = pb4.pb3.r2.build_generation_messages
ORIGINAL_POSTPROCESS_PROBLEM = pb4.pb3.r2.postprocess_problem
ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM = pb4.pb3.r2.build_local_fallback_problem


def normalized(text):
    return " ".join((text or "").split())


def has_numeric_boundary(text):
    return bool(re.search(r"\d+\s*(세|일|개월|년|회|명|원|퍼센트|%)", text)) or any(
        marker in text for marker in ["미만", "이하", "이상", "초과", "이내", "부터", "까지"]
    )


def classify_law_seed(label_input, label_output):
    # preflight에서 seed가 어떤 guardrail class에 걸렸는지 reviewer가 바로 볼 수 있게 분류한다.
    input_text = normalized(label_input)
    output_text = normalized(label_output)
    joined = f"{input_text} {output_text}"

    categories = []
    if has_numeric_boundary(joined) and any(marker in joined for marker in ["정의", "뜻", "아동", "청소년", "나이", "연령"]):
        categories.append("simple_definition_numeric_boundary")
    if any(marker in joined for marker in ["목적", "위하여", "효율적으로 추진", "설립목적"]):
        categories.append("purpose_only_recall")
    if any(marker in joined for marker in ["공단", "위원회", "협회", "기관", "장관", "청장"]) and any(
        marker in joined for marker in ["설립", "두어", "소속", "권한", "업무"]
    ):
        categories.append("institution_name_recall")
    if any(marker in joined for marker in [" 및 ", " 또는 ", "그리고", "동시에", "각 호", "요건", "경우", "대통령령"]):
        categories.append("compound_requirement")
    if any(marker in joined for marker in ["다만", "제외", "한정", "범위", "예외", "아니", "면제", "적용하지"]):
        categories.append("exception_or_scope")

    if not categories:
        categories.append("plain_law_requirement")
    return categories


def should_skip_law_seed(categories):
    # `pb4_009`류 숫자 경계 동치와 `pb4_005`류 단순회상을 pilot에서 먼저 줄이는 보수적 filter다.
    category_set = set(categories)
    if "simple_definition_numeric_boundary" in category_set:
        return True, "skip_numeric_boundary_equivalence_risk"
    if "purpose_only_recall" in category_set and not ({"compound_requirement", "exception_or_scope"} & category_set):
        return True, "skip_purpose_only_recall"
    if "institution_name_recall" in category_set and not ({"compound_requirement", "exception_or_scope"} & category_set):
        return True, "skip_institution_name_recall"
    return False, ""


def collect_excluded_rows():
    # targeted pilot은 reviewer sign-off 전 count 미합산 후보 run이므로 기존 current objective seed 전체를 보호한다.
    rows = pb4.collect_excluded_rows()
    rows.extend(pb4.pb3.load_csv_rows_if_exists(REFERENCE_PB4_SEED_REGISTRY_PATH))
    return rows


def build_exclusion_sets(rows):
    return {
        "sample_ids": {
            row.get("sample_id", "") or row.get("seed_sample_id", "")
            for row in rows
            if row.get("sample_id", "") or row.get("seed_sample_id", "")
        },
        "reference_sample_ids": {row.get("reference_sample_id", "") for row in rows if row.get("reference_sample_id", "")},
        "family_ids": {row.get("family_id", "") for row in rows if row.get("family_id", "")},
        "label_paths": {row.get("label_path", "") for row in rows if row.get("label_path", "")},
        "raw_paths": {row.get("raw_path", "") for row in rows if row.get("raw_path", "")},
    }


def select_fresh_registry_records():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)
    records = []
    sample_order = 1
    batch_family_ids = set()
    batch_label_paths = set()
    batch_raw_paths = set()

    for spec in TARGET_LAW_SPECS:
        label_paths = pb4.pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb4.pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb4.pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            last_skip_reason = ""
            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = pb4.pb3.explanation_common.normalize_label_payload(
                    candidate_label_path,
                    pb4.pb3.explanation_common.load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                categories = classify_law_seed(
                    candidate_payload["label"]["input"],
                    candidate_payload["label"]["output"],
                )
                should_skip, skip_reason = should_skip_law_seed(categories)
                last_skip_reason = skip_reason
                if should_skip:
                    continue
                passes_filter, skip_reason = pb4.passes_seed_quality_filter(
                    spec["doc_type_name"],
                    candidate_payload["label"]["input"],
                    candidate_payload["label"]["output"],
                )
                last_skip_reason = skip_reason
                if not passes_filter:
                    continue
                try:
                    candidate_raw_path = pb4.pb3.explanation_common.locate_raw_path(
                        raw_paths,
                        spec["doc_type_name"],
                        candidate_payload["info"],
                    )
                except FileNotFoundError:
                    continue
                candidate_family_id = pb4.pb3.explanation_common.make_family_id(spec["doc_type_name"], candidate_payload["info"])
                candidate_label_path_text = str(candidate_label_path)
                candidate_raw_path_text = str(candidate_raw_path)

                if candidate_family_id in exclusion_sets["family_ids"] or candidate_family_id in batch_family_ids:
                    continue
                if candidate_label_path_text in exclusion_sets["label_paths"] or candidate_label_path_text in batch_label_paths:
                    continue
                if candidate_raw_path_text in exclusion_sets["raw_paths"] or candidate_raw_path_text in batch_raw_paths:
                    continue

                chosen = (
                    candidate_index,
                    candidate_label_path,
                    candidate_payload,
                    candidate_raw_path,
                    candidate_family_id,
                    categories,
                )
                break

            if chosen is None:
                raise RuntimeError(f"{spec['source_subset']}에서 targeted law seed를 충분히 찾지 못했습니다. last_skip={last_skip_reason}")

            used_indices.add(chosen[0])
            label_path, payload, raw_path, family_id, categories = chosen[1], chosen[2], chosen[3], chosen[4], chosen[5]
            info = payload["info"]
            label = payload["label"]
            sample_id = f"law_guardrail_{sample_order:03d}"

            record = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec["sampling_lane"],
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb4.pb3.explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "law_guardrail_categories": "|".join(categories),
                "local_selection_order": local_order,
                "selected_index": chosen[0],
                "selection_note": "law guardrail targeted pilot: current objective seed와 high-risk definition/recall seed 제외 후 선택",
            }
            records.append(record)
            batch_family_ids.add(family_id)
            batch_label_paths.add(str(label_path))
            batch_raw_paths.add(str(raw_path))
            sample_order += 1

    return records, exclusion_sets


def build_seed_row(record):
    row = pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_law_guardrail_targeted_pilot_seed"
    row["selection_note"] = "pb4 법령형 정답 유일성/선택지 중복 tail을 검증하기 위한 targeted pilot seed"
    row["law_guardrail_categories"] = record["law_guardrail_categories"]
    row["law_guardrail_filter_note"] = "numeric_boundary_purpose_only_institution_name_high_risk_filtered"
    return row


def build_preflight_rows(seed_rows, exclusion_sets):
    seed_counts = Counter(row["seed_sample_id"] for row in seed_rows)
    reference_counts = Counter(row["reference_sample_id"] for row in seed_rows)
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows = []

    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "reference_sample_id": row["reference_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "law_guardrail_categories": row.get("law_guardrail_categories", ""),
                "law_guardrail_filter_note": row.get("law_guardrail_filter_note", ""),
                "seed_sample_id_duplicate_in_batch": "예" if seed_counts[row["seed_sample_id"]] > 1 else "아니오",
                "reference_sample_id_duplicate_in_batch": "예" if reference_counts[row["reference_sample_id"]] > 1 else "아니오",
                "family_duplicate_in_batch": "예" if family_counts[row["family_id"]] > 1 else "아니오",
                "label_path_duplicate_in_batch": "예" if label_counts[row["label_path"]] > 1 else "아니오",
                "raw_path_duplicate_in_batch": "예" if raw_counts[row["raw_path"]] > 1 else "아니오",
                "seed_sample_id_overlap_with_prior": "예" if row["seed_sample_id"] in exclusion_sets["sample_ids"] else "아니오",
                "reference_sample_id_overlap_with_prior": "예"
                if row["reference_sample_id"] in exclusion_sets["reference_sample_ids"]
                else "아니오",
                "family_overlap_with_prior": "예" if row["family_id"] in exclusion_sets["family_ids"] else "아니오",
                "label_path_overlap_with_prior": "예" if row["label_path"] in exclusion_sets["label_paths"] else "아니오",
                "raw_path_overlap_with_prior": "예" if row["raw_path"] in exclusion_sets["raw_paths"] else "아니오",
                "answer_mode": row["answer_mode"],
                "problem_generation_mode": row["problem_generation_mode"],
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )

    return preflight_rows


def assert_preflight(seed_rows, preflight_rows):
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    if len(seed_rows) != 16:
        raise RuntimeError(f"law guardrail seed 수가 16개가 아닙니다: {len(seed_rows)}")
    if doc_type_counts != {"법령_QA": 16}:
        raise RuntimeError(f"법령_QA only 조건 실패: {dict(doc_type_counts)}")
    if lane_counts.get("generalization_03_04", 0) != 8 or lane_counts.get("expansion_01_02", 0) != 8:
        raise RuntimeError(f"lane 8/8 조건 실패: {dict(lane_counts)}")

    for row in preflight_rows:
        overlap_flags = [
            row["seed_sample_id_duplicate_in_batch"],
            row["reference_sample_id_duplicate_in_batch"],
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
            row["seed_sample_id_overlap_with_prior"],
            row["reference_sample_id_overlap_with_prior"],
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        if "예" in overlap_flags:
            raise RuntimeError(f"law guardrail seed preflight 중복/누수 실패: {row['seed_sample_id']}")


def write_preflight_report(seed_rows, preflight_rows):
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    category_counts = Counter()
    for row in seed_rows:
        category_counts.update(row.get("law_guardrail_categories", "").split("|"))

    pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        "- doc_type_counts: `{'법령_QA': 16}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        "",
        "## law guardrail category counts",
        "| category | count |",
        "| --- | ---: |",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | `{count}` |")
    lines.extend(["", "## source subset counts", "| source_subset | count |", "| --- | ---: |"])
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    lines.extend(
        [
            "| total seed count is 16 | `pass` |",
            "| doc type is 법령_QA only | `pass` |",
            "| lane split is 8/8 | `pass` |",
            "| no batch seed_sample_id/reference_sample_id duplicate | `pass` |",
            "| no batch family_id/label_path/raw_path duplicate | `pass` |",
            "| no prior current/held-out/audit/tail overlap | `pass` |",
            "| law guardrail categories recorded | `pass` |",
            "| high-risk numeric boundary / purpose-only / institution-only seeds filtered | `pass` |",
        ]
    )
    pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry():
    pb4.pb3.base.ensure_dirs(
        INTERIM_DIR,
        PROCESSED_DIR,
        RUN_DIR,
        RUN_PROMPTS_DIR,
        RUN_INPUTS_DIR,
        RUN_GENERATIONS_DIR,
        RUN_JUDGE_LOGS_DIR,
        RUN_MERGED_DIR,
        RUN_EXPORTS_DIR,
    )
    records, exclusion_sets = select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["sampling_lane"], row["source_subset"], row["seed_sample_id"]))
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_preflight(seed_rows, preflight_rows)

    pb4.pb3.base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def build_generation_messages(seed, reference_v2):
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    # API prompt에 법령형 정답 유일성 guardrail을 추가해 seed filter와 generation이 같은 의도를 갖게 한다.
    messages[1]["content"] += """

## law guardrail targeted pilot 추가 지시
- 이번 run은 `법령_QA` 정답 유일성·선택지 중복 재발 방지 검증이다.
- 숫자·기간·나이 표현이 실질적으로 같은 범위라면 정답과 오답으로 동시에 두지 말 것.
- 정답이 복합 요건이면 오답을 `A만`, `B만` 같은 부분정답형으로 두지 말고, 법적으로 명확히 틀린 축을 하나만 비틀 것.
- 목적·기관명·정의만 맞히는 단순회상형 stem을 피하고, 요건·효과·적용 범위·예외 중 하나를 비교하게 만들 것.
- 오답끼리도 의미가 중복되지 않게 각각 다른 축을 비틀 것.
"""
    return messages


def choices_are_duplicate_or_equivalent(payload):
    # strict generation 이후에도 같은 표현이나 18세 이하/19세 미만류 동치 표현은 local guardrail로 차단한다.
    choices = [normalized(payload[f"choice_{label.lower()}"]) for label in pb4.pb3.r2.base.CHOICE_LABELS]
    if len(set(choices)) != len(choices):
        return True
    joined = " || ".join(choices)
    equivalence_pairs = [
        ("19세 미만", "18세 이하"),
        ("20세 미만", "19세 이하"),
        ("30일 이내", "1개월 이내"),
    ]
    return any(left in joined and right in joined for left, right in equivalence_pairs)


def build_local_fallback_problem(seed):
    payload = ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM(seed)
    payload["near_miss_notes"] = payload.get("near_miss_notes", "") + " / law guardrail fallback: equivalence and partial-answer risk blocked"
    return payload


def postprocess_problem(seed, payload):
    payload = ORIGINAL_POSTPROCESS_PROBLEM(seed, payload)
    if choices_are_duplicate_or_equivalent(payload):
        return build_local_fallback_problem(seed)
    return payload


def build_batch_summary(pilot_rows):
    selected_rows = [row for row in pilot_rows if row["selected_for_seed"] == "예"]
    selected_rows.sort(key=lambda row: (row.get("sampling_lane", ""), row["seed_sample_id"]))
    summary = pb4.pb3.summarize_rows(pilot_rows)
    category_by_seed = {
        row["seed_sample_id"]: row.get("law_guardrail_categories", "")
        for row in pb4.pb3.base.load_jsonl(SEED_READY_PATH)
    }
    # 정답 비유일/선택지 중복 재발 여부는 이번 pilot의 핵심 성공 기준이라 summary에도 직접 남긴다.
    unique_answer_recurrence_count = sum(
        1
        for row in selected_rows
        if any(tag in row.get("error_tags", "") for tag in ["정답 비유일", "오답이 정답 가능", "선택지 중복"])
    )

    summary_rows = [
        {
            "doc_type_name": "법령_QA",
            "planned_seed_count": "16",
            "train_eligible_count": str(summary["doc_type_train_counter"].get("법령_QA", 0)),
            "audit_required_count": str(summary["doc_type_audit_counter"].get("법령_QA", 0)),
            "hard_fail_count": str(summary["doc_type_hard_fail_counter"].get("법령_QA", 0)),
            "soft_fail_count": str(summary["doc_type_soft_fail_counter"].get("법령_QA", 0)),
        }
    ]
    lane_rows = []
    for sampling_lane, planned_count in sorted(summary["lane_planned_counter"].items()):
        lane_rows.append(
            {
                "sampling_lane": sampling_lane,
                "planned_seed_count": str(planned_count),
                "train_eligible_count": str(summary["lane_train_counter"].get(sampling_lane, 0)),
                "audit_required_count": str(summary["lane_audit_counter"].get(sampling_lane, 0)),
                "hard_fail_count": str(summary["lane_hard_fail_counter"].get(sampling_lane, 0)),
                "soft_fail_count": str(summary["lane_soft_fail_counter"].get(sampling_lane, 0)),
            }
        )

    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected_rows)}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        "",
        "## success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| usable | `>= 14` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `0` | `{summary['selected_hard_fail_count']}` |",
        f"| audit | `<= 1` | `{summary['selected_audit_required_count']}` |",
        f"| soft_fail | `<= 1` | `{summary['selected_soft_fail_count']}` |",
        f"| 정답 비유일·선택지 중복 재발 | `0` | `{unique_answer_recurrence_count}` |",
        "",
        "## lane yield",
        "| sampling_lane | planned | train_eligible | audit | hard_fail | soft_fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in lane_rows:
        lines.append(
            f"| `{row['sampling_lane']}` | `{row['planned_seed_count']}` | `{row['train_eligible_count']}` | `{row['audit_required_count']}` | `{row['hard_fail_count']}` | `{row['soft_fail_count']}` |"
        )
    lines.extend(
        [
            "",
            "## row status",
            "| seed_sample_id | lane | category | final_status | train_eligible | audit_required | error_tags |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in selected_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row.get('sampling_lane', '')}` | `{category_by_seed.get(row['seed_sample_id'], '')}` | `{row['final_status']}` | `{row.get('train_eligible', '')}` | `{row.get('audit_required', '')}` | `{row.get('error_tags', '')}` |"
        )

    pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, list(summary_rows[0].keys()))
    pb4.pb3.base.write_csv_atomic(BATCH_LANE_SUMMARY_CSV_PATH, lane_rows, list(lane_rows[0].keys()))
    pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_tail_memo(merged_rows):
    seed_rows = {row["seed_sample_id"]: row for row in pb4.pb3.base.load_jsonl(SEED_READY_PATH)}
    tail_rows = []
    for row in merged_rows:
        if row.get("selected_for_seed") != "예" or row.get("train_eligible") == "예":
            continue
        seed = seed_rows.get(row["seed_sample_id"], {})
        recurrence = "정답 비유일/선택지 중복 재발" if any(tag in row.get("error_tags", "") for tag in ["정답 비유일", "오답이 정답 가능", "선택지 중복"]) else "기타 tail"
        tail_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "sampling_lane": row.get("sampling_lane", ""),
                "law_guardrail_categories": seed.get("law_guardrail_categories", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "recurrence_class": recurrence,
                "generated_stem": row.get("generated_stem", ""),
            }
        )

    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "sampling_lane": "",
                "law_guardrail_categories": "",
                "final_status": "",
                "audit_required": "",
                "error_tags": "",
                "recurrence_class": "tail 없음",
                "generated_stem": "",
            }
        ]

    pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | lane | category | status | audit | error_tags | recurrence_class |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['sampling_lane']}` | `{row['law_guardrail_categories']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['error_tags']}` | `{row['recurrence_class']}` |"
        )
    pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    tail_rows = build_tail_memo(merged_rows)
    summary = pb4.pb3.summarize_rows(merged_rows)
    recurrence_count = sum(1 for row in tail_rows if row.get("recurrence_class") == "정답 비유일/선택지 중복 재발")
    manifest.update(
        {
            "version_tag": VERSION_TAG,
            "run_name": RUN_NAME,
            "seed_registry_strategy": "law_qa_only_16_seed_guardrail_targeted_pilot_excluding_current_objective_seed",
            "candidate_recipe_source": "v2_difficulty_patch_r2_plus_law_guardrail_candidate",
            "count_reflection": "reviewer_signoff_before_current_count_excluded",
            "success_criteria": {
                "usable_min": 14,
                "hard_fail_max": 0,
                "audit_max": 1,
                "soft_fail_max": 1,
                "unique_answer_recurrence_max": 0,
            },
            "success_result": {
                "usable": summary["selected_train_eligible_count"],
                "hard_fail": summary["selected_hard_fail_count"],
                "audit": summary["selected_audit_required_count"],
                "soft_fail": summary["selected_soft_fail_count"],
                "unique_answer_recurrence": recurrence_count,
            },
            "tail_memo_csv_path": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md_path": str(TAIL_MEMO_MD_PATH),
        }
    )
    pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_targeted_pilot_paths():
    # `pb4`와 같은 검증된 main flow를 쓰되, 모든 path와 prompt/judge output만 pilot 전용으로 재배선한다.
    pb4.configure_pb4_paths()
    pb4.pb3.VERSION_TAG = VERSION_TAG
    pb4.pb3.RUN_PURPOSE = RUN_PURPOSE
    pb4.pb3.RUN_NAME = RUN_NAME
    pb4.pb3.INTERIM_DIR = INTERIM_DIR
    pb4.pb3.PROCESSED_DIR = PROCESSED_DIR
    pb4.pb3.RUN_DIR = RUN_DIR
    pb4.pb3.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb4.pb3.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb4.pb3.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb4.pb3.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb4.pb3.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb4.pb3.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb4.pb3.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb4.pb3.SEED_READY_PATH = SEED_READY_PATH
    pb4.pb3.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb4.pb3.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb4.pb3.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb4.pb3.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb4.pb3.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb4.pb3.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb4.pb3.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb4.pb3.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb4.pb3.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb4.pb3.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb4.pb3.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb4.pb3.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb4.pb3.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb4.pb3.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb4.pb3.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb4.pb3.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb4.pb3.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH


def main():
    configure_targeted_pilot_paths()
    pb4.pb3.select_fresh_registry_records = select_fresh_registry_records
    pb4.pb3.build_seed_row = build_seed_row
    pb4.pb3.build_preflight_rows = build_preflight_rows
    pb4.pb3.assert_preflight = assert_preflight
    pb4.pb3.write_preflight_report = write_preflight_report
    pb4.pb3.build_seed_registry = build_seed_registry
    pb4.pb3.build_batch_summary = build_batch_summary
    pb4.pb3.build_run_manifest = build_run_manifest
    pb4.pb3.r2.build_generation_messages = build_generation_messages
    pb4.pb3.r2.postprocess_problem = postprocess_problem
    pb4.pb3.r2.build_local_fallback_problem = build_local_fallback_problem
    return pb4.pb3.main()


if __name__ == "__main__":
    main()
