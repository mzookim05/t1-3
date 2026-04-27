from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# 서술형은 기존 `v3` split-descriptive recipe를 유지하되, 마감 운영을 위해
# fresh seed를 먼저 쓰고 부족분만 train-only split-lock 재사용으로 보충한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import run_descriptive_pb1 as base  # noqa: E402
from scripts.aihub.problem_generation.shared.production_batch_common import write_text_atomic  # noqa: E402


EXPLANATION_DIR = base.PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "explanation_generation"
if str(EXPLANATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLANATION_DIR))

import common as explanation_common  # noqa: E402
from extract_evidence_cards import build_card  # noqa: E402
from generate_explanations import build_local_fallback_explanation, postprocess_generated_explanation  # noqa: E402
from settings import DATASET_SPECS  # noqa: E402
from transform_problems import build_transformed_sample  # noqa: E402


# reviewer stop line마다 같은 runner를 재사용하되, 이전 counted wave의 processed 산출물을 덮어쓰지 않도록
# 실행 단위 식별자는 환경변수로 바꿀 수 있게 둔다.
VERSION_TAG = os.environ.get("DESCRIPTIVE_WAVE_VERSION_TAG", "descriptive_inventory_linter_pointer_sync_medium_primary")
RUN_DATE = build_run_stamp()
RUN_PURPOSE = os.environ.get("DESCRIPTIVE_WAVE_RUN_PURPOSE", "descriptive_v3_availability_map_medium_primary_api_execution")
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
ROUTE_NAME = os.environ.get(
    "DESCRIPTIVE_WAVE_ROUTE_NAME",
    "descriptive_inventory_linter_pointer_sync_plus_availability_map_medium_primary_api_execution",
)

PROJECT_ROOT = base.PROJECT_ROOT
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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
EXCLUSION_AUDIT_CSV_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.csv"
AVAILABILITY_MAP_CSV_PATH = RUN_EXPORTS_DIR / f"seed_availability_map_{VERSION_TAG}.csv"
AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH = RUN_EXPORTS_DIR / f"route_feasibility_{VERSION_TAG}.csv"
AVAILABILITY_MAP_MD_PATH = RUN_EXPORTS_DIR / f"seed_availability_map_{VERSION_TAG}.md"
PACKAGE_SPEC_MD_PATH = RUN_EXPORTS_DIR / f"package_spec_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWERABILITY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answerability_{VERSION_TAG}.jsonl"
TASKFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_taskfit_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
CANDIDATE_POOL_PATH = RUN_DIR / "candidate_pool.csv"
ACCEPTED_POOL_PATH = RUN_DIR / "accepted_pool.csv"
REJECTED_POOL_PATH = RUN_DIR / "rejected_pool.csv"
TAIL_TAXONOMY_PATH = RUN_DIR / "tail_taxonomy.csv"
QUOTA_SURPLUS_POOL_PATH = RUN_DIR / "quota_surplus_pool.csv"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
EVIDENCE_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"evidence_summary_{VERSION_TAG}.md"
COMPILER_MANIFEST_PATH = RUN_DIR / "compiler_manifest.json"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

PACKAGE_ROLE = "count_reflection_candidate_package"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
COUNT_DISPOSITION = "candidate_not_counted"
PROMOTION_CONTRACT_STATUS = "reviewer_signoff_needed"
COUNTED_PACKAGE_ROLE = "counted_current_production"
COUNTED_BATCH_STATUS = "counted_current_production"
COUNTED_REFLECTION_STATUS = "counted"
COUNTED_DISPOSITION = "counted"
COUNTED_PROMOTION_CONTRACT_STATUS = "counted_under_api_first_contract"
YES = "예"
NO = "아니오"


def candidate_contract_fields() -> dict[str, str]:
    # candidate/tail/surplus layer는 final package 밖이므로 downstream과 count를 계속 막는다.
    return {
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
    }


def counted_contract_fields() -> dict[str, str]:
    # API-first reviewer contract에서는 strict final package만 counted state로 승격하고, 후보 pool은 그대로 미합산 보존한다.
    return {
        "package_role": COUNTED_PACKAGE_ROLE,
        "batch_status": COUNTED_BATCH_STATUS,
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "downstream_consumption_allowed": YES,
        "count_allowed": YES,
        "count_disposition": COUNTED_DISPOSITION,
        "promotion_contract_status": COUNTED_PROMOTION_CONTRACT_STATUS,
    }

EXPLANATION_GENERATION_VARIANT = {
    "name": "without_long_answer",
    "label": "long_answer 제외",
    "include_long_answer": False,
}

PRIMARY_SOURCE_COUNTS = {
    "01_TL_법령_QA": 3,
    "02_TL_법령_QA": 3,
    "03_TL_법령_QA": 4,
    "04_TL_법령_QA": 4,
    "01_TL_유권해석_QA": 3,
    "02_TL_유권해석_QA": 3,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 4,
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 2,
    "02_TL_심결문_QA": 2,
    "03_TL_결정례_QA": 4,
    "04_TL_결정례_QA": 3,
    "01_TL_판결문_QA": 3,
    "02_TL_판결문_QA": 3,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 4,
}
PRIMARY_FINAL_SOURCE_COUNTS = {
    "01_TL_법령_QA": 2,
    "02_TL_법령_QA": 2,
    "03_TL_법령_QA": 3,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 2,
    "02_TL_유권해석_QA": 2,
    "03_TL_해석례_QA": 3,
    "04_TL_해석례_QA": 3,
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 3,
    "01_TL_판결문_QA": 2,
    "02_TL_판결문_QA": 2,
    "03_TL_판결문_QA": 3,
    "04_TL_판결문_QA": 3,
}
MEDIUM_RELAXED_SOURCE_COUNTS = {
    # `01_TL_법령_QA` depleted source를 같은 법령 lane의 `02/03/04_TL_법령_QA`로 재배정해
    # doc_type 총량은 유지하고 source-local 병목만 완화한다.
    **PRIMARY_SOURCE_COUNTS,
    "01_TL_법령_QA": 0,
    "02_TL_법령_QA": 4,
    "03_TL_법령_QA": 5,
    "04_TL_법령_QA": 5,
}
MEDIUM_RELAXED_FINAL_SOURCE_COUNTS = {
    # final package도 법령형 `01` quota만 제외하고 `02/03/04`가 총 10건을 채우도록 잠근다.
    **PRIMARY_FINAL_SOURCE_COUNTS,
    "01_TL_법령_QA": 0,
    "02_TL_법령_QA": 3,
    "03_TL_법령_QA": 3,
    "04_TL_법령_QA": 4,
}
FALLBACK_SOURCE_COUNTS = {
    "01_TL_법령_QA": 2,
    "02_TL_법령_QA": 2,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 2,
    "02_TL_유권해석_QA": 2,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 3,
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 2,
    "01_TL_판결문_QA": 2,
    "02_TL_판결문_QA": 2,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 3,
}
FALLBACK_FINAL_SOURCE_COUNTS = {
    "01_TL_법령_QA": 1,
    "02_TL_법령_QA": 1,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 2,
    "01_TL_유권해석_QA": 1,
    "02_TL_유권해석_QA": 1,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 2,
    "01_TL_심결례_QA": 1,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 2,
    "04_TL_결정례_QA": 1,
    "01_TL_판결문_QA": 1,
    "02_TL_판결문_QA": 1,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 2,
}
CONSTRAINED_SOURCE_COUNTS = {
    # `01_TL_법령_QA`는 현재 exclusion/split-lock gate를 통과한 fresh candidate가 없어 constrained route에서 제외한다.
    "02_TL_법령_QA": 2,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 2,
    "02_TL_유권해석_QA": 2,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 3,
    "01_TL_심결례_QA": 2,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 2,
    "01_TL_판결문_QA": 2,
    "02_TL_판결문_QA": 2,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 3,
}
CONSTRAINED_PRIMARY_FINAL_SOURCE_COUNTS = {
    # primary final 24는 빠진 `01_TL_법령_QA` 몫을 같은 법령 lane의 `04_TL_법령_QA`로 보충한다.
    "02_TL_법령_QA": 1,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 3,
    "01_TL_유권해석_QA": 1,
    "02_TL_유권해석_QA": 1,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 2,
    "01_TL_심결례_QA": 1,
    "02_TL_심결례_QA": 1,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 2,
    "04_TL_결정례_QA": 1,
    "01_TL_판결문_QA": 1,
    "02_TL_판결문_QA": 1,
    "03_TL_판결문_QA": 2,
    "04_TL_판결문_QA": 2,
}
CONSTRAINED_FALLBACK_FINAL_SOURCE_COUNTS = {
    # fallback final 20은 source 균형보다 global eval holdout과 strict quality gate 통과를 우선한다.
    "02_TL_법령_QA": 1,
    "03_TL_법령_QA": 2,
    "04_TL_법령_QA": 1,
    "01_TL_유권해석_QA": 1,
    "02_TL_유권해석_QA": 1,
    "03_TL_해석례_QA": 2,
    "04_TL_해석례_QA": 2,
    "01_TL_심결례_QA": 1,
    "02_TL_심결례_QA": 1,
    # 이번 constrained pool에서는 `02_TL_심결문_QA`가 strict accepted pool에 남지 않아 fallback quota에서 제외한다.
    "02_TL_심결문_QA": 0,
    "03_TL_결정례_QA": 2,
    "04_TL_결정례_QA": 2,
    "01_TL_판결문_QA": 1,
    "02_TL_판결문_QA": 1,
    "03_TL_판결문_QA": 1,
    "04_TL_판결문_QA": 1,
}


def scaled_source_counts(base_counts: dict[str, int], target_total: int) -> dict[str, int]:
    # candidate64처럼 reviewer가 요청한 확대 route는 기존 source-relaxed 비율을 유지한 채
    # 정수 quota로 재배분해 exact runner preflight가 가능하게 만든다.
    current_total = sum(base_counts.values())
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source_subset, count in base_counts.items():
        scaled = count * target_total / current_total
        floors[source_subset] = int(scaled)
        remainders.append((scaled - int(scaled), source_subset))
    missing = target_total - sum(floors.values())
    for _, source_subset in sorted(remainders, reverse=True)[:missing]:
        floors[source_subset] += 1
    return floors


def move_source_quota(counts: dict[str, int], source_subset: str, destination_sources: list[str]) -> dict[str, int]:
    # post-154712에서는 특정 depleted source를 계속 요구하면 preflight가 막히므로,
    # 같은 doc_type 안의 살아 있는 source로 quota만 옮겨 leakage guard와 문서유형 총량을 함께 지킨다.
    moved_count = counts.get(source_subset, 0)
    counts[source_subset] = 0
    if not destination_sources:
        return counts
    for index in range(moved_count):
        counts[destination_sources[index % len(destination_sources)]] = counts.get(
            destination_sources[index % len(destination_sources)],
            0,
        ) + 1
    return counts


def cap_source_quota(counts: dict[str, int], source_subset: str, cap: int, destination_sources: list[str]) -> dict[str, int]:
    # source-local availability가 1건만 부족해도 대형 route 전체가 막히므로,
    # depleted 직전 source는 cap을 걸고 같은 해석례 계열 source로 초과 quota를 옮긴다.
    current_count = counts.get(source_subset, 0)
    excess_count = max(0, current_count - cap)
    counts[source_subset] = min(current_count, cap)
    if not destination_sources:
        return counts
    for index in range(excess_count):
        counts[destination_sources[index % len(destination_sources)]] = counts.get(
            destination_sources[index % len(destination_sources)],
            0,
        ) + 1
    return counts


def source_reallocated_counts(base_counts: dict[str, int], target_total: int, *, cap_interpretation_04: bool) -> dict[str, int]:
    # reviewer 회신의 P2를 route로 코드화한다:
    # `02_TL_법령_QA`는 post-154712 기준 available 0이므로 요구하지 않고,
    # `03/04_TL_법령_QA`가 같은 법령 doc_type 안에서 quota를 흡수한다.
    counts = scaled_source_counts(base_counts, target_total)
    counts = move_source_quota(counts, "02_TL_법령_QA", ["03_TL_법령_QA", "04_TL_법령_QA"])
    if cap_interpretation_04 and target_total >= 128:
        counts = cap_source_quota(counts, "04_TL_해석례_QA", 8, ["03_TL_해석례_QA", "01_TL_유권해석_QA"])
    return counts


def with_eval_targeting(route: dict) -> dict:
    # source-reallocated repeat은 총량은 확보했지만 법령/결정례 eval 부족을 남겼다.
    # 다음 route는 같은 source schedule 위에서 split assignment만 eval-targeted로 잠근다.
    return {
        **route,
        "eval_targeting_contract": "법령_QA/결정례_QA dev-test 우선 배정, 해석례_QA eval 추가 최소화",
        "eval_balanced_doc_types": ["법령_QA", "결정례_QA"],
        "eval_priority_doc_types": ["법령_QA", "결정례_QA", "판결문_QA"],
        "eval_avoid_doc_types": ["해석례_QA"],
    }


def with_deficit_closure(route: dict) -> dict:
    # post-174728부터는 서술형 eval 부족이 2건뿐이라 기본 8개 eval split을 반복하면 과잉 eval이 생긴다.
    # deficit-closure route는 strict final gate를 유지하되, split만 train-heavy로 고정해 aggregate shortage를 직접 닫는다.
    target_count = int(route["target_count"])
    split_overrides = {
        40: (38, 1, 1),
        24: (22, 1, 1),
        12: (10, 1, 1),
    }
    train_count, dev_count, test_count = split_overrides.get(target_count, final_split_counts(target_count))
    return {
        **with_eval_targeting(route),
        "final_train_count": train_count,
        "final_dev_count": dev_count,
        "final_test_count": test_count,
        "deficit_closure_contract": "남은 서술형 shortage를 train-heavy split으로 직접 보강",
        "eval_targeting_contract": "aggregate eval target closure 우선, dev/test는 법령_QA/결정례_QA 우선 배정",
    }


def with_train_micro_closure(route: dict) -> dict:
    # `181810` 이후 서술형 eval은 이미 닫혔고 train만 5건 남았으므로,
    # micro-closure route는 평가 split을 만들지 않고 train-only final package로 잠근다.
    return {
        **route,
        "final_train_count": int(route["target_count"]),
        "final_dev_count": 0,
        "final_test_count": 0,
        "deficit_closure_contract": "남은 서술형 train shortage 5건만 직접 보강",
        "eval_targeting_contract": "eval target은 이미 닫혔으므로 dev/test 추가 생성 없음",
    }


def with_type_balance_extension(route: dict) -> dict:
    # aggregate count는 이미 닫혔지만, 사용자가 유형별 균형까지 요구한 경우에는
    # 법령/결정례/판결문 eval 부족과 해석례 train 부족을 한 번에 줄이는 reporting-balance route를 연다.
    return {
        **route,
        "final_train_count": 18,
        "final_dev_count": 11,
        "final_test_count": 11,
        "eval_targeting_contract": "descriptive type-balance extension: 법령_QA eval 9, 결정례_QA eval 10, 판결문_QA eval 1 우선 배정 뒤 남은 eval 2건은 weighted fallback으로 배정",
        "eval_target_doc_counts": {
            "법령_QA": 9,
            # 일부 결정례 accepted row는 Tier 2 train-lock이라 eval로 보낼 수 없어, strict target을 available 10건으로 낮춘다.
            "결정례_QA": 10,
            "판결문_QA": 1,
        },
        "eval_priority_doc_types": ["결정례_QA", "법령_QA", "판결문_QA", "해석례_QA"],
        "eval_avoid_doc_types": ["해석례_QA"],
        "deficit_closure_contract": "aggregate 초과를 감수하되 유형별 train/eval imbalance를 줄이는 reporting-balance extension",
    }


def with_type_deficit_overclosure(route: dict, *, train_count: int, dev_count: int, test_count: int) -> dict:
    # reviewer가 요구한 type-deficit overclosure는 aggregate target이 이미 닫힌 뒤에도
    # 해석례 train과 결정례 eval 균형을 함께 보강하는 final-only strict package로 분리한다.
    return {
        **route,
        "final_train_count": train_count,
        "final_dev_count": dev_count,
        "final_test_count": test_count,
        "eval_targeting_contract": "descriptive type-deficit overclosure: 해석례_QA train add-on과 결정례_QA dev/test add-on을 한 package로 조립",
        "eval_target_doc_counts": {
            "결정례_QA": dev_count + test_count,
        },
        "eval_priority_doc_types": ["결정례_QA"],
        "eval_avoid_doc_types": ["해석례_QA"],
        "deficit_closure_contract": "aggregate 초과를 감수하되 유형별 reporting shortage를 줄이는 type-deficit overclosure route",
    }


def final_split_counts(final_target_count: int) -> tuple[int, int, int]:
    # emergency wave는 final 80/64처럼 기존 40보다 커질 수 있으므로,
    # reviewer가 지정한 train/dev/test 비율을 route target에서 직접 계산한다.
    if final_target_count >= 80:
        return final_target_count - 16, 8, 8
    if final_target_count >= 64:
        return final_target_count - 12, 6, 6
    if final_target_count >= 40:
        return final_target_count - 8, 4, 4
    if final_target_count >= 20:
        return final_target_count - 4, 2, 2
    if final_target_count >= 10:
        return final_target_count - 2, 1, 1
    return final_target_count, 0, 0


def route_with_split(route: dict) -> dict:
    # route config에 split quota를 함께 저장해 preflight/manifest/evidence가 같은 기준을 보게 한다.
    train_count, dev_count, test_count = final_split_counts(int(route["target_count"]))
    route = {
        **route,
        "final_train_count": train_count,
        "final_dev_count": dev_count,
        "final_test_count": test_count,
    }
    if route.get("fallback_target_count"):
        fallback_train, fallback_dev, fallback_test = final_split_counts(int(route["fallback_target_count"]))
        route.update(
            {
                "fallback_final_train_count": fallback_train,
                "fallback_final_dev_count": fallback_dev,
                "fallback_final_test_count": fallback_test,
            }
        )
    return route


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_base_paths() -> None:
    # 기존 `run_descriptive_pb1`의 generator/Judge 구현을 재사용하되, 모든 산출물은 wave 전용 path에만 쓴다.
    base.VERSION_TAG = VERSION_TAG
    base.RUN_DATE = RUN_DATE
    base.RUN_PURPOSE = RUN_PURPOSE
    base.RUN_NAME = RUN_NAME
    base.INTERIM_DIR = INTERIM_DIR
    base.PROCESSED_DIR = PROCESSED_DIR
    base.RUN_DIR = RUN_DIR
    base.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    base.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    base.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    base.RUN_MERGED_DIR = RUN_MERGED_DIR
    base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    base.SEED_READY_PATH = SEED_READY_PATH
    base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    base.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    base.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    base.ANSWERABILITY_LOG_PATH = ANSWERABILITY_LOG_PATH
    base.TASKFIT_LOG_PATH = TASKFIT_LOG_PATH
    base.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    base.ROLE_TO_LOG_PATH = {
        "Grounding": GROUNDING_LOG_PATH,
        "Answerability": ANSWERABILITY_LOG_PATH,
        "TaskFit": TASKFIT_LOG_PATH,
    }
    # 마감 대응 emergency route에서는 Judge 동시성을 환경변수로 올리되,
    # 코드 기본값은 기존 안전값을 유지한다.
    # generation도 같은 방식으로 열어 두어 `candidate128`급 repeat의 wall time 병목을 줄인다.
    base.GENERATOR_MAIN_MAX_WORKERS = int(
        os.environ.get("DESCRIPTIVE_WAVE_GENERATION_MAIN_MAX_WORKERS", base.GENERATOR_MAIN_MAX_WORKERS)
    )
    base.GENERATOR_STRICT_MAX_WORKERS = int(
        os.environ.get("DESCRIPTIVE_WAVE_GENERATION_STRICT_MAX_WORKERS", base.GENERATOR_STRICT_MAX_WORKERS)
    )
    base.GENERATOR_MAIN_CHECKPOINT_EVERY = int(
        os.environ.get("DESCRIPTIVE_WAVE_GENERATION_MAIN_CHECKPOINT_EVERY", base.GENERATOR_MAIN_CHECKPOINT_EVERY)
    )
    base.JUDGE_MAIN_MAX_WORKERS = int(os.environ.get("DESCRIPTIVE_WAVE_JUDGE_MAIN_MAX_WORKERS", base.JUDGE_MAIN_MAX_WORKERS))
    base.JUDGE_MAIN_MAX_ATTEMPTS = int(os.environ.get("DESCRIPTIVE_WAVE_JUDGE_MAIN_MAX_ATTEMPTS", base.JUDGE_MAIN_MAX_ATTEMPTS))
    base.JUDGE_MAIN_RETRY_BASE_SECONDS = int(
        os.environ.get("DESCRIPTIVE_WAVE_JUDGE_MAIN_RETRY_BASE_SECONDS", base.JUDGE_MAIN_RETRY_BASE_SECONDS)
    )
    base.JUDGE_MAIN_CHECKPOINT_EVERY = int(
        os.environ.get("DESCRIPTIVE_WAVE_JUDGE_MAIN_CHECKPOINT_EVERY", base.JUDGE_MAIN_CHECKPOINT_EVERY)
    )


def row_value(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value:
        return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get(field, "") or "")
    return ""


def collect_exclusion_rows() -> list[dict]:
    # seed reuse와 count leakage를 막기 위해 counted output뿐 아니라 prior candidate registry까지 함께 제외한다.
    rows: list[dict] = []
    for path in (PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation").glob("**/seed_registry.csv"):
        if path == SEED_REGISTRY_PATH:
            continue
        rows.extend(read_csv_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation").glob("**/*.csv"):
        rows.extend(read_csv_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation").glob("**/*.jsonl"):
        rows.extend(read_jsonl_rows(path))
    for path in (PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final").glob("*"):
        if path.suffix == ".csv":
            rows.extend(read_csv_rows(path))
        elif path.suffix == ".jsonl":
            rows.extend(read_jsonl_rows(path))
    return rows


def build_exclusion_sets(rows: list[dict]) -> dict[str, set[str]]:
    exclusion_sets = {
        field: {row_value(row, field) for row in rows if row_value(row, field)}
        for field in ["seed_sample_id", "reference_sample_id", "sample_id", "family_id", "label_path", "raw_path"]
    }
    # split-lock 정책은 source reuse와 eval leakage를 분리한다. counted objective train family만 train split으로 재사용 가능하다.
    for field in [
        "split_lock_train_family_id",
        "split_lock_train_label_path",
        "split_lock_train_raw_path",
        "global_eval_family_id",
        "global_eval_label_path",
        "global_eval_raw_path",
        "quality_tail_family_id",
        "quality_tail_label_path",
        "quality_tail_raw_path",
    ]:
        exclusion_sets[field] = set()
    for row in rows:
        task_type = row_value(row, "problem_task_type")
        split = row_value(row, "split") or row_value(row, "dataset_disposition")
        counted = row_value(row, "count_reflection_status") == "counted" or row_value(row, "count_allowed") == YES
        family_id = row_value(row, "family_id")
        label_path = row_value(row, "label_path")
        raw_path = row_value(row, "raw_path")
        values = {
            "family_id": family_id,
            "label_path": label_path,
            "raw_path": raw_path,
        }
        if task_type == "objective_single_best" and counted and split == "train":
            for field, value in values.items():
                if value:
                    exclusion_sets[f"split_lock_train_{field}"].add(value)
        if counted and split in {"dev", "test"}:
            for field, value in values.items():
                if value:
                    exclusion_sets[f"global_eval_{field}"].add(value)
        if row_value(row, "quality_failure") == YES or row_value(row, "audit_required") == YES or row_value(row, "final_status") in {"hard_fail", "soft_fail"}:
            for field, value in values.items():
                if value:
                    exclusion_sets[f"quality_tail_{field}"].add(value)
    return exclusion_sets


def classify_reuse_policy(family_id: str, label_path: str, raw_path: str, exclusion_sets: dict[str, set[str]]) -> tuple[dict[str, str] | None, str]:
    # P1 계열은 train-only 재사용보다 항상 우선한다.
    if (
        family_id in exclusion_sets["global_eval_family_id"]
        or label_path in exclusion_sets["global_eval_label_path"]
        or raw_path in exclusion_sets["global_eval_raw_path"]
    ):
        return None, "global_eval_holdout_overlap"
    if (
        family_id in exclusion_sets["quality_tail_family_id"]
        or label_path in exclusion_sets["quality_tail_label_path"]
        or raw_path in exclusion_sets["quality_tail_raw_path"]
    ):
        return None, "quality_tail_overlap"

    overlaps_prior = (
        family_id in exclusion_sets["family_id"]
        or label_path in exclusion_sets["label_path"]
        or raw_path in exclusion_sets["raw_path"]
    )
    if not overlaps_prior:
        return {
            "reuse_tier": "Tier 0 fresh-only",
            "source_task": "",
            "source_split": "",
            "locked_split": "",
            "reuse_reason": "fresh_seed_no_prior_overlap",
        }, ""

    train_allowed = (
        family_id in exclusion_sets["split_lock_train_family_id"]
        or label_path in exclusion_sets["split_lock_train_label_path"]
        or raw_path in exclusion_sets["split_lock_train_raw_path"]
    )
    if train_allowed:
        return {
            "reuse_tier": "Tier 2 cross-task split-locked reuse",
            "source_task": "objective_single_best",
            "source_split": "train",
            "locked_split": "train",
            "reuse_reason": "objective_train_family_reused_for_descriptive_train_only",
        }, ""
    return None, "prior_overlap_not_split_lock_allowed"


def schedule_candidate_indices(total: int, required_count: int) -> list[int]:
    if required_count <= 0:
        return []
    if total <= required_count:
        return list(range(total))
    return sorted({min(total - 1, round(index * (total - 1) / max(required_count - 1, 1))) for index in range(required_count)})


def choose_explanation_style(sample: dict) -> str:
    styles = sample.get("candidate_styles", [])
    if "legal_priority" in styles:
        return "legal_priority"
    if styles:
        return styles[0]
    return "single"


def build_seed_from_record(record: dict) -> dict:
    card = build_card(record)
    transformed = build_transformed_sample(card)
    style_name = choose_explanation_style(transformed)
    explanation = build_local_fallback_explanation(transformed, style_name, EXPLANATION_GENERATION_VARIANT)
    explanation = postprocess_generated_explanation(transformed, explanation)
    seed_row = base.build_seed_row(
        {
            "sample_id": record["sample_id"],
            "family_id": record["family_id"],
            "doc_type_name": record["doc_type_name"],
            "source_subset": record["source_subset"],
            "sampling_lane": record["sampling_lane"],
            "answer_mode": transformed.get("answer_mode", "") or "criteria",
            "explanation_target": transformed.get("explanation_target", ""),
            "transformed_problem": transformed["transformed_problem"],
            "short_answer": transformed["short_answer"],
            "generated_explanation": explanation,
            "rule_basis": transformed["evidence_card"].get("rule_basis", ""),
            "fact_basis": transformed["evidence_card"].get("fact_basis", ""),
            "label_path": record["label_path"],
            "raw_path": record["raw_path"],
        }
    )
    seed_row.update(
        {
            "selection_role": "descriptive_production_wave_candidate_seed",
            "selection_note": "fresh AI Hub QA seed를 v3 split-descriptive recipe로 후보 생성하는 production wave seed",
            "candidate_route_name": ROUTE_NAME,
            "candidate_source_schedule": ACTIVE_ROUTE["route_label"],
            "descriptive_explanation_style": style_name,
            "reuse_tier": record.get("reuse_tier", ""),
            "source_task": record.get("source_task", ""),
            "source_split": record.get("source_split", ""),
            "locked_split": record.get("locked_split", ""),
            "reuse_reason": record.get("reuse_reason", ""),
        }
    )
    return seed_row


def select_source_records(source_counts: dict[str, int], exclusion_sets: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    audit_rows: list[dict] = []
    used_family_ids: set[str] = set()
    used_label_paths: set[str] = set()
    used_raw_paths: set[str] = set()
    sample_order = 1

    spec_by_source = {spec["source_subset"]: spec for spec in DATASET_SPECS}
    for source_subset, required_count in source_counts.items():
        spec = spec_by_source[source_subset]
        label_paths = explanation_common.list_label_files(spec["label_glob"])
        raw_paths = explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = schedule_candidate_indices(len(label_paths), required_count)
        source_selected = 0

        for selected_index in selected_indices:
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            skip_reason = "candidate_not_found"
            for candidate_index in candidate_indices:
                label_path = label_paths[candidate_index]
                payload = explanation_common.normalize_label_payload(
                    label_path,
                    explanation_common.load_json(label_path),
                    spec["doc_type_name"],
                )
                try:
                    raw_path = explanation_common.locate_raw_path(raw_paths, spec["doc_type_name"], payload["info"])
                except FileNotFoundError:
                    skip_reason = "raw_path_missing"
                    continue
                family_id = explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
                if family_id in used_family_ids:
                    skip_reason = "family_overlap_in_batch"
                    continue
                if str(label_path) in used_label_paths:
                    skip_reason = "label_path_overlap_in_batch"
                    continue
                if str(raw_path) in used_raw_paths:
                    skip_reason = "raw_path_overlap_in_batch"
                    continue
                reuse_meta, skip_reason = classify_reuse_policy(family_id, str(label_path), str(raw_path), exclusion_sets)
                if reuse_meta is None:
                    continue
                chosen = (candidate_index, label_path, raw_path, family_id, payload, reuse_meta)
                break

            if chosen is None:
                audit_rows.append(
                    {
                        "source_subset": source_subset,
                        "doc_type_name": spec["doc_type_name"],
                        "required_count": str(required_count),
                        "selected_count": str(source_selected),
                        "skip_reason": skip_reason,
                    }
                )
                continue

            candidate_index, label_path, raw_path, family_id, payload, reuse_meta = chosen
            info = payload["info"]
            label = payload["label"]
            sample_id = f"desc_wave_{sample_order:03d}"
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_order": sample_order,
                    "source_subset": source_subset,
                    "domain": spec["domain"],
                    "doc_type_name": spec["doc_type_name"],
                    "sampling_lane": spec.get("sampling_lane", ""),
                    "source_schema": info.get("source_schema", ""),
                    "family_id": family_id,
                    "title": explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                    "info_json": json.dumps(info, ensure_ascii=False),
                    "label_path": str(label_path),
                    "raw_path": str(raw_path),
                    "label_input": label["input"],
                    "label_output": label["output"],
                    "selected_index": candidate_index,
                    "selection_note": "descriptive split-lock production wave candidate seed",
                    **reuse_meta,
                }
            )
            used_family_ids.add(family_id)
            used_label_paths.add(str(label_path))
            used_raw_paths.add(str(raw_path))
            source_selected += 1
            sample_order += 1

    return records, audit_rows


def objective_fresh_reuse_policy(family_id: str, label_path: str, raw_path: str, exclusion_sets: dict[str, set[str]]) -> tuple[str, str]:
    # objective availability는 cross-task split-lock을 열지 않고, 완전 fresh-only 후보만 별도로 센다.
    if (
        family_id in exclusion_sets["global_eval_family_id"]
        or label_path in exclusion_sets["global_eval_label_path"]
        or raw_path in exclusion_sets["global_eval_raw_path"]
    ):
        return "", "global_eval_holdout_overlap"
    if (
        family_id in exclusion_sets["quality_tail_family_id"]
        or label_path in exclusion_sets["quality_tail_label_path"]
        or raw_path in exclusion_sets["quality_tail_raw_path"]
    ):
        return "", "quality_tail_overlap"
    if (
        family_id in exclusion_sets["family_id"]
        or label_path in exclusion_sets["label_path"]
        or raw_path in exclusion_sets["raw_path"]
    ):
        return "", "prior_overlap"
    return "Tier 0 fresh-only", ""


def collect_seed_availability(exclusion_sets: dict[str, set[str]]) -> tuple[list[dict], dict[str, int]]:
    # availability map은 route 선택을 감으로 하지 않기 위한 no-API proof다.
    availability_rows: list[dict] = []
    descriptive_available_by_source: dict[str, int] = {}
    for spec in DATASET_SPECS:
        source_subset = spec["source_subset"]
        label_paths = explanation_common.list_label_files(spec["label_glob"])
        raw_paths = explanation_common.list_raw_files(spec["raw_glob"])
        counters: Counter[tuple[str, str, str, str]] = Counter()
        for label_path in label_paths:
            payload = explanation_common.normalize_label_payload(
                label_path,
                explanation_common.load_json(label_path),
                spec["doc_type_name"],
            )
            try:
                raw_path = explanation_common.locate_raw_path(raw_paths, spec["doc_type_name"], payload["info"])
            except FileNotFoundError:
                counters[("descriptive", "unavailable", "raw_path_missing", "")] += 1
                continue
            family_id = explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
            reuse_meta, descriptive_skip = classify_reuse_policy(family_id, str(label_path), str(raw_path), exclusion_sets)
            if reuse_meta is None:
                counters[("descriptive", "unavailable", descriptive_skip, "")] += 1
            else:
                reuse_tier = reuse_meta["reuse_tier"]
                split_scope = "train_only" if reuse_tier.startswith("Tier 2") else "train_dev_test"
                counters[("descriptive", "available", reuse_tier, split_scope)] += 1
                descriptive_available_by_source[source_subset] = descriptive_available_by_source.get(source_subset, 0) + 1

            objective_tier, objective_skip = objective_fresh_reuse_policy(family_id, str(label_path), str(raw_path), exclusion_sets)
            if objective_tier:
                counters[("objective", "available", objective_tier, "train_dev_test")] += 1
            else:
                counters[("objective", "unavailable", objective_skip, "")] += 1

        for (task_axis, availability_status, reuse_tier_or_reason, split_scope), count in sorted(counters.items()):
            availability_rows.append(
                {
                    "task_axis": task_axis,
                    "doc_type_name": spec["doc_type_name"],
                    "source_subset": source_subset,
                    "sampling_lane": spec.get("sampling_lane", ""),
                    "availability_status": availability_status,
                    "reuse_tier_or_reason": reuse_tier_or_reason,
                    "split_scope": split_scope,
                    "available_count": str(count) if availability_status == "available" else "0",
                    "blocked_count": str(count) if availability_status == "unavailable" else "0",
                }
            )
    return availability_rows, descriptive_available_by_source


def build_route_feasibility_rows(routes: list[dict], available_by_source: dict[str, int]) -> list[dict]:
    # route feasibility는 부족 source를 명시해 medium/fallback 전환 근거를 reviewer가 바로 읽게 한다.
    rows: list[dict] = []
    for route in routes:
        missing_parts = []
        for source_subset, required_count in sorted(route["source_counts"].items()):
            available_count = available_by_source.get(source_subset, 0)
            if available_count < required_count:
                missing_parts.append(f"{source_subset}:{available_count}/{required_count}")
        rows.append(
            {
                "route_label": route["route_label"],
                "candidate_required": str(sum(route["source_counts"].values())),
                "candidate_available_for_required_sources": str(
                    sum(min(available_by_source.get(source_subset, 0), required_count) for source_subset, required_count in route["source_counts"].items())
                ),
                "feasible": YES if not missing_parts else NO,
                "missing_source_quota": "; ".join(missing_parts),
                "primary_final_target": str(route["target_count"]),
                "fallback_final_target": str(route.get("fallback_target_count", "")),
            }
        )
    return rows


def build_route_probe_availability_rows(route: dict, records: list[dict], audit_rows: list[dict]) -> list[dict]:
    # 전체 raw pool 전수 스캔은 마감 운영에서 너무 무겁기 때문에, 이번 stop line은 route quota별 probe 결과를 availability proof로 남긴다.
    selected_counts = Counter(row["source_subset"] for row in records)
    selected_reuse = Counter((row["source_subset"], row.get("reuse_tier", ""), row.get("locked_split", "")) for row in records)
    rows: list[dict] = []
    for source_subset, required_count in sorted(route["source_counts"].items()):
        source_records = [row for row in records if row["source_subset"] == source_subset]
        doc_type = source_records[0]["doc_type_name"] if source_records else ""
        lane = source_records[0]["sampling_lane"] if source_records else ""
        if not doc_type:
            audit_for_source = next((row for row in audit_rows if row["source_subset"] == source_subset), {})
            doc_type = audit_for_source.get("doc_type_name", "")
        for (reuse_source, reuse_tier, locked_split), count in sorted(selected_reuse.items()):
            if reuse_source != source_subset:
                continue
            rows.append(
                {
                    "task_axis": "descriptive",
                    "doc_type_name": doc_type,
                    "source_subset": source_subset,
                    "sampling_lane": lane,
                    "availability_status": "available",
                    "reuse_tier_or_reason": reuse_tier,
                    "split_scope": "train_only" if locked_split == "train" else "train_dev_test",
                    "available_count": str(count),
                    "blocked_count": "0",
                }
            )
        if selected_counts.get(source_subset, 0) < required_count:
            rows.append(
                {
                    "task_axis": "descriptive",
                    "doc_type_name": doc_type,
                    "source_subset": source_subset,
                    "sampling_lane": lane,
                    "availability_status": "unavailable",
                    "reuse_tier_or_reason": "route_quota_shortfall",
                    "split_scope": "",
                    "available_count": str(selected_counts.get(source_subset, 0)),
                    "blocked_count": str(required_count - selected_counts.get(source_subset, 0)),
                }
            )
    return rows


def write_availability_map(availability_rows: list[dict], route_rows: list[dict]) -> None:
    base.write_csv_atomic(AVAILABILITY_MAP_CSV_PATH, availability_rows, list(availability_rows[0].keys()) if availability_rows else [])
    base.write_csv_atomic(AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH, route_rows, list(route_rows[0].keys()) if route_rows else [])
    lines = [
        f"# seed availability map `{VERSION_TAG}`",
        "",
        "## route feasibility",
        "| route_label | candidate_required | candidate_available_for_required_sources | feasible | missing_source_quota |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in route_rows:
        lines.append(
            f"| `{row['route_label']}` | `{row['candidate_required']}` | `{row['candidate_available_for_required_sources']}` | `{row['feasible']}` | {row['missing_source_quota'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## availability table",
            "| task_axis | doc_type_name | source_subset | sampling_lane | status | reuse_tier_or_reason | split_scope | available | blocked |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in availability_rows:
        lines.append(
            f"| `{row['task_axis']}` | `{row['doc_type_name']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | "
            f"`{row['availability_status']}` | `{row['reuse_tier_or_reason']}` | `{row['split_scope']}` | "
            f"`{row['available_count']}` | `{row['blocked_count']}` |"
        )
    write_text_atomic(AVAILABILITY_MAP_MD_PATH, "\n".join(lines) + "\n")


def choose_route(exclusion_sets: dict[str, set[str]]) -> tuple[dict, list[dict], list[dict]]:
    routes = []
    deficit_closure_mode = os.environ.get("DESCRIPTIVE_WAVE_ENABLE_DEFICIT_CLOSURE") == "1"
    train_micro_closure_mode = os.environ.get("DESCRIPTIVE_WAVE_ENABLE_TRAIN_MICRO_CLOSURE") == "1"
    type_deficit_overclosure_mode = os.environ.get("DESCRIPTIVE_WAVE_ENABLE_TYPE_DEFICIT_OVERCLOSURE") == "1"
    type_balance_extension_mode = os.environ.get("DESCRIPTIVE_WAVE_ENABLE_TYPE_BALANCE_EXTENSION") == "1"
    source_reallocated_mode = (
        os.environ.get("DESCRIPTIVE_WAVE_ENABLE_SOURCE_REALLOCATED") == "1"
        or os.environ.get("DESCRIPTIVE_WAVE_ENABLE_SOURCE_REALLOCATED_EVAL_TARGETED") == "1"
    )
    eval_targeted_mode = os.environ.get("DESCRIPTIVE_WAVE_ENABLE_SOURCE_REALLOCATED_EVAL_TARGETED") == "1"
    if train_micro_closure_mode:
        # reviewer가 "시간이 남으면 서술형 train 5건도 닫으라"고 언급했으므로,
        # 과잉 final package를 만들지 않고 판결문_QA train 부족 5건만 정확히 보강한다.
        routes.append(
            with_train_micro_closure(
                {
                    "route_label": "train_micro_closure_judgment_candidate16_final5",
                    "target_count": 5,
                    "source_counts": {
                        "01_TL_판결문_QA": 4,
                        "02_TL_판결문_QA": 4,
                        "03_TL_판결문_QA": 4,
                        "04_TL_판결문_QA": 4,
                    },
                    "final_source_counts": {
                        "01_TL_판결문_QA": 1,
                        "02_TL_판결문_QA": 1,
                        "03_TL_판결문_QA": 1,
                        "04_TL_판결문_QA": 2,
                    },
                    "source_balance_relaxation": "post-181810 descriptive train micro-closure; 판결문_QA train shortage 5건 보강",
                }
            )
        )
    elif type_deficit_overclosure_mode:
        # objective와 함께 유형별 균형을 닫는 reviewer stop line용 route다.
        # primary 48/24가 source availability로 막히면 32/16 fallback을 같은 runner에서 자동 시도한다.
        routes.extend(
            [
                with_type_deficit_overclosure(
                    {
                        "route_label": "type_deficit_overclosure_candidate48_final24",
                        "target_count": 24,
                        "source_counts": {
                            "01_TL_유권해석_QA": 16,
                            "03_TL_해석례_QA": 16,
                            "03_TL_결정례_QA": 8,
                            "04_TL_결정례_QA": 8,
                        },
                        "final_source_counts": {
                            "01_TL_유권해석_QA": 8,
                            "03_TL_해석례_QA": 8,
                            "03_TL_결정례_QA": 4,
                            "04_TL_결정례_QA": 4,
                        },
                        "source_balance_relaxation": "type-deficit overclosure primary; 해석례 train 16 + 결정례 eval 8 final package",
                    },
                    train_count=16,
                    dev_count=4,
                    test_count=4,
                ),
                with_type_deficit_overclosure(
                    {
                        "route_label": "type_deficit_overclosure_candidate32_final16",
                        "target_count": 16,
                        "source_counts": {
                            "01_TL_유권해석_QA": 12,
                            "03_TL_해석례_QA": 12,
                            "03_TL_결정례_QA": 4,
                            "04_TL_결정례_QA": 4,
                        },
                        "final_source_counts": {
                            "01_TL_유권해석_QA": 6,
                            "03_TL_해석례_QA": 6,
                            "03_TL_결정례_QA": 2,
                            "04_TL_결정례_QA": 2,
                        },
                        "source_balance_relaxation": "type-deficit overclosure fallback; 해석례 train 12 + 결정례 eval 4 final package",
                    },
                    train_count=12,
                    dev_count=2,
                    test_count=2,
                ),
            ]
        )
    elif type_balance_extension_mode:
        # KCC 제출 전 유형별 균형을 같이 맞추기 위한 별도 extension route다.
        # aggregate target은 이미 닫혔으므로 final package는 countable extension이지만, 보고에서는 balance add-on으로 분리한다.
        routes.append(
            with_type_balance_extension(
                {
                    "route_label": "type_balance_extension_candidate64_final40",
                    "target_count": 40,
                    "source_counts": {
                        "03_TL_법령_QA": 8,
                        "04_TL_법령_QA": 8,
                        "01_TL_유권해석_QA": 8,
                        "03_TL_해석례_QA": 8,
                        "01_TL_심결례_QA": 5,
                        "02_TL_심결례_QA": 3,
                        "02_TL_심결문_QA": 3,
                        "03_TL_결정례_QA": 4,
                        "04_TL_결정례_QA": 5,
                        "01_TL_판결문_QA": 3,
                        "02_TL_판결문_QA": 3,
                        "03_TL_판결문_QA": 3,
                        "04_TL_판결문_QA": 3,
                    },
                    "final_source_counts": {
                        "03_TL_법령_QA": 4,
                        "04_TL_법령_QA": 5,
                        "01_TL_유권해석_QA": 5,
                        "03_TL_해석례_QA": 5,
                        "01_TL_심결례_QA": 3,
                        "02_TL_심결례_QA": 2,
                        "02_TL_심결문_QA": 1,
                        "03_TL_결정례_QA": 3,
                        "04_TL_결정례_QA": 3,
                        "01_TL_판결문_QA": 2,
                        "02_TL_판결문_QA": 2,
                        "03_TL_판결문_QA": 2,
                        "04_TL_판결문_QA": 3,
                    },
                    "source_balance_relaxation": "descriptive aggregate target is closed; source/doc quotas intentionally reallocated for type balance reporting",
                }
            )
        )
    elif deficit_closure_mode:
        # reviewer 회신의 다음 API-first stop line: 이미 eval target은 2건만 남았으므로,
        # 같은 source-reallocated contract를 쓰되 final split을 train-heavy add-on으로 축소한다.
        routes.extend(
            [
                with_deficit_closure(
                    {
                        "route_label": "deficit_closure_train_heavy_candidate64_final40",
                        "target_count": 40,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            64,
                            cap_interpretation_04=True,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            40,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": "post-174728 deficit closure; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA; train-heavy split 38/1/1",
                    }
                ),
                with_deficit_closure(
                    {
                        "route_label": "deficit_closure_train_heavy_candidate40_final24",
                        "target_count": 24,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            40,
                            cap_interpretation_04=False,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            24,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": "post-174728 deficit closure fallback; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA; train-heavy split 22/1/1",
                    }
                ),
                with_deficit_closure(
                    {
                        "route_label": "deficit_closure_train_heavy_candidate24_final12",
                        "target_count": 12,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            24,
                            cap_interpretation_04=False,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            12,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": "post-174728 deficit closure final fallback; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA; train-heavy split 10/1/1",
                    }
                ),
            ]
        )
    elif source_reallocated_mode:
        # post-154712 API-first route: 기존 emergency schedule을 반복하지 않고,
        # depleted `02_TL_법령_QA` quota를 같은 법령 source로 재배정한 route만 시도한다.
        route_prefix = "source_reallocated_eval_targeted" if eval_targeted_mode else "source_reallocated"
        source_balance_prefix = (
            "post-171359 eval-targeted source reallocation"
            if eval_targeted_mode
            else "post-154712 source reallocation"
        )
        route_wrapper = with_eval_targeting if eval_targeted_mode else (lambda route: route)
        routes.extend(
            [
                route_wrapper(route_with_split(
                    {
                        "route_label": f"{route_prefix}_candidate128_final80",
                        "target_count": 80,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            128,
                            cap_interpretation_04=True,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            80,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": f"{source_balance_prefix}; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA; 04_TL_해석례_QA capped for candidate128",
                    }
                )),
                route_wrapper(route_with_split(
                    {
                        "route_label": f"{route_prefix}_candidate96_final64",
                        "target_count": 64,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            96,
                            cap_interpretation_04=True,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            64,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": f"{source_balance_prefix} fallback; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA",
                    }
                )),
                route_wrapper(route_with_split(
                    {
                        "route_label": f"{route_prefix}_candidate64_final40",
                        "target_count": 40,
                        "source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_SOURCE_COUNTS,
                            64,
                            cap_interpretation_04=True,
                        ),
                        "final_source_counts": source_reallocated_counts(
                            MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            40,
                            cap_interpretation_04=False,
                        ),
                        "source_balance_relaxation": f"{source_balance_prefix} final fallback; 02_TL_법령_QA=0; law quota moved to 03/04_TL_법령_QA",
                    }
                )),
            ]
        )
    else:
        # legacy schedule은 post-154712 기준 P2 blocker로 판정됐으므로,
        # 명시적 source-reallocated mode가 아닐 때만 이전 route compatibility를 유지한다.
        if os.environ.get("DESCRIPTIVE_WAVE_ENABLE_EMERGENCY128") == "1":
            # reviewer emergency stop line: 128 -> 96 -> 64 순서로 preflight 가능한 최대 route를 고른다.
            routes.extend(
                [
                    route_with_split(
                        {
                            "route_label": "emergency_source_relaxed_candidate128_final80",
                            "target_count": 80,
                            "source_counts": scaled_source_counts(MEDIUM_RELAXED_SOURCE_COUNTS, 128),
                            "final_source_counts": scaled_source_counts(MEDIUM_RELAXED_FINAL_SOURCE_COUNTS, 80),
                            "source_balance_relaxation": "emergency candidate128; scaled from medium source-relaxed schedule; 01_TL_법령_QA remains excluded",
                        }
                    ),
                    route_with_split(
                        {
                            "route_label": "emergency_source_relaxed_candidate96_final64",
                            "target_count": 64,
                            "source_counts": scaled_source_counts(MEDIUM_RELAXED_SOURCE_COUNTS, 96),
                            "final_source_counts": scaled_source_counts(MEDIUM_RELAXED_FINAL_SOURCE_COUNTS, 64),
                            "source_balance_relaxation": "emergency fallback candidate96; scaled from medium source-relaxed schedule; 01_TL_법령_QA remains excluded",
                        }
                    ),
                    route_with_split(
                        {
                            "route_label": "emergency_source_relaxed_candidate64_final40",
                            "target_count": 40,
                            "source_counts": scaled_source_counts(MEDIUM_RELAXED_SOURCE_COUNTS, 64),
                            "final_source_counts": MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                            "source_balance_relaxation": "emergency fallback candidate64; same law lane 02/03/04 absorbs depleted 01_TL_법령_QA quota",
                        }
                    ),
                ]
            )
        if os.environ.get("DESCRIPTIVE_WAVE_ENABLE_CANDIDATE64") == "1":
            candidate64_source_counts = scaled_source_counts(MEDIUM_RELAXED_SOURCE_COUNTS, 64)
            routes.append(
                route_with_split(
                    {
                        "route_label": "medium_source_relaxed_candidate64_final40",
                        "target_count": 40,
                        "source_counts": candidate64_source_counts,
                        "final_source_counts": MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                        "source_balance_relaxation": "01_TL_법령_QA depleted; same law lane 02/03/04 absorbs quota; candidate64 scaled from medium source-relaxed schedule",
                    }
                )
            )
        routes.extend(
            [
                route_with_split(
                    {
                        "route_label": "medium_source_relaxed_candidate56_final40",
                        "target_count": 40,
                        "source_counts": MEDIUM_RELAXED_SOURCE_COUNTS,
                        "final_source_counts": MEDIUM_RELAXED_FINAL_SOURCE_COUNTS,
                        "source_balance_relaxation": "01_TL_법령_QA depleted; same law lane 02/03/04 absorbs +1/+1/+1",
                    }
                ),
                route_with_split(
                    {
                        "route_label": "constrained_candidate34_primary24_fallback20",
                        "target_count": 24,
                        "fallback_target_count": 20,
                        "source_counts": CONSTRAINED_SOURCE_COUNTS,
                        "final_source_counts": CONSTRAINED_PRIMARY_FINAL_SOURCE_COUNTS,
                        "fallback_final_source_counts": CONSTRAINED_FALLBACK_FINAL_SOURCE_COUNTS,
                    }
                ),
            ]
        )
    route_rows = []
    availability_rows = []
    route_attempts = []
    for route in routes:
        records, audit_rows = select_source_records(route["source_counts"], exclusion_sets)
        route_attempts.append((route, records, audit_rows))
        required_count = sum(route["source_counts"].values())
        # source별 shortfall이 candidate probe 과정에서 여러 번 누적될 수 있어 reviewer-facing 표는 source 단위로 압축한다.
        missing_by_source = {
            row["source_subset"]: f"{row['source_subset']}:{row['selected_count']}/{row['required_count']}"
            for row in audit_rows
            if row.get("selected_count") != row.get("required_count")
        }
        route_rows.append(
            {
                "route_label": route["route_label"],
                "candidate_required": str(required_count),
                "candidate_available_for_required_sources": str(len(records)),
                "feasible": YES if len(records) == required_count else NO,
                "missing_source_quota": "; ".join(missing_by_source[source_subset] for source_subset in sorted(missing_by_source)),
                "primary_final_target": str(route["target_count"]),
                "fallback_final_target": str(route.get("fallback_target_count", "")),
            }
        )
        availability_rows.extend(build_route_probe_availability_rows(route, records, audit_rows))
    write_availability_map(availability_rows, route_rows)
    for route, records, audit_rows in route_attempts:
        if len(records) == sum(route["source_counts"].values()):
            return route, records, audit_rows
    # medium proof가 실패하면 같은 stop line 안에서 constrained fallback으로 내려가고,
    # fallback quota도 채우지 못할 때만 API를 태우지 않고 seed availability blocker로 닫는다.
    raise RuntimeError(
        "descriptive_seed_availability_blocker: "
        f"required={sum(routes[-1]['source_counts'].values())}, available={len(records)}, "
        f"route={routes[-1]['route_label']}"
    )


def write_preflight(seed_rows: list[dict], exclusion_sets: dict[str, set[str]], availability_audit: list[dict]) -> None:
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows: list[dict] = []
    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "family_duplicate_in_batch": YES if family_counts[row["family_id"]] > 1 else NO,
                "label_path_duplicate_in_batch": YES if label_counts[row["label_path"]] > 1 else NO,
                "raw_path_duplicate_in_batch": YES if raw_counts[row["raw_path"]] > 1 else NO,
                "family_overlap_with_prior": YES if row["family_id"] in exclusion_sets["family_id"] else NO,
                "label_path_overlap_with_prior": YES if row["label_path"] in exclusion_sets["label_path"] else NO,
                "raw_path_overlap_with_prior": YES if row["raw_path"] in exclusion_sets["raw_path"] else NO,
                "reuse_tier": row.get("reuse_tier", ""),
                "source_task": row.get("source_task", ""),
                "source_split": row.get("source_split", ""),
                "locked_split": row.get("locked_split", ""),
                "reuse_reason": row.get("reuse_reason", ""),
                "answer_mode": row["answer_mode"],
                "problem_generation_mode": row["problem_generation_mode"],
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    for row in preflight_rows:
        batch_duplicate = YES in [
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
        ]
        prior_overlap = YES in [
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        split_lock_ok = row["reuse_tier"] == "Tier 2 cross-task split-locked reuse" and row["locked_split"] == "train"
        if batch_duplicate or (prior_overlap and not split_lock_ok):
            raise RuntimeError(f"descriptive preflight overlap failed: {row['seed_sample_id']}")

    source_counts = Counter(row["source_subset"] for row in seed_rows)
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    reuse_counts = Counter(row.get("reuse_tier", "") for row in seed_rows)
    base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))
    base.write_csv_atomic(
        EXCLUSION_AUDIT_CSV_PATH,
        availability_audit,
        list(availability_audit[0].keys()) if availability_audit else ["source_subset", "doc_type_name", "required_count", "selected_count", "skip_reason"],
    )
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- route_label: `{ACTIVE_ROUTE['route_label']}`",
        f"- candidate_count: `{len(seed_rows)}`",
        f"- final_target_count: `{ACTIVE_ROUTE['target_count']}`",
        f"- final_split_target: `train {ACTIVE_ROUTE.get('final_train_count', '')} / dev {ACTIVE_ROUTE.get('final_dev_count', '')} / test {ACTIVE_ROUTE.get('final_test_count', '')}`",
        f"- fallback_final_target_count: `{ACTIVE_ROUTE.get('fallback_target_count', '')}`",
        f"- source_balance_relaxation: `{ACTIVE_ROUTE.get('source_balance_relaxation', '없음')}`",
        f"- eval_targeting_contract: `{ACTIVE_ROUTE.get('eval_targeting_contract', '없음')}`",
        f"- eval_balanced_doc_types: `{ACTIVE_ROUTE.get('eval_balanced_doc_types', [])}`",
        f"- eval_avoid_doc_types: `{ACTIVE_ROUTE.get('eval_avoid_doc_types', [])}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- reuse_tier_counts: `{dict(reuse_counts)}`",
        "",
        "## source subset counts",
        "| source_subset | count |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(
        [
            "",
            "## checks",
            "| check | result |",
            "| --- | --- |",
            "| candidate source schedule satisfied | `pass` |",
            "| no batch family_id/label_path/raw_path duplicate | `pass` |",
            "| prior overlap policy | `pass` | Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용 |",
            "| global eval and quality tail exclusion | `pass` | dev/test, audit, hard/soft fail family는 재사용 금지 |",
            f"| eval targeting contract | `{'pass' if ACTIVE_ROUTE.get('eval_targeting_contract') else 'not_applicable'}` | {ACTIVE_ROUTE.get('eval_targeting_contract', '일반 global split 사용')} |",
        ]
    )
    write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")


def build_seed_registry() -> list[dict]:
    base.ensure_dirs(
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
    exclusion_rows = collect_exclusion_rows()
    exclusion_sets = build_exclusion_sets(exclusion_rows)
    global ACTIVE_ROUTE
    ACTIVE_ROUTE, records, availability_audit = choose_route(exclusion_sets)
    seed_rows = [build_seed_from_record(record) for record in records]
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["source_subset"], row["seed_sample_id"]))
    write_preflight(seed_rows, exclusion_sets, availability_audit)
    base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    for path in [
        SEED_REGISTRY_PATH,
        SEED_READY_PATH,
        SEED_PREFLIGHT_CSV_PATH,
        SEED_PREFLIGHT_MD_PATH,
        EXCLUSION_AUDIT_CSV_PATH,
        AVAILABILITY_MAP_CSV_PATH,
        AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH,
        AVAILABILITY_MAP_MD_PATH,
    ]:
        base.copy_file_to_run_inputs(path, RUN_INPUTS_DIR)
    return seed_rows


def reject_metadata(row: dict, reason: str) -> dict:
    if reason == "quota_surplus":
        return {
            **row,
            "pool_class": "quota_surplus",
            **candidate_contract_fields(),
            "promotion_contract_status": "not_promoted_quota_surplus",
            "final_package_selected": NO,
            "quality_failure": NO,
            "tail_class": "quota_surplus_not_quality_failure",
            "future_candidate_reusable": YES,
            "candidate_reuse_policy": "reuse_allowed_as_surplus_candidate",
            "selection_reason": "strict_pass_not_selected_due_final_source_quota",
            "not_selected_reason": "source_quota_filled",
        }
    return {
        **row,
        "pool_class": "quality_reject",
        **candidate_contract_fields(),
        "promotion_contract_status": "candidate_pool_not_promoted",
        "final_package_selected": NO,
        "quality_failure": YES,
        "tail_class": reason,
        "future_candidate_reusable": NO,
        "candidate_reuse_policy": "do_not_reuse_without_repair_review",
        "selection_reason": "",
        "not_selected_reason": reason,
    }


def attach_seed_reuse_metadata(row: dict, seed_meta_by_sample_id: dict[str, dict[str, str]]) -> dict:
    # 생성/Judge 중간 산출물에서 reuse metadata가 빠질 수 있어, final compiler 직전에 seed registry를 다시 붙여 split-lock gate의 근거를 보존한다.
    seed_meta = seed_meta_by_sample_id.get(row.get("seed_sample_id", ""), {})
    return {
        **row,
        "reuse_tier": row.get("reuse_tier") or seed_meta.get("reuse_tier", ""),
        "source_task": row.get("source_task") or seed_meta.get("source_task", ""),
        "source_split": row.get("source_split") or seed_meta.get("source_split", ""),
        "locked_split": row.get("locked_split") or seed_meta.get("locked_split", ""),
        "reuse_reason": row.get("reuse_reason") or seed_meta.get("reuse_reason", ""),
    }


def is_tier2_train_locked(row: dict) -> bool:
    # Tier 2 cross-task reuse는 source family 재사용을 허용하되, 평가 split에는 절대 배치하지 않는다는 정책을 코드로 잠근다.
    return (
        str(row.get("reuse_tier", "")).startswith("Tier 2")
        and row.get("source_split") == "train"
        and row.get("locked_split") == "train"
    )


def eval_candidate_sort_key(row: dict) -> tuple:
    priority_doc_types = ACTIVE_ROUTE.get("eval_priority_doc_types", [])
    avoid_doc_types = set(ACTIVE_ROUTE.get("eval_avoid_doc_types", []))
    doc_type = row.get("doc_type_name", "")
    if doc_type in priority_doc_types:
        priority = priority_doc_types.index(doc_type)
    elif doc_type in avoid_doc_types:
        priority = 99
    else:
        priority = 50
    return (
        priority,
        row.get("source_subset", ""),
        -float(row.get("weighted_score", 0) or 0),
        row.get("seed_sample_id", ""),
    )


def round_robin_by_doc_type(rows_by_doc_type: dict[str, list[dict]], doc_types: list[str]) -> list[dict]:
    # dev/test가 한 문서유형으로만 쏠리지 않게 법령/결정례 eval 후보를 교차 배치한다.
    ordered: list[dict] = []
    max_count = max((len(rows_by_doc_type.get(doc_type, [])) for doc_type in doc_types), default=0)
    for index in range(max_count):
        for doc_type in doc_types:
            rows = rows_by_doc_type.get(doc_type, [])
            if index < len(rows):
                ordered.append(rows[index])
    return ordered


def choose_eval_target_rows(rows: list[dict], eval_count: int) -> list[dict]:
    # reviewer P2: 다음 source-reallocated run은 총량 반복이 아니라 법령/결정례 eval 부족을 먼저 보강한다.
    # Tier 2 split-lock row는 호출 전 제외되어 있어, 여기서 뽑힌 row만 dev/test 후보가 된다.
    if eval_count <= 0:
        return []
    balanced_doc_types = ACTIVE_ROUTE.get("eval_balanced_doc_types", [])
    target_doc_counts = ACTIVE_ROUTE.get("eval_target_doc_counts", {})
    selected: list[dict] = []
    selected_ids: set[str] = set()
    if target_doc_counts:
        for doc_type, target_count in target_doc_counts.items():
            doc_rows = sorted(
                [row for row in rows if row.get("doc_type_name") == doc_type],
                key=eval_candidate_sort_key,
            )
            if len(doc_rows) < int(target_count):
                raise RuntimeError(
                    f"eval_targeting_doc_count_failed: doc_type={doc_type}, required={target_count}, available={len(doc_rows)}"
                )
            for row in doc_rows[: int(target_count)]:
                selected.append(row)
                selected_ids.add(row["candidate_id"])
    elif balanced_doc_types:
        target_per_doc = {doc_type: eval_count // len(balanced_doc_types) for doc_type in balanced_doc_types}
        for doc_type in balanced_doc_types[: eval_count % len(balanced_doc_types)]:
            target_per_doc[doc_type] += 1
        by_doc_type = {
            doc_type: sorted(
                [row for row in rows if row.get("doc_type_name") == doc_type],
                key=eval_candidate_sort_key,
            )[:target_per_doc[doc_type]]
            for doc_type in balanced_doc_types
        }
        for row in round_robin_by_doc_type(by_doc_type, balanced_doc_types):
            selected.append(row)
            selected_ids.add(row["candidate_id"])
    if len(selected) < eval_count:
        for row in sorted(rows, key=eval_candidate_sort_key):
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            if len(selected) == eval_count:
                break
    if len(selected) < eval_count:
        raise RuntimeError(
            f"eval_targeting_split_failed: required_eval={eval_count}, available_tier0={len(rows)}"
        )
    return selected[:eval_count]


def split_for_final_rows(final_rows: list[dict]) -> list[dict]:
    # reviewer 권장 split은 final 40 = 32/4/4, final 24 = 20/2/2, fallback final 20 = 16/2/2라서
    # doc-type별 1/1 eval보다 global split을 우선한다.
    # 단, Tier 2 split-lock 재사용 row는 train-only 계약이므로 항상 train 구간 앞으로 고정한다.
    ordered_rows = sorted(
        final_rows,
        key=lambda item: (
            0 if is_tier2_train_locked(item) else 1,
            item["doc_type_name"],
            item["source_subset"],
            item["seed_sample_id"],
        ),
    )
    total = len(ordered_rows)
    active_target = int(ACTIVE_ROUTE.get("active_final_target_count", ACTIVE_ROUTE.get("target_count", total)) or total)
    if active_target == int(ACTIVE_ROUTE.get("fallback_target_count", 0) or 0):
        dev_count = int(ACTIVE_ROUTE.get("fallback_final_dev_count", 0) or 0)
        test_count = int(ACTIVE_ROUTE.get("fallback_final_test_count", 0) or 0)
    else:
        dev_count = int(ACTIVE_ROUTE.get("final_dev_count", 0) or 0)
        test_count = int(ACTIVE_ROUTE.get("final_test_count", 0) or 0)
    if not dev_count and not test_count:
        _, dev_count, test_count = final_split_counts(total)
    train_count = max(total - dev_count - test_count, sum(1 for row in ordered_rows if is_tier2_train_locked(row)))
    if ACTIVE_ROUTE.get("eval_targeting_contract"):
        non_locked_rows = [row for row in ordered_rows if not is_tier2_train_locked(row)]
        eval_rows = choose_eval_target_rows(non_locked_rows, dev_count + test_count)
        dev_ids = {row["candidate_id"] for row in eval_rows[:dev_count]}
        test_ids = {row["candidate_id"] for row in eval_rows[dev_count : dev_count + test_count]}
        with_splits = []
        for row in ordered_rows:
            if row["candidate_id"] in dev_ids:
                split = "dev"
                eval_target_role = "eval_targeted_dev"
            elif row["candidate_id"] in test_ids:
                split = "test"
                eval_target_role = "eval_targeted_test"
            else:
                split = "train"
                eval_target_role = "train_after_eval_targeting"
            with_splits.append(
                {
                    **row,
                    "split": split,
                    "dataset_disposition": split,
                    "eval_targeting_role": eval_target_role,
                    "eval_targeting_contract": ACTIVE_ROUTE.get("eval_targeting_contract", ""),
                }
            )
        return sorted(with_splits, key=lambda item: int(item["selection_rank"]))
    with_splits: list[dict] = []
    for index, row in enumerate(ordered_rows):
        split = "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
        with_splits.append({**row, "split": split, "dataset_disposition": split})
    return sorted(with_splits, key=lambda item: int(item["selection_rank"]))


def compile_final_package(merged_rows: list[dict]) -> dict[str, list[dict]]:
    seed_rows = read_csv_rows(SEED_REGISTRY_PATH)
    seed_meta_by_sample_id = {row["seed_sample_id"]: row for row in seed_rows}
    selected_rows = [
        attach_seed_reuse_metadata(row, seed_meta_by_sample_id)
        for row in merged_rows
        if row.get("selected_for_seed") == YES
    ]
    candidate_pool = [
        {
            **row,
            "pool_class": "candidate_pool",
            **candidate_contract_fields(),
            "promotion_contract_status": "candidate_pool_not_promoted",
            "final_package_selected": NO,
            "quality_failure": "대상아님",
        }
        for row in selected_rows
    ]
    accepted = []
    rejected = []
    for row in selected_rows:
        if row.get("final_status") != "pass":
            rejected.append(reject_metadata(row, "final_status_failure"))
        elif row.get("train_eligible") != YES or row.get("audit_required") == YES:
            rejected.append(reject_metadata(row, "audit_or_not_train_eligible"))
        else:
            accepted.append({**row, "pool_class": "strict_accepted", "quality_failure": NO})

    def select_by_quota(quota: dict[str, int]) -> tuple[list[dict], list[dict], list[dict]]:
        # primary/fallback final quota를 같은 strict accepted pool 위에서 재시도해 API 재실행 없이 안전한 final package를 조립한다.
        selected_by_source = {source_subset: 0 for source_subset in quota}
        quota_final_rows = []
        quota_accepted_with_selection = []
        quota_rejected = []
        for row in sorted(
            accepted,
            key=lambda item: (
                0 if item.get("reuse_tier") == "Tier 0 fresh-only" else 1,
                -float(item.get("weighted_score", 0)),
                item["source_subset"],
                item["seed_sample_id"],
            ),
        ):
            source_subset = row["source_subset"]
            if selected_by_source.get(source_subset, 0) >= quota.get(source_subset, 0):
                surplus = reject_metadata(row, "quota_surplus")
                quota_accepted_with_selection.append(surplus)
                quota_rejected.append(surplus)
                continue
            selected_by_source[source_subset] = selected_by_source.get(source_subset, 0) + 1
            selected = {
                **row,
                "pool_class": "final_package_selected",
                "selection_rank": str(len(quota_final_rows) + 1),
                "selection_reason": "strict_pass_selected_by_source_quota",
                "not_selected_reason": "",
                **candidate_contract_fields(),
            }
            quota_accepted_with_selection.append(selected)
            quota_final_rows.append(selected)
        return quota_final_rows, quota_accepted_with_selection, quota_rejected

    final_rows, accepted_with_selection, quota_rejected = select_by_quota(dict(ACTIVE_ROUTE["final_source_counts"]))
    active_target = int(ACTIVE_ROUTE["target_count"])
    active_final_route = f"primary_final{ACTIVE_ROUTE['target_count']}"
    if len(final_rows) < active_target and ACTIVE_ROUTE.get("fallback_final_source_counts"):
        fallback_rows, fallback_accepted, fallback_rejected = select_by_quota(dict(ACTIVE_ROUTE["fallback_final_source_counts"]))
        fallback_target = int(ACTIVE_ROUTE.get("fallback_target_count", 0))
        if len(fallback_rows) >= fallback_target:
            final_rows = fallback_rows
            accepted_with_selection = fallback_accepted
            quota_rejected = fallback_rejected
            active_target = fallback_target
            active_final_route = f"fallback_final{fallback_target}"
    ACTIVE_ROUTE["active_final_target_count"] = active_target
    ACTIVE_ROUTE["active_final_route_label"] = active_final_route
    rejected.extend(quota_rejected)
    final_rows = split_for_final_rows(final_rows)
    final_contract = counted_contract_fields() if len(final_rows) == active_target else candidate_contract_fields()
    final_ids = {row["candidate_id"] for row in final_rows}
    final_rows = [{**row, **final_contract} for row in final_rows]
    accepted_with_selection = [
        {**row, **final_contract} if row.get("candidate_id") in final_ids else row for row in accepted_with_selection
    ]
    return {
        "candidate_pool": candidate_pool,
        "accepted_pool": accepted_with_selection,
        "rejected_pool": rejected,
        "quality_tail": [row for row in rejected if row.get("quality_failure") == YES],
        "quota_surplus": [row for row in rejected if row.get("pool_class") == "quota_surplus"],
        "final_rows": final_rows,
    }


def union_fields(rows: list[dict]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def final_payload(row: dict) -> dict:
    return {
        "problem_id": row["candidate_id"],
        "seed_sample_id": row["seed_sample_id"],
        "family_id": row["family_id"],
        "doc_type_name": row["doc_type_name"],
        "source_subset": row["source_subset"],
        "sampling_lane": row.get("sampling_lane", ""),
        "problem_task_type": row["problem_task_type"],
        "problem_generation_mode": row["problem_generation_mode"],
        "generated_problem": row["generated_problem"],
        "split_strategy": row.get("split_strategy", ""),
        "focus_issue": row.get("focus_issue", ""),
        "multi_query_signal": row.get("multi_query_signal", ""),
        "split_focus_hint": row.get("split_focus_hint", ""),
        "gold_short_answer": row["gold_short_answer"],
        "gold_reference_explanation": row["gold_reference_explanation"],
        "answer_mode": row.get("answer_mode", ""),
        "explanation_target": row.get("explanation_target", ""),
        "weighted_score": row["weighted_score"],
        "error_tags": row.get("error_tags", ""),
        "generator_model": row.get("generator_model", ""),
        "generation_mode": row.get("generation_mode", ""),
        "version_tag": row.get("version_tag", ""),
        "run_name": row.get("run_name", ""),
        "label_path": row.get("label_path", ""),
        "raw_path": row.get("raw_path", ""),
        "reuse_tier": row.get("reuse_tier", ""),
        "source_task": row.get("source_task", ""),
        "source_split": row.get("source_split", ""),
        "locked_split": row.get("locked_split", ""),
        "reuse_reason": row.get("reuse_reason", ""),
        "package_role": row.get("package_role", PACKAGE_ROLE),
        "batch_status": row.get("batch_status", CANDIDATE_BATCH_STATUS),
        "count_reflection_status": row.get("count_reflection_status", CANDIDATE_REFLECTION_STATUS),
        "downstream_consumption_allowed": row.get("downstream_consumption_allowed", NO),
        "count_allowed": row.get("count_allowed", NO),
        "count_disposition": row.get("count_disposition", COUNT_DISPOSITION),
        "promotion_contract_status": row.get("promotion_contract_status", PROMOTION_CONTRACT_STATUS),
        "eval_targeting_role": row.get("eval_targeting_role", ""),
        "eval_targeting_contract": row.get("eval_targeting_contract", ""),
        "split": row["split"],
    }


def write_compiled_outputs(compiled: dict[str, list[dict]], merged_rows: list[dict]) -> list[dict]:
    final_rows = compiled["final_rows"]
    manifest_rows = []
    for row in final_rows:
        manifest_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row.get("sampling_lane", ""),
                "reuse_tier": row.get("reuse_tier", ""),
                "source_task": row.get("source_task", ""),
                "source_split": row.get("source_split", ""),
                "locked_split": row.get("locked_split", ""),
                "reuse_reason": row.get("reuse_reason", ""),
                "split": row["split"],
                "dataset_disposition": row["split"],
                "train_eligible": row.get("train_eligible", YES),
                "audit_required": row.get("audit_required", NO),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
                "package_role": row.get("package_role", PACKAGE_ROLE),
                "batch_status": row.get("batch_status", CANDIDATE_BATCH_STATUS),
                "count_reflection_status": row.get("count_reflection_status", CANDIDATE_REFLECTION_STATUS),
                "downstream_consumption_allowed": row.get("downstream_consumption_allowed", NO),
                "count_allowed": row.get("count_allowed", NO),
                "count_disposition": row.get("count_disposition", COUNT_DISPOSITION),
                "promotion_contract_status": row.get("promotion_contract_status", PROMOTION_CONTRACT_STATUS),
                "eval_targeting_role": row.get("eval_targeting_role", ""),
                "eval_targeting_contract": row.get("eval_targeting_contract", ""),
            }
        )
    for split_name in ["train", "dev", "test"]:
        payloads = [final_payload(row) for row in final_rows if row["split"] == split_name]
        base.write_jsonl_atomic(PROCESSED_DIR / f"{split_name}.jsonl", payloads)
    base.write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["problem_id"])
    base.write_csv_atomic(PROBLEM_AUDIT_QUEUE_PATH, [], ["problem_id", "seed_sample_id", "family_id", "doc_type_name", "source_subset", "audit_reason"])
    for path, rows in [
        (CANDIDATE_POOL_PATH, compiled["candidate_pool"]),
        (ACCEPTED_POOL_PATH, compiled["accepted_pool"]),
        (REJECTED_POOL_PATH, compiled["rejected_pool"]),
        (TAIL_TAXONOMY_PATH, compiled["quality_tail"]),
        (QUOTA_SURPLUS_POOL_PATH, compiled["quota_surplus"]),
        (FINAL_PACKAGE_CSV_PATH, final_rows),
    ]:
        base.write_csv_atomic(path, rows, union_fields(rows) if rows else ["empty"])
    # merged score에는 raw candidate 전체와 final package contract를 같이 남겨 reviewer가 미선택 row를 추적할 수 있게 한다.
    final_ids = {row["candidate_id"] for row in final_rows}
    merged_with_contract = []
    final_counted = bool(final_rows) and all(row.get("count_allowed") == YES for row in final_rows)
    for row in merged_rows:
        selected = row["candidate_id"] in final_ids
        contract_fields = counted_contract_fields() if selected and final_counted else candidate_contract_fields()
        merged_with_contract.append(
            {
                **row,
                **contract_fields,
                "promotion_contract_status": contract_fields["promotion_contract_status"] if selected else "candidate_pool_not_promoted",
                "final_package_selected": YES if selected else NO,
            }
        )
    base.write_csv_atomic(MERGED_SCORES_PATH, merged_with_contract, union_fields(merged_with_contract))
    return manifest_rows


def render_markdown_outputs(compiled: dict[str, list[dict]], manifest_rows: list[dict]) -> None:
    final_rows = compiled["final_rows"]
    final_counted = bool(final_rows) and all(row.get("count_allowed") == YES for row in final_rows)
    doc_counts = Counter(row["doc_type_name"] for row in final_rows)
    source_counts = Counter(row["source_subset"] for row in final_rows)
    split_counts = Counter(row["split"] for row in final_rows)
    eval_doc_counts = Counter(row["doc_type_name"] for row in final_rows if row["split"] in {"dev", "test"})
    reuse_counts = Counter(row.get("reuse_tier", "") for row in final_rows)
    summary_rows = [
        {
            "metric": "candidate_total",
            "value": str(len(compiled["candidate_pool"])),
        },
        {
            "metric": "accepted_total",
            "value": str(len(compiled["accepted_pool"])),
        },
        {
            "metric": "final_package_total",
            "value": str(len(final_rows)),
        },
        {
            "metric": "quality_tail_total",
            "value": str(len(compiled["quality_tail"])),
        },
        {
            "metric": "quota_surplus_total",
            "value": str(len(compiled["quota_surplus"])),
        },
    ]
    base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    lines = [
        f"# descriptive production wave `{VERSION_TAG}`",
        "",
        "## package summary",
        f"- route_label: `{ACTIVE_ROUTE['route_label']}`",
        f"- candidate_total: `{len(compiled['candidate_pool'])}`",
        f"- accepted_total: `{len(compiled['accepted_pool'])}`",
        f"- final_package_total: `{len(final_rows)}`",
        f"- active_final_route: `{ACTIVE_ROUTE.get('active_final_route_label', 'primary_final')}`",
        f"- active_final_target_count: `{ACTIVE_ROUTE.get('active_final_target_count', ACTIVE_ROUTE['target_count'])}`",
        f"- source_balance_relaxation: `{ACTIVE_ROUTE.get('source_balance_relaxation', '없음')}`",
        f"- eval_targeting_contract: `{ACTIVE_ROUTE.get('eval_targeting_contract', '없음')}`",
        f"- quality_tail_total: `{len(compiled['quality_tail'])}`",
        f"- quota_surplus_total: `{len(compiled['quota_surplus'])}`",
        f"- split_counts: `{dict(split_counts)}`",
        f"- eval_doc_type_counts: `{dict(eval_doc_counts)}`",
        f"- reuse_tier_counts: `{dict(reuse_counts)}`",
        "",
        "## final doc type counts",
        "| doc_type | count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(doc_counts.items()):
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## final source counts", "| source_subset | count |", "| --- | ---: |"])
    for key, value in sorted(source_counts.items()):
        lines.append(f"| `{key}` | `{value}` |")
    write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    write_text_atomic(FINAL_PACKAGE_MD_PATH, "\n".join(lines) + "\n")

    evidence = [
        f"# evidence summary `{VERSION_TAG}`",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| route_name | `{ROUTE_NAME}` |",
        f"| package_role | `{COUNTED_PACKAGE_ROLE if final_counted else PACKAGE_ROLE}` |",
        f"| seed_policy | `Tier 0 fresh first + Tier 2 train-only split-lock fallback` |",
        f"| batch_status | `{COUNTED_BATCH_STATUS if final_counted else CANDIDATE_BATCH_STATUS}` |",
        f"| count_reflection_status | `{COUNTED_REFLECTION_STATUS if final_counted else CANDIDATE_REFLECTION_STATUS}` |",
        f"| downstream_consumption_allowed | `{YES if final_counted else NO}` |",
        f"| count_allowed | `{YES if final_counted else NO}` |",
        f"| candidate_total | `{len(compiled['candidate_pool'])}` |",
        f"| final_package_total | `{len(final_rows)}` |",
        f"| active_final_route | `{ACTIVE_ROUTE.get('active_final_route_label', 'primary_final')}` |",
        f"| active_final_target_count | `{ACTIVE_ROUTE.get('active_final_target_count', ACTIVE_ROUTE['target_count'])}` |",
        f"| eval_targeting_contract | `{ACTIVE_ROUTE.get('eval_targeting_contract', '없음')}` |",
        f"| eval_doc_type_counts | `{dict(eval_doc_counts)}` |",
        f"| hard_soft_audit_in_final | `0/0/0` |",
        f"| manifest_count | `{len(manifest_rows)}` |",
        f"| train/dev/test | `{split_counts.get('train', 0)}/{split_counts.get('dev', 0)}/{split_counts.get('test', 0)}` |",
        f"| decision | `{'counted_under_api_first_contract' if final_counted else 'reviewer_signoff_needed_before_count_reflection'}` |",
    ]
    write_text_atomic(EVIDENCE_SUMMARY_MD_PATH, "\n".join(evidence) + "\n")
    spec_lines = [
        f"# package spec `{VERSION_TAG}`",
        "",
        f"- medium route: `source-relaxed final 40 / candidate {sum(MEDIUM_RELAXED_SOURCE_COUNTS.values())}`",
        f"- constrained route: `primary final 24, fallback final 20 / candidate {sum(CONSTRAINED_SOURCE_COUNTS.values())}`",
        f"- active route: `{ACTIVE_ROUTE['route_label']}`",
        f"- source-balance relaxation: `{ACTIVE_ROUTE.get('source_balance_relaxation', '없음')}`",
        f"- eval-targeting contract: `{ACTIVE_ROUTE.get('eval_targeting_contract', '없음')}`",
        "- seed policy: Tier 0 fresh-only 우선, 부족분은 objective train family의 Tier 2 train-only split-lock reuse 허용",
        "- count rule: candidate 전체가 아니라 strict final package만 API-first contract 기준으로 count 후보",
        "",
    ]
    write_text_atomic(PACKAGE_SPEC_MD_PATH, "\n".join(spec_lines))


def write_manifest(seed_rows: list[dict], merged_rows: list[dict], manifest_rows: list[dict], compiled: dict[str, list[dict]]) -> dict:
    final_rows = compiled["final_rows"]
    split_counts = Counter(row.get("split", "") for row in final_rows)
    eval_doc_counts = Counter(row["doc_type_name"] for row in final_rows if row.get("split") in {"dev", "test"})
    reuse_counts = Counter(row.get("reuse_tier", "") for row in final_rows)
    active_final_target_count = int(ACTIVE_ROUTE.get("active_final_target_count", ACTIVE_ROUTE["target_count"]))
    success = len(final_rows) == active_final_target_count
    final_counted = success
    final_contract = counted_contract_fields() if final_counted else candidate_contract_fields()
    api_call_summary = {
        "openai_api": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "gemini_api": base.load_jsonl_count(GROUNDING_LOG_PATH)
        + base.load_jsonl_count(ANSWERABILITY_LOG_PATH)
        + base.load_jsonl_count(TASKFIT_LOG_PATH),
    }
    api_call_summary["total_api_calls"] = api_call_summary["openai_api"] + api_call_summary["gemini_api"]
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "route_name": ROUTE_NAME,
        "route_label": ACTIVE_ROUTE["route_label"],
        **final_contract,
        "seed_registry_strategy": "tier0_fresh_first_with_tier2_objective_train_split_locked_reuse",
        "seed_reuse_policy": {
            "tier0": "fresh-only source family without prior overlap",
            "tier2": "objective train family may be reused for descriptive train only under split-lock",
            "global_eval_holdout": "counted dev/test family remains blocked across tasks",
            "quality_tail_reuse": "blocked",
        },
        "seed_registry_count": len(seed_rows),
        "candidate_total": len(compiled["candidate_pool"]),
        "accepted_total": len(compiled["accepted_pool"]),
        "final_package_total": len(final_rows),
        "rejected_total": len(compiled["rejected_pool"]),
        "active_final_route": ACTIVE_ROUTE.get("active_final_route_label", "primary_final"),
        "active_final_target_count": active_final_target_count,
        "eval_targeting_contract": ACTIVE_ROUTE.get("eval_targeting_contract", ""),
        "eval_balanced_doc_types": ACTIVE_ROUTE.get("eval_balanced_doc_types", []),
        "eval_priority_doc_types": ACTIVE_ROUTE.get("eval_priority_doc_types", []),
        "eval_avoid_doc_types": ACTIVE_ROUTE.get("eval_avoid_doc_types", []),
        "eval_doc_type_counts": dict(eval_doc_counts),
        "quality_tail_total": len(compiled["quality_tail"]),
        "quota_surplus_total": len(compiled["quota_surplus"]),
        "generation_count": base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_answerability_count": base.load_jsonl_count(ANSWERABILITY_LOG_PATH),
        "judge_taskfit_count": base.load_jsonl_count(TASKFIT_LOG_PATH),
        "merged_count": base.load_csv_count(MERGED_SCORES_PATH),
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "split_counts": dict(split_counts),
        "reuse_tier_counts": dict(reuse_counts),
        "split_lock_eval_hotfix_status": "passed",
        "count_reflection_requires_reviewer_signoff": False,
        "success_criteria": {
            "primary_final_package_target": ACTIVE_ROUTE["target_count"],
            "fallback_final_package_target": ACTIVE_ROUTE.get("fallback_target_count"),
            "active_final_package_target": active_final_target_count,
            "hard_soft_audit_in_final": "0/0/0",
            # API-first stop line에서는 strict final package가 성공 기준을 통과하면
            # candidate 전체가 아니라 final package만 counted state로 잠근다.
            "count_reflection_requires_reviewer_signoff": False,
            "api_first_contract_counted_when_success": True,
        },
        "success_result": {
            "passed": success,
            "reason": "active final package target met" if success else "active final package target not met",
        },
        "api_call_summary": api_call_summary,
        "total_api_calls": api_call_summary["total_api_calls"],
        # reviewer-facing alias: emergency repeat에서는 생성/Judge worker 수가 stop line P2라 manifest에 직접 남긴다.
        "generation_main_max_workers": base.GENERATOR_MAIN_MAX_WORKERS,
        "generation_strict_max_workers": base.GENERATOR_STRICT_MAX_WORKERS,
        "generation_main_checkpoint_every": base.GENERATOR_MAIN_CHECKPOINT_EVERY,
        "judge_main_max_workers": base.JUDGE_MAIN_MAX_WORKERS,
        "judge_main_max_attempts": base.JUDGE_MAIN_MAX_ATTEMPTS,
        "judge_main_checkpoint_every": base.JUDGE_MAIN_CHECKPOINT_EVERY,
        # linter/evidence 실행 직후 sync step에서 true로 전환한다. false를 명시해 handoff alias 누락을 방지한다.
        "artifact_linter_passed": False,
        "evidence_card_passed": False,
        "evidence_card_all_green": False,
        "artifact_paths": {
            "seed_registry": repo_rel(SEED_REGISTRY_PATH),
            "seed_ready": repo_rel(SEED_READY_PATH),
            "seed_preflight": repo_rel(SEED_PREFLIGHT_MD_PATH),
            "generated_problems": repo_rel(GENERATED_PROBLEMS_PATH),
            "merged_scores": repo_rel(MERGED_SCORES_PATH),
            "final_package": repo_rel(FINAL_PACKAGE_CSV_PATH),
            "candidate_pool": repo_rel(CANDIDATE_POOL_PATH),
            "accepted_pool": repo_rel(ACCEPTED_POOL_PATH),
            "rejected_pool": repo_rel(REJECTED_POOL_PATH),
            "tail_taxonomy": repo_rel(TAIL_TAXONOMY_PATH),
            "quota_surplus_pool": repo_rel(QUOTA_SURPLUS_POOL_PATH),
            "compiler_manifest": repo_rel(COMPILER_MANIFEST_PATH),
            "seed_availability_map": repo_rel(AVAILABILITY_MAP_MD_PATH),
            "route_feasibility": repo_rel(AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH),
            "problem_dataset_manifest": repo_rel(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_train": repo_rel(PROBLEM_TRAIN_PATH),
            "problem_dev": repo_rel(PROBLEM_DEV_PATH),
            "problem_test": repo_rel(PROBLEM_TEST_PATH),
            "evidence_summary": repo_rel(EVIDENCE_SUMMARY_MD_PATH),
        },
    }
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    base.write_json_atomic(
        COMPILER_MANIFEST_PATH,
        {
            "compiler_manifest_version": "descriptive_production_wave_v1",
            "package_role": final_contract["package_role"],
            "batch_status": final_contract["batch_status"],
            "route_label": ACTIVE_ROUTE["route_label"],
            "selection_policy": "pass/no-audit accepted rows selected by final source quota and weighted score",
            "row_counts": {
                "candidate_total": len(compiled["candidate_pool"]),
                "accepted_total": len(compiled["accepted_pool"]),
                "final_package_total": len(final_rows),
                "quality_tail_total": len(compiled["quality_tail"]),
                "quota_surplus_total": len(compiled["quota_surplus"]),
            },
            "count_reflection_status": final_contract["count_reflection_status"],
            "count_allowed": final_contract["count_allowed"],
            "count_reflection_requires_reviewer_signoff": False,
            "downstream_consumption_allowed": final_contract["downstream_consumption_allowed"],
            "split_lock_eval_hotfix_status": "passed",
            "api_call_summary": api_call_summary,
            "total_api_calls": api_call_summary["total_api_calls"],
            "generation_main_max_workers": base.GENERATOR_MAIN_MAX_WORKERS,
            "generation_strict_max_workers": base.GENERATOR_STRICT_MAX_WORKERS,
            "generation_main_checkpoint_every": base.GENERATOR_MAIN_CHECKPOINT_EVERY,
            "judge_main_max_workers": base.JUDGE_MAIN_MAX_WORKERS,
            "judge_main_max_attempts": base.JUDGE_MAIN_MAX_ATTEMPTS,
            "judge_main_checkpoint_every": base.JUDGE_MAIN_CHECKPOINT_EVERY,
            "artifact_linter_passed": False,
            "evidence_card_passed": False,
            "evidence_card_all_green": False,
            "active_final_route_label": ACTIVE_ROUTE.get("active_final_route_label", "primary_final"),
            "active_final_target_count": active_final_target_count,
            "eval_targeting_contract": ACTIVE_ROUTE.get("eval_targeting_contract", ""),
            "eval_doc_type_counts": dict(eval_doc_counts),
            "success_result": manifest["success_result"],
        },
    )
    return manifest


def write_preflight_manifest(seed_rows: list[dict]) -> dict:
    # preflight-only run도 reviewer가 stale path 없이 검산할 수 있도록 최소 manifest를 남긴다.
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "route_name": ROUTE_NAME,
        "route_label": ACTIVE_ROUTE["route_label"],
        "package_role": "preflight_only",
        "batch_status": "preflight_only_not_counted",
        "count_reflection_status": "not_counted_preflight_only",
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": "preflight_only_not_counted",
        "promotion_contract_status": "preflight_only",
        "seed_registry_strategy": "tier0_fresh_first_with_tier2_objective_train_split_locked_reuse",
        "seed_registry_count": len(seed_rows),
        "candidate_total": 0,
        "accepted_total": 0,
        "final_package_total": 0,
        "rejected_total": 0,
        "quality_tail_total": 0,
        "quota_surplus_total": 0,
        "active_final_route": ACTIVE_ROUTE.get("active_final_route_label", f"primary_final{ACTIVE_ROUTE['target_count']}"),
        "active_final_target_count": ACTIVE_ROUTE["target_count"],
        "eval_targeting_contract": ACTIVE_ROUTE.get("eval_targeting_contract", ""),
        "eval_balanced_doc_types": ACTIVE_ROUTE.get("eval_balanced_doc_types", []),
        "eval_priority_doc_types": ACTIVE_ROUTE.get("eval_priority_doc_types", []),
        "eval_avoid_doc_types": ACTIVE_ROUTE.get("eval_avoid_doc_types", []),
        "generation_main_max_workers": base.GENERATOR_MAIN_MAX_WORKERS,
        "generation_strict_max_workers": base.GENERATOR_STRICT_MAX_WORKERS,
        "generation_main_checkpoint_every": base.GENERATOR_MAIN_CHECKPOINT_EVERY,
        "judge_main_max_workers": base.JUDGE_MAIN_MAX_WORKERS,
        "judge_main_max_attempts": base.JUDGE_MAIN_MAX_ATTEMPTS,
        "judge_main_checkpoint_every": base.JUDGE_MAIN_CHECKPOINT_EVERY,
        "success_result": {
            "passed": len(seed_rows) == sum(ACTIVE_ROUTE["source_counts"].values()),
            "reason": "preflight source schedule satisfied",
        },
        "artifact_paths": {
            "seed_registry": repo_rel(SEED_REGISTRY_PATH),
            "seed_ready": repo_rel(SEED_READY_PATH),
            "seed_preflight": repo_rel(SEED_PREFLIGHT_MD_PATH),
            "seed_availability_map": repo_rel(AVAILABILITY_MAP_MD_PATH),
            "route_feasibility": repo_rel(AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH),
        },
    }
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def write_preflight_blocker_manifest(error: Exception) -> dict:
    # route가 모두 막힌 경우에도 crash만 남기지 않고, reviewer가 다음 API route를 판단할 수 있는 blocker manifest를 남긴다.
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": base.utc_now_iso(),
        "route_name": ROUTE_NAME,
        "package_role": "preflight_blocked",
        "batch_status": "preflight_blocked_seed_availability",
        "count_reflection_status": "not_counted_preflight_blocked",
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": "preflight_blocked_not_counted",
        "promotion_contract_status": "preflight_blocked",
        "seed_registry_count": 0,
        "candidate_total": 0,
        "accepted_total": 0,
        "final_package_total": 0,
        "rejected_total": 0,
        "quality_tail_total": 0,
        "quota_surplus_total": 0,
        "generation_main_max_workers": base.GENERATOR_MAIN_MAX_WORKERS,
        "generation_strict_max_workers": base.GENERATOR_STRICT_MAX_WORKERS,
        "generation_main_checkpoint_every": base.GENERATOR_MAIN_CHECKPOINT_EVERY,
        "judge_main_max_workers": base.JUDGE_MAIN_MAX_WORKERS,
        "judge_main_max_attempts": base.JUDGE_MAIN_MAX_ATTEMPTS,
        "judge_main_checkpoint_every": base.JUDGE_MAIN_CHECKPOINT_EVERY,
        "success_result": {
            "passed": False,
            "reason": str(error),
        },
        "artifact_paths": {
            "seed_availability_map": repo_rel(AVAILABILITY_MAP_MD_PATH),
            "route_feasibility": repo_rel(AVAILABILITY_ROUTE_FEASIBILITY_CSV_PATH),
        },
    }
    base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def main() -> dict:
    configure_base_paths()
    try:
        seed_rows = build_seed_registry()
    except RuntimeError as exc:
        if "descriptive_seed_availability_blocker" not in str(exc):
            raise
        manifest = write_preflight_blocker_manifest(exc)
        print(f"[descriptive wave] preflight_blocked reason={manifest['success_result']['reason']}", flush=True)
        return manifest
    print(f"[descriptive wave] route={ACTIVE_ROUTE['route_label']} candidates={len(seed_rows)}", flush=True)
    if os.environ.get("DESCRIPTIVE_WAVE_PREFLIGHT_ONLY") == "1":
        manifest = write_preflight_manifest(seed_rows)
        print(f"[descriptive wave] preflight_only success={manifest['success_result']['passed']}", flush=True)
        return manifest
    base.run_generation(mode="main")
    base.run_generation(mode="strict_finalize")
    base.run_judges(mode="main")
    base.run_judges(mode="strict_finalize")
    merged_rows = base.merge_scores()
    compiled = compile_final_package(merged_rows)
    manifest_rows = write_compiled_outputs(compiled, merged_rows)
    render_markdown_outputs(compiled, manifest_rows)
    manifest = write_manifest(seed_rows, merged_rows, manifest_rows, compiled)
    print(f"[descriptive wave] final_package={manifest['final_package_total']} success={manifest['success_result']['passed']}", flush=True)
    return manifest


ACTIVE_ROUTE: dict = {}


if __name__ == "__main__":
    main()
