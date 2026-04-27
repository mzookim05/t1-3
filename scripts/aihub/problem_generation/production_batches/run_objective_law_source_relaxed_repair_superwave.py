from __future__ import annotations

import sys
from pathlib import Path

# source-relaxed preflight가 고정한 법령_QA 03/04 candidate 192개를 실제 API로 태운 뒤,
# package compiler가 strict final 120개만 count reflection candidate로 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_pilot as medium,
)


def configure_law_source_relaxed_repair_superwave() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base

    base.VERSION_TAG = "objective_law_source_relaxed_repair_superwave"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_law_03_04_candidate192_final120_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective law source-relaxed repair superwave API execution"
    base.LINTER_FIXTURE_ID = "law_source_relaxed_repair_superwave_candidate_package_pass"
    base.GATE_FIELD_LABEL = "law source-relaxed repair gate fields"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_law_source_relaxed_repair_superwave_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_law_03_04_candidate192_final120_preflight"
    base.SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = (
        base.PROJECT_ROOT
        / "data"
        / "interim"
        / "aihub"
        / "problem_generation"
        / "production_batches"
        / base.SOURCE_PREFLIGHT_VERSION_TAG
        / "seed_registry.csv"
    )

    base.EXPECTED_CANDIDATE_SEED_COUNT = 192
    base.FINAL_PACKAGE_TARGET_COUNT = 120
    base.EXPECTED_DOC_TYPE_COUNTS = {"법령_QA": 192}
    base.EXPECTED_LANE_BY_DOC = {("법령_QA", "generalization_03_04"): 192}
    base.EXPECTED_SOURCE_COUNTS = {
        "03_TL_법령_QA": 96,
        "04_TL_법령_QA": 96,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 48, "B": 48, "C": 48, "D": 48}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 30, "B": 30, "C": 30, "D": 30}
    base.FINAL_SOURCE_COUNTS = {
        "03_TL_법령_QA": 60,
        "04_TL_법령_QA": 60,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 120}
    base.FINAL_DEV_COUNT = 15
    base.FINAL_TEST_COUNT = 15
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 511,
        "train": 410,
        "eval": 101,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS = 16
    base.pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS = 16

    def build_law_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
        messages = base.judgment_pilot.BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
        messages[1]["content"] += f"""

## 법령_QA source-relaxed repair superwave 추가 지시
- 이번 run은 `03/04_TL_법령_QA` 중심 candidate 192개 중 strict final 120개만 조립하는 package factory run이다.
- stem은 하나의 predicate만 묻는다. `요건`, `효과`, `절차`, `적용범위`, `예외` 중 하나만 선택한다.
- 목적 조항, 기관명, 숫자 경계만 맞히는 단순회상형을 피하고, 법적 적용 축 하나를 비교하게 만든다.
- 정답은 `gold_short_answer`의 법령상 결론 하나에만 닫혀야 하며, 다른 choice가 같은 결론으로 읽히면 실패다.
- 오답은 같은 조문 또는 같은 법적 anchor를 공유하되, 요건·효과·범위·예외 중 정확히 한 축만 어긋나야 한다.
- 오답끼리도 의미가 중복되면 안 되고, 상하위 포함 관계로 인해 둘 다 정답처럼 보이면 안 된다.
- 후처리 validator가 target label `{seed.get('target_correct_choice', '')}`로 choice를 재배치하므로, 생성 단계에서는 label 위치보다 정답 유일성을 우선한다.
"""
        return messages

    base.build_generation_messages = build_law_generation_messages
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}
    base.refresh_paths()


def main() -> dict:
    configure_law_source_relaxed_repair_superwave()
    return medium.base.main()


if __name__ == "__main__":
    main()
