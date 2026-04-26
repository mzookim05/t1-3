from __future__ import annotations

import sys
from pathlib import Path

# descriptive production wave가 seed availability에서 막힌 경우의 API-first fallback이다.
# 판결문 medium package factory 실행기를 재사용하되, 결정례_QA seed/preflight와 decision-specific prompt로 재배선한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_small_overgeneration_pilot as base,
)


def build_decision_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = base.judgment_pilot.BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += f"""

## decision medium overgeneration pilot 추가 지시
- 이번 run은 `결정례_QA` candidate {base.EXPECTED_CANDIDATE_SEED_COUNT}개를 생성한 뒤 strict final package {base.FINAL_PACKAGE_TARGET_COUNT}개만 컴파일하는 fallback package factory pilot이다.
- 생성 단계에서는 label 위치보다 정답 유일성, 하나의 쟁점, 선택지 의미 분리를 우선한다.
- stem은 하나의 판단 기준, 하나의 적용 사실, 하나의 절차 효과 중 하나만 묻는다.
- 결정례의 이유와 결론을 한 stem 안에서 동시에 묻지 않는다.
- 정답은 `gold_short_answer`와 같은 결정례상 결론 하나에만 닫혀야 한다.
- 오답은 같은 결정례 근거를 공유하되 각각 요건, 절차, 효과, 예외, 적용 범위 중 정확히 한 축만 어긋나야 한다.
- 다른 choice가 별도 판단 요소나 별도 절차 설명으로도 정답 가능하게 읽히면 answer uniqueness failure로 본다.
- 후처리 validator가 target label `{seed.get('target_correct_choice', '')}`로 choice를 재배치하므로, 생성 단계에서는 target label을 억지로 맞추지 않는다.
"""
    return messages


def configure_decision_medium_overgeneration() -> None:
    base.VERSION_TAG = "objective_decision_medium_overgeneration_pilot"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_decision_target40_candidate64_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective decision medium overgeneration API execution"
    base.LINTER_FIXTURE_ID = "decision_medium_overgeneration_candidate_package_pass"
    base.GATE_FIELD_LABEL = "decision gate fields"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_decision_medium_overgeneration_pilot_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_decision_target40_candidate64_seed_spec_wiring_check"
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
    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("결정례_QA", "generalization_03_04"): 32,
        ("결정례_QA", "expansion_01_02"): 32,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_심결례_QA": 12,
        "02_TL_심결례_QA": 10,
        "02_TL_심결문_QA": 10,
        "03_TL_결정례_QA": 16,
        "04_TL_결정례_QA": 16,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_심결례_QA": 8,
        "02_TL_심결례_QA": 6,
        "02_TL_심결문_QA": 6,
        "03_TL_결정례_QA": 10,
        "04_TL_결정례_QA": 10,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}
    base.FINAL_DEV_COUNT = 1
    base.FINAL_TEST_COUNT = 1
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 263,
        "train": 218,
        "eval": 45,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.build_generation_messages = build_decision_generation_messages
    base.refresh_paths()


def main() -> dict:
    configure_decision_medium_overgeneration()
    return base.main()


if __name__ == "__main__":
    main()
