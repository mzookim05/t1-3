from __future__ import annotations

import sys
from pathlib import Path

# descriptive split-lock wave가 막힌 같은 stop line 안에서 count를 움직이기 위한 결정례 add-on 실행기다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_medium_overgeneration_pilot as medium,
)
from scripts.aihub.problem_generation.production_batches.run_objective_decision_addon_overgeneration_preflight import (  # noqa: E402
    DECISION_ADDON_FINAL_SOURCE_COUNTS,
    DECISION_ADDON_SOURCE_COUNTS,
)


def build_decision_addon_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = medium.build_decision_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## decision add-on overgeneration 추가 지시
- 이번 run은 `결정례_QA` candidate {medium.base.EXPECTED_CANDIDATE_SEED_COUNT}개에서 strict final {medium.base.FINAL_PACKAGE_TARGET_COUNT}개만 컴파일하는 add-on fallback이다.
- `062200` counted package 이후 부족량 보충용이므로, quota surplus와 quality tail은 final package 밖에 보존하고 count 후보로 올리지 않는다.
- final package는 정답 유일성, weak distractor 분리, label/source/lane quota를 모두 통과한 row만 포함한다.
"""
    return messages


def configure_decision_addon_overgeneration() -> None:
    medium.configure_decision_medium_overgeneration()
    medium.base.VERSION_TAG = "objective_decision_addon_overgeneration_pilot"
    medium.base.RUN_DATE = build_run_stamp()
    medium.base.RUN_PURPOSE = "objective_r2_decision_target24_candidate40_api_execution"
    medium.base.RUN_NAME = f"{medium.base.RUN_DATE}_{medium.base.VERSION_TAG}_{medium.base.RUN_PURPOSE}"
    medium.base.RUN_LABEL = "objective decision add-on overgeneration API execution"
    medium.base.LINTER_FIXTURE_ID = "decision_addon_overgeneration_candidate_package_pass"
    medium.base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_decision_addon_overgeneration_preflight"
    medium.base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_decision_target24_candidate40_seed_spec_wiring_check"
    medium.base.SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = (
        medium.base.PROJECT_ROOT
        / "data"
        / "interim"
        / "aihub"
        / "problem_generation"
        / "production_batches"
        / medium.base.SOURCE_PREFLIGHT_VERSION_TAG
        / "seed_registry.csv"
    )
    medium.base.EXPECTED_CANDIDATE_SEED_COUNT = 40
    medium.base.FINAL_PACKAGE_TARGET_COUNT = 24
    medium.base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 40}
    medium.base.EXPECTED_LANE_BY_DOC = {
        ("결정례_QA", "generalization_03_04"): 20,
        ("결정례_QA", "expansion_01_02"): 20,
    }
    medium.base.EXPECTED_SOURCE_COUNTS = DECISION_ADDON_SOURCE_COUNTS
    medium.base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    medium.base.FINAL_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    medium.base.FINAL_SOURCE_COUNTS = DECISION_ADDON_FINAL_SOURCE_COUNTS
    medium.base.FINAL_LANE_COUNTS = {"generalization_03_04": 12, "expansion_01_02": 12}
    medium.base.FINAL_DEV_COUNT = 1
    medium.base.FINAL_TEST_COUNT = 1
    medium.base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 303,
        "train": 256,
        "eval": 47,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    medium.base.build_generation_messages = build_decision_addon_generation_messages
    medium.base.refresh_paths()


def main() -> dict:
    configure_decision_addon_overgeneration()
    return medium.base.main()


if __name__ == "__main__":
    main()
