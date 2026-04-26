from __future__ import annotations

import sys
from pathlib import Path

# constrained preflight가 고정한 40개 해석례_QA candidate seed를 실제 API로 실행한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_small_overgeneration_pilot as base,
)


def configure_constrained_overgeneration() -> None:
    # medium 64개가 seed availability에서 막힌 상황에서도 API-first stop line을 유지하기 위한 fallback 실행이다.
    base.VERSION_TAG = "objective_interpretation_constrained_overgeneration_pilot"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_target24_candidate40_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective interpretation constrained overgeneration API execution"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_interpretation_constrained_overgeneration_pilot_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_interpretation_target24_candidate40_seed_spec_wiring_check"
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

    # strict final 24개만 count reflection 후보로 두고, candidate/tail/surplus는 미합산으로 보존한다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 40
    base.FINAL_PACKAGE_TARGET_COUNT = 24
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 40}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "generalization_03_04"): 20,
        ("해석례_QA", "expansion_01_02"): 20,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 10,
        "02_TL_유권해석_QA": 10,
        "03_TL_해석례_QA": 10,
        "04_TL_해석례_QA": 10,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 6,
        "02_TL_유권해석_QA": 6,
        "03_TL_해석례_QA": 6,
        "04_TL_해석례_QA": 6,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 12, "expansion_01_02": 12}
    base.FINAL_DEV_COUNT = 1
    base.FINAL_TEST_COUNT = 1

    # reviewer-visible count reflection 전까지 current objective count는 기존 239를 유지한다.
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 239,
        "train": 196,
        "eval": 43,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.refresh_paths()


def main() -> dict:
    configure_constrained_overgeneration()
    return base.main()


if __name__ == "__main__":
    main()
