from __future__ import annotations

import sys
from pathlib import Path

# medium preflight가 고정한 64개 해석례_QA candidate seed를 실제 API로 태우는 execution wrapper다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_small_overgeneration_pilot as base,
)


def configure_medium_overgeneration() -> None:
    # small 해석례 API runner의 validator/compiler를 재사용하되 target 40 / candidate 64 quota로 확장한다.
    base.VERSION_TAG = "objective_interpretation_medium_overgeneration_pilot"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_target40_candidate64_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective interpretation medium overgeneration API execution"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_interpretation_medium_overgeneration_pilot_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_interpretation_target40_candidate64_seed_spec_wiring_check"
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

    # strict final 40개만 count reflection 후보가 되고, quality tail과 quota surplus는 final 밖에 보존된다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "generalization_03_04"): 32,
        ("해석례_QA", "expansion_01_02"): 32,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 16,
        "02_TL_유권해석_QA": 16,
        "03_TL_해석례_QA": 16,
        "04_TL_해석례_QA": 16,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 10,
        "02_TL_유권해석_QA": 10,
        "03_TL_해석례_QA": 10,
        "04_TL_해석례_QA": 10,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}
    base.FINAL_DEV_COUNT = 1
    base.FINAL_TEST_COUNT = 1

    # 이번 API route는 reviewer-visible evidence 전까지 current count를 유지하는 candidate package다.
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
    configure_medium_overgeneration()
    return base.main()


if __name__ == "__main__":
    main()
