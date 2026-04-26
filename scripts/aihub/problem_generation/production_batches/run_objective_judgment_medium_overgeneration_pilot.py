from __future__ import annotations

import sys
from pathlib import Path

# reviewer가 승인한 `034638` preflight seed registry를 그대로 고정해 target 40 / candidate 64 API execution을 수행한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_small_overgeneration_pilot as base,
)


def configure_medium_overgeneration() -> None:
    # 16개 package 반복 병목을 줄이되, count에는 strict final 40개 package만 올릴 수 있게 same-run compiler를 쓴다.
    base.VERSION_TAG = "objective_judgment_medium_overgeneration_pilot"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_judgment_target40_candidate64_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective judgment medium overgeneration API execution"
    base.LINTER_FIXTURE_ID = "judgment_medium_overgeneration_candidate_package_pass"
    base.GATE_FIELD_LABEL = "judgment gate fields"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_judgment_medium_overgeneration_pilot_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_judgment_target40_candidate64_seed_spec_wiring_check"
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

    # candidate pool과 final package quota를 분리해 실패 row를 final count 후보에서 자동 제외한다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("판결문_QA", "generalization_03_04"): 32,
        ("판결문_QA", "expansion_01_02"): 32,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 16,
        "02_TL_판결문_QA": 16,
        "03_TL_판결문_QA": 16,
        "04_TL_판결문_QA": 16,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 10,
        "02_TL_판결문_QA": 10,
        "03_TL_판결문_QA": 10,
        "04_TL_판결문_QA": 10,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}
    base.FINAL_DEV_COUNT = 1
    base.FINAL_TEST_COUNT = 1

    # 이번 API execution 전 current count는 `199 / 158 / 41`로 고정하고, reviewer sign-off 전에는 미합산 상태를 유지한다.
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 199,
        "train": 158,
        "eval": 41,
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
