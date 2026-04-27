from __future__ import annotations

import sys
from pathlib import Path

# preflight가 고정한 판결문_QA final 8 eval-only seed를 실제 API로 태운다.
# 이 package는 candidate layer로 남겨 reviewer sign-off 뒤에만 count reflection 된다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_pilot as medium,
)


def configure_type_deficit_overclosure_judgment_eval_micro() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base

    base.VERSION_TAG = "objective_type_deficit_overclosure_judgment_eval_micro"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_judgment_eval_micro_candidate16_final8_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective type-deficit overclosure judgment eval micro API execution"
    base.LINTER_FIXTURE_ID = "objective_type_deficit_overclosure_judgment_eval_micro_candidate_package_pass"
    base.GATE_FIELD_LABEL = "judgment eval micro gate fields"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_type_deficit_overclosure_judgment_eval_micro_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_judgment_eval_micro_candidate16_final8_preflight"
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

    base.EXPECTED_CANDIDATE_SEED_COUNT = 16
    base.FINAL_PACKAGE_TARGET_COUNT = 8
    base.EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 16}
    base.EXPECTED_LANE_BY_DOC = {
        ("판결문_QA", "expansion_01_02"): 8,
        ("판결문_QA", "generalization_03_04"): 8,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 4,
        "02_TL_판결문_QA": 4,
        "03_TL_판결문_QA": 4,
        "04_TL_판결문_QA": 4,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 2,
        "02_TL_판결문_QA": 2,
        "03_TL_판결문_QA": 2,
        "04_TL_판결문_QA": 2,
    }
    base.FINAL_LANE_COUNTS = {"expansion_01_02": 4, "generalization_03_04": 4}
    base.FINAL_DEV_COUNT = 4
    base.FINAL_TEST_COUNT = 4
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 683,
        "train": 524,
        "eval": 159,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.refresh_paths()


def main() -> dict:
    configure_type_deficit_overclosure_judgment_eval_micro()
    return medium.base.main()


if __name__ == "__main__":
    main()
