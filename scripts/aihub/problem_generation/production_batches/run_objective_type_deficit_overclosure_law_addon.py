from __future__ import annotations

import sys
from pathlib import Path

# preflight가 잠근 법령_QA final 32 add-on seed registry를 그대로 사용해
# strict final package만 reviewer sign-off 후보로 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_law_source_relaxed_repair_superwave as law_source_relaxed,
)


def configure_type_deficit_overclosure_law_addon() -> None:
    law_source_relaxed.configure_law_source_relaxed_repair_superwave()
    base = law_source_relaxed.medium.base

    base.VERSION_TAG = "objective_type_deficit_overclosure_law_addon"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_law_type_deficit_candidate64_final32_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective type-deficit overclosure law add-on API execution"
    base.LINTER_FIXTURE_ID = "objective_type_deficit_overclosure_law_addon_candidate_package_pass"
    base.GATE_FIELD_LABEL = "law type-deficit overclosure gate fields"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_type_deficit_overclosure_law_addon_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_law_type_deficit_candidate64_final32_preflight"
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
    base.FINAL_PACKAGE_TARGET_COUNT = 32
    base.EXPECTED_DOC_TYPE_COUNTS = {"법령_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {("법령_QA", "generalization_03_04"): 64}
    base.EXPECTED_SOURCE_COUNTS = {
        "03_TL_법령_QA": 32,
        "04_TL_법령_QA": 32,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 8, "B": 8, "C": 8, "D": 8}
    base.FINAL_SOURCE_COUNTS = {
        "03_TL_법령_QA": 16,
        "04_TL_법령_QA": 16,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 32}
    base.FINAL_DEV_COUNT = 2
    base.FINAL_TEST_COUNT = 2
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
    configure_type_deficit_overclosure_law_addon()
    return law_source_relaxed.medium.base.main()


if __name__ == "__main__":
    main()
