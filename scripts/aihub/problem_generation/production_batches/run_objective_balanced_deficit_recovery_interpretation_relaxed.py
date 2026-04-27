from __future__ import annotations

import sys
from pathlib import Path

# source-relaxed preflight가 고정한 해석례_QA candidate를 final 40 package로 컴파일한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_small_overgeneration_pilot as base,
)


def configure_balanced_deficit_recovery_interpretation_relaxed() -> None:
    # medium wrapper는 기존 strict preflight를 먼저 찾으므로, source-relaxed contract에서는 small base를 직접 재배선한다.
    base.VERSION_TAG = "objective_balanced_deficit_recovery_interpretation_relaxed"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_source_relaxed_candidate64_final40_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery interpretation source-relaxed API execution"
    base.LINTER_FIXTURE_ID = "balanced_deficit_recovery_interpretation_relaxed_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_balanced_deficit_recovery_interpretation_relaxed_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_interpretation_source_relaxed_candidate64_final40_preflight"
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
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "expansion_01_02"): 32,
        ("해석례_QA", "generalization_03_04"): 32,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 32,
        "03_TL_해석례_QA": 32,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 20,
        "03_TL_해석례_QA": 20,
    }
    base.FINAL_LANE_COUNTS = {"expansion_01_02": 20, "generalization_03_04": 20}
    base.FINAL_DEV_COUNT = 5
    base.FINAL_TEST_COUNT = 5
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 407,
        "train": 338,
        "eval": 69,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    # 해석례 repair pilot 원형은 16개 고정 registry를 가정하므로, source-relaxed 64개 registry 검산값을 같이 덮어쓴다.
    base.interpretation_pilot.EXPECTED_TOTAL_SEED_COUNT = 64
    base.interpretation_pilot.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 64}
    base.interpretation_pilot.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "expansion_01_02"): 32,
        ("해석례_QA", "generalization_03_04"): 32,
    }
    base.interpretation_pilot.EXPECTED_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 32,
        "03_TL_해석례_QA": 32,
    }
    base.interpretation_pilot.TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS = 16
    base.pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS = 16
    base.refresh_paths()
    # r2 postprocess 기준 patch가 새 run의 미생성 merged를 찾지 않게, 신규 source-relaxed run에서는 fallback map을 비운다.
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_balanced_deficit_recovery_interpretation_relaxed()
    return base.main()


if __name__ == "__main__":
    main()
