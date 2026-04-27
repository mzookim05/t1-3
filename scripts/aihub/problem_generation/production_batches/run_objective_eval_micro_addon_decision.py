from __future__ import annotations

import sys
from pathlib import Path

# eval micro add-on은 final 12건 전체를 dev/test로 배치해 objective eval closure만 좁게 보강한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_balanced_deficit_recovery_decision_eval_addon as addon,
)
from scripts.aihub.problem_generation.production_batches.run_objective_eval_micro_addon_decision_preflight import (  # noqa: E402
    MICRO_FINAL_SOURCE_COUNTS,
    MICRO_SOURCE_COUNTS,
)


def configure_eval_micro_addon_decision() -> None:
    addon.configure_balanced_deficit_recovery_decision_eval_addon()
    base = addon.addon.medium.base
    base.VERSION_TAG = "objective_eval_micro_addon_decision"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_decision_candidate24_final12_eval_micro_addon_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective eval micro add-on decision API execution"
    base.LINTER_FIXTURE_ID = "objective_eval_micro_addon_decision_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_eval_micro_addon_decision_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_decision_candidate24_final12_eval_micro_addon_preflight"
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
    base.EXPECTED_CANDIDATE_SEED_COUNT = 24
    base.FINAL_PACKAGE_TARGET_COUNT = 12
    # preflight와 API runner가 같은 micro candidate universe를 검산하도록 doc/lane expected 값을 같이 낮춘다.
    base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 24}
    base.EXPECTED_LANE_BY_DOC = {
        # preflight source schedule과 동일하게 candidate lane은 13/11이고, final package만 6/6으로 균형 조립한다.
        ("결정례_QA", "expansion_01_02"): 13,
        ("결정례_QA", "generalization_03_04"): 11,
    }
    base.EXPECTED_SOURCE_COUNTS = MICRO_SOURCE_COUNTS
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 3, "B": 3, "C": 3, "D": 3}
    base.FINAL_SOURCE_COUNTS = MICRO_FINAL_SOURCE_COUNTS
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 6, "expansion_01_02": 6}
    base.FINAL_DEV_COUNT = 6
    base.FINAL_TEST_COUNT = 6
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
    base.refresh_paths()
    # micro run도 신규 merged 파일이 생기기 전까지 reference patch map을 비운다.
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_eval_micro_addon_decision()
    return addon.addon.medium.base.main()


if __name__ == "__main__":
    main()
