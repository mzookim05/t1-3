from __future__ import annotations

import sys
from pathlib import Path

# preflight가 고정한 판결문_QA 안정 repeat candidate를 eval-aware final package로 컴파일한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_pilot as medium,
)


def configure_balanced_deficit_recovery_judgment() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base
    base.VERSION_TAG = "objective_balanced_deficit_recovery_judgment"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_judgment_candidate64_final40_eval_aware_repeat_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery judgment API execution"
    base.LINTER_FIXTURE_ID = "balanced_deficit_recovery_judgment_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_balanced_deficit_recovery_judgment_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_judgment_candidate64_final40_eval_aware_repeat_preflight"
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
    base.pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS = 16
    base.pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS = 16
    base.refresh_paths()
    # r2 postprocess가 새 run의 아직 생성되지 않은 merged 파일을 참조하지 않도록 기준 patch lookup을 비활성화한다.
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_balanced_deficit_recovery_judgment()
    return medium.base.main()


if __name__ == "__main__":
    main()
