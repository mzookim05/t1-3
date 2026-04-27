from __future__ import annotations

import sys
from pathlib import Path

# 해석례 repeat API runner는 기존 source-relaxed recipe를 그대로 쓰고,
# reviewer가 권장한 eval-heavy split만 덮어써 새 run의 의미를 분리한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_balanced_deficit_recovery_interpretation_relaxed as relaxed,
)


def configure_interpretation_source_relaxed_repeat() -> None:
    relaxed.configure_balanced_deficit_recovery_interpretation_relaxed()
    base = relaxed.base
    base.VERSION_TAG = "objective_interpretation_source_relaxed_repeat"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_source_relaxed_candidate64_final40_eval_heavy_repeat_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective interpretation source-relaxed repeat eval-heavy API execution"
    base.LINTER_FIXTURE_ID = "interpretation_source_relaxed_repeat_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_interpretation_source_relaxed_repeat_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = (
        "objective_r2_interpretation_source_relaxed_candidate64_final40_eval_heavy_repeat_preflight"
    )
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
    base.FINAL_DEV_COUNT = 8
    base.FINAL_TEST_COUNT = 8
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
    # 새 repeat run은 자기 자신의 merged output이 아직 없으므로 r2 reference patch lookup을 비운다.
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_interpretation_source_relaxed_repeat()
    return relaxed.base.main()


if __name__ == "__main__":
    main()
