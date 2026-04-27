from __future__ import annotations

import sys
from pathlib import Path

# 해석례 repeat는 이미 통과한 source-relaxed route를 재사용하되,
# 이번 objective recovery에서는 eval 부족을 더 직접 줄이기 위해 final split을 24/8/8로 바꾼다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_balanced_deficit_recovery_interpretation_relaxed_preflight as relaxed,
)


def configure_interpretation_source_relaxed_repeat_preflight() -> None:
    relaxed.configure_balanced_deficit_recovery_interpretation_relaxed_preflight()
    base = relaxed.medium.base
    base.VERSION_TAG = "objective_interpretation_source_relaxed_repeat_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_source_relaxed_candidate64_final40_eval_heavy_repeat_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective interpretation source-relaxed repeat eval-heavy preflight"
    base.SEED_ID_PREFIX = "interpretation_source_relaxed_repeat_preflight"
    base.SEED_SELECTION_ROLE = "objective_interpretation_source_relaxed_repeat_candidate_seed"
    base.SEED_SELECTION_NOTE = (
        "해석례_QA source-relaxed route를 반복하되 final split 24/8/8로 eval 부족을 더 직접 보강하는 seed"
    )
    base.NON_LAW_SCOPE_NOTE = "interpretation_source_relaxed_repeat_preflight_no_api_candidate_not_counted"
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 511,
        "train": 410,
        "eval": 101,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "current_objective_counted_seed_pool_after_balanced_recovery_signoff": 525,
    }
    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_interpretation_source_relaxed_repeat_preflight()
    relaxed.medium.base.main()


if __name__ == "__main__":
    main()
