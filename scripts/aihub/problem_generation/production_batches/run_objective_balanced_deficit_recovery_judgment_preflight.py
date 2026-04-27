from __future__ import annotations

import sys
from pathlib import Path

# reviewer가 제안한 objective deficit recovery bundle에서 판결문_QA 안정 repeat route의 seed/spec을 먼저 고정한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_preflight as medium,
)


def configure_balanced_deficit_recovery_judgment_preflight() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base
    base.VERSION_TAG = "objective_balanced_deficit_recovery_judgment_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_judgment_candidate64_final40_eval_aware_repeat_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery judgment preflight"
    base.SEED_ID_PREFIX = "balanced_recovery_judgment_preflight"
    base.SEED_SELECTION_ROLE = "objective_balanced_deficit_recovery_judgment_candidate_seed"
    base.SEED_SELECTION_NOTE = "판결문_QA 안정 route를 candidate 64 / final 40으로 재확인하는 eval-aware repeat seed"
    base.SEED_FILTER_NOTE = "judgment_eval_aware_repeat_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "judgment_eval_aware_repeat_preflight_no_api_candidate_not_counted"
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 407,
        "train": 338,
        "eval": 69,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_balanced_deficit_recovery_judgment_preflight()
    medium.base.main()


if __name__ == "__main__":
    main()
