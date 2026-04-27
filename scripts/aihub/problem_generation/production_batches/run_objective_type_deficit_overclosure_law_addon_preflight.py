from __future__ import annotations

import sys
from pathlib import Path

# 유형별 균형 overclosure stop line은 기존 law source-relaxed 공정을 재사용하되,
# final 120 대신 법령_QA final 32 add-on만 검산해 과잉 법령 실행 리스크를 낮춘다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_law_source_relaxed_repair_superwave_preflight as law_source_relaxed,
)


def configure_type_deficit_overclosure_law_addon_preflight() -> None:
    law_source_relaxed.configure_law_source_relaxed_repair_superwave()
    base = law_source_relaxed.medium.base

    base.VERSION_TAG = "objective_type_deficit_overclosure_law_addon_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_law_type_deficit_candidate64_final32_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective type-deficit overclosure law add-on preflight"
    base.SEED_ID_PREFIX = "type_deficit_law_addon_preflight"
    base.SEED_SELECTION_ROLE = "objective_type_deficit_overclosure_law_addon_candidate_seed"
    base.SEED_SELECTION_NOTE = "법령_QA 유형 부족 보강용 candidate 64 / final 32 source-relaxed add-on seed"
    base.NON_LAW_SCOPE_NOTE = "type_deficit_law_addon_preflight_no_api_candidate_not_counted"

    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 32
    base.EXPECTED_DOC_TYPE_COUNTS = {"법령_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {("법령_QA", "generalization_03_04"): 64}
    base.JUDGMENT_SOURCE_COUNTS = {
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
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "current_objective_counted_seed_pool_after_type_deficit_reflection": 683,
    }
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 683,
        "train": 524,
        "eval": 159,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.CURRENT_OBJECTIVE_COUNT = dict(base.OBJECTIVE_CURRENT_COUNT_REFERENCE)

    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_type_deficit_overclosure_law_addon_preflight()
    law_source_relaxed.medium.base.main()


if __name__ == "__main__":
    main()
