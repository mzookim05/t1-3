from __future__ import annotations

import sys
from pathlib import Path

# 해석례 medium 64개 preflight가 fresh 02/04 source 부족으로 막혀, 실행 가능한 균형 fallback을 먼저 닫는다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_small_overgeneration_preflight as base,
)


def configure_constrained_overgeneration() -> None:
    # source별 fresh seed 가용량을 넘지 않도록 candidate 40개에서 final 24개를 컴파일하는 보수적 API-first route다.
    base.VERSION_TAG = "objective_interpretation_constrained_overgeneration_pilot_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_target24_candidate40_seed_spec_wiring_check"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "interpretation constrained overgeneration preflight"
    base.SEED_ID_PREFIX = "interpretation_constrained_overgen_preflight"
    base.SEED_SELECTION_ROLE = "objective_interpretation_constrained_overgeneration_candidate_seed"
    base.SEED_SELECTION_NOTE = "해석례_QA package factory no-API candidate seed for target 24 / candidate 40"
    base.SEED_FILTER_NOTE = "interpretation_only_constrained_overgeneration_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "interpretation_constrained_overgeneration_preflight_no_api_candidate_not_counted"

    # candidate 40개는 source 10개씩, final 24개는 source/label/lane exact balance로 조립한다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 40
    base.FINAL_PACKAGE_TARGET_COUNT = 24
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 40}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "generalization_03_04"): 20,
        ("해석례_QA", "expansion_01_02"): 20,
    }
    base.INTERPRETATION_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 10,
        "02_TL_유권해석_QA": 10,
        "03_TL_해석례_QA": 10,
        "04_TL_해석례_QA": 10,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 6,
        "02_TL_유권해석_QA": 6,
        "03_TL_해석례_QA": 6,
        "04_TL_해석례_QA": 6,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 12, "expansion_01_02": 12}

    # 현행 objective count는 040231 판결문 medium count reflection 이후 값을 기준으로 고정한다.
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 239,
        "train": 196,
        "eval": 43,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "r2": 16,
        "pb2": 13,
        "pb3": 40,
        "pb4": 40,
        "pb9_final_package": 40,
        "judgment_a_slot_final_package": 16,
        "interpretation_dslot_final_package": 16,
        "judgment_small_overgeneration_final_package": 16,
        "interpretation_small_overgeneration_final_package": 16,
        "judgment_medium_overgeneration_final_package": 40,
    }
    base.refresh_paths()


def main() -> None:
    configure_constrained_overgeneration()
    base.main()


if __name__ == "__main__":
    main()
