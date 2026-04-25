import os
from datetime import datetime
from zoneinfo import ZoneInfo


RUN_STAMP_ENV_VAR = "AIHUB_RUN_STAMP"
KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


def build_run_stamp() -> str:
    # llm_runs 이름의 날짜/시각은 실제 실행 시각과 맞아야 하므로 직접 하드코딩하지 않는다.
    # 다만 과거 run 재현이 필요할 때는 환경변수로 명시해 기존 산출물 이름을 그대로 만들 수 있게 둔다.
    override = os.environ.get(RUN_STAMP_ENV_VAR, "").strip()
    if override:
        return override
    return datetime.now(KOREA_TIMEZONE).strftime("%Y-%m-%d_%H%M%S")
