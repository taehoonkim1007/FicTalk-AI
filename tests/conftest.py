import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """테스트 환경 변수 설정."""
    os.environ.setdefault("DEBUG", "false")
