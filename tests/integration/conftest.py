import os
import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# 테스트 DB URL (docker-compose.test.yml의 포트 5433)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:testpassword@localhost:5433/fictalk_test",
)


@pytest.fixture
async def test_engine():
    """테스트용 DB 엔진."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def setup_database(test_engine):
    """테스트 DB 스키마 초기화."""
    async with test_engine.begin() as conn:
        # pgvector 확장 활성화
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # StoryContent 테이블 생성 (Prisma 스키마와 동일)
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS "StoryContent" (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                "chunkIndex" INTEGER NOT NULL,
                embedding vector(768),
                "createdAt" TIMESTAMP DEFAULT NOW(),
                "storyId" TEXT NOT NULL
            )
        """)
        )

    yield


@pytest.fixture
async def db_session(test_engine, setup_database) -> AsyncGenerator[AsyncSession, None]:
    """테스트용 DB 세션."""
    async_session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        # 테스트 후 롤백
        await session.rollback()


@pytest.fixture
async def clean_db(db_session: AsyncSession):
    """각 테스트 전 DB 정리."""
    await db_session.execute(text('DELETE FROM "StoryContent"'))
    await db_session.commit()
    yield
    await db_session.execute(text('DELETE FROM "StoryContent"'))
    await db_session.commit()


@pytest.fixture
def story_id() -> str:
    """테스트용 스토리 ID."""
    return f"test-story-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_story_data() -> dict:
    """테스트용 스토리 데이터."""
    return {
        "title": "마법사의 여정",
        "description": "젊은 마법사가 세계를 구하는 이야기",
        "summary": """
        어린 시절부터 마법에 재능을 보인 주인공 아리안은 마법 아카데미에 입학한다.
        그곳에서 그는 친구 레온과 소피아를 만나고, 함께 수련하며 성장한다.
        어느 날 고대의 봉인이 풀리며 어둠의 세력이 깨어나고,
        아리안과 친구들은 세계를 구하기 위한 모험을 떠난다.
        수많은 시련을 겪으며 그들은 진정한 용기와 우정의 의미를 깨닫게 된다.
        마침내 아리안은 자신의 잠재력을 깨우고 어둠의 군주와 맞서 싸운다.
        """,
    }


def is_db_available() -> bool:
    """테스트 DB 사용 가능 여부 확인."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 5433))
        sock.close()
        return result == 0
    except Exception:
        return False


def is_gemini_available() -> bool:
    """Gemini API 키 설정 여부 확인."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return bool(api_key) and len(api_key) > 10


def is_elevenlabs_available() -> bool:
    """ElevenLabs API 키 설정 여부 확인."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    return bool(api_key) and len(api_key) > 10


# 조건부 스킵 마커
skip_if_no_db = pytest.mark.skipif(
    not is_db_available(),
    reason="테스트 DB 미실행 (docker-compose -f docker-compose.test.yml up -d)",
)

skip_if_no_gemini = pytest.mark.skipif(
    not is_gemini_available(),
    reason="GOOGLE_API_KEY 환경변수 미설정",
)

skip_if_no_elevenlabs = pytest.mark.skipif(
    not is_elevenlabs_available(),
    reason="ELEVENLABS_API_KEY 환경변수 미설정",
)
