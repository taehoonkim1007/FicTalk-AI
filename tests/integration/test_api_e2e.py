import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import chat, embedding, health, image_generation, story_generation, tts
from src.services.chat_service import ChatService
from src.services.embedding_service import EmbeddingService
from src.services.image_generation_service import ImageGenerationService
from src.services.story_generation_service import StoryGenerationService
from src.services.voice_service import VoiceService
from tests.integration.conftest import (
    skip_if_no_db,
    skip_if_no_elevenlabs,
    skip_if_no_gemini,
)


def create_test_app(with_db: bool = False) -> FastAPI:
    """테스트용 FastAPI 앱 생성."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # 서비스 싱글톤 초기화 (DB 초기화 없이)
        app.state.chat_service = ChatService()
        app.state.embedding_service = EmbeddingService()
        app.state.story_generation_service = StoryGenerationService()
        app.state.image_generation_service = ImageGenerationService()
        app.state.voice_service = VoiceService()
        yield

    test_app = FastAPI(lifespan=lifespan)

    test_app.include_router(health.router, tags=["Health"])
    test_app.include_router(
        story_generation.router, prefix="/api/story-generation", tags=["Story Generation"]
    )
    test_app.include_router(
        image_generation.router, prefix="/api/image-generation", tags=["Image Generation"]
    )
    test_app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    test_app.include_router(embedding.router, prefix="/api/embeddings", tags=["Embeddings"])
    test_app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])

    return test_app


class TestHealthEndpoint:
    """헬스 체크 엔드포인트 테스트."""

    def test_health_endpoint_returns_ok(self):
        """헬스 엔드포인트가 정상 응답한다."""
        test_app = create_test_app()
        with TestClient(test_app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


@skip_if_no_gemini
class TestStoryGenerationEndpoints:
    """스토리 생성 API 엔드포인트 테스트."""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트."""
        test_app = create_test_app()
        with TestClient(test_app) as client:
            yield client

    def test_generate_summary_endpoint(self, client):
        """줄거리 생성 엔드포인트가 동작한다."""
        response = client.post(
            "/api/story-generation/summary",
            json={
                "title": "용사의 모험",
                "description": "평범한 소년이 용사가 되는 이야기",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert len(data["summary"]) > 50

    def test_generate_characters_endpoint(self, client):
        """캐릭터 생성 엔드포인트가 동작한다."""
        response = client.post(
            "/api/story-generation/characters",
            json={
                "title": "기사의 모험",
                "description": "공주를 구하는 기사 이야기",
                "summary": "용감한 기사 알렉스가 공주 엘라를 구하기 위해 모험을 떠난다.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "characters" in data
        assert len(data["characters"]) >= 1

    def test_generate_summary_validation_error(self, client):
        """잘못된 요청에 422를 반환한다."""
        response = client.post(
            "/api/story-generation/summary",
            json={"title": "테스트"},  # description 누락
        )

        assert response.status_code == 422


@skip_if_no_gemini
class TestImageGenerationEndpoints:
    """이미지 생성 API 엔드포인트 테스트."""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트."""
        test_app = create_test_app()
        with TestClient(test_app) as client:
            yield client

    def test_generate_profile_image_endpoint(self, client):
        """프로필 이미지 생성 엔드포인트가 동작한다."""
        response = client.post(
            "/api/image-generation/profile-image",
            json={
                "description": "푸른 눈의 젊은 마법사",
                "personality": "지적이고 차분한 성격",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert len(data["image_base64"]) > 100

    def test_generate_cover_image_endpoint(self, client):
        """커버 이미지 생성 엔드포인트가 동작한다."""
        response = client.post(
            "/api/image-generation/cover-image",
            json={
                "title": "마법사의 여정",
                "description": "어린 마법사가 세계를 구하는 이야기",
                "summary": "주인공 아리안은 마법 아카데미에서 성장한다.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data


@skip_if_no_gemini
@skip_if_no_elevenlabs
class TestTTSEndpoints:
    """TTS API 엔드포인트 테스트."""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트."""
        test_app = create_test_app()
        with TestClient(test_app) as client:
            yield client

    def test_recommend_voice_endpoint(self, client):
        """음성 추천 엔드포인트가 동작한다."""
        response = client.post(
            "/api/tts/voice-id",
            json={
                "description": "20대 초반의 젊은 여성 마법사",
                "personality": "지적이고 차분한 성격",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "voice_id" in data
        assert "voice_name" in data

    def test_generate_sample_audio_endpoint(self, client):
        """샘플 오디오 생성 엔드포인트가 동작한다."""
        response = client.post(
            "/api/tts/sample",
            json={
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "text": "안녕하세요, 테스트입니다.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "audio_base64" in data
        assert len(data["audio_base64"]) > 0


@skip_if_no_db
@skip_if_no_gemini
class TestEmbeddingEndpoints:
    """임베딩 API 엔드포인트 테스트."""

    @pytest.fixture
    async def async_client(self, test_engine, setup_database):
        """비동기 테스트 클라이언트 (DB 포함)."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from src.models.database import get_db

        # lifespan 없이 앱 생성 (서비스 수동 초기화)
        test_app = FastAPI()

        # 서비스 초기화
        test_app.state.chat_service = ChatService()
        test_app.state.embedding_service = EmbeddingService()
        test_app.state.story_generation_service = StoryGenerationService()
        test_app.state.image_generation_service = ImageGenerationService()
        test_app.state.voice_service = VoiceService()

        # 라우터 등록
        test_app.include_router(health.router, tags=["Health"])
        test_app.include_router(embedding.router, prefix="/api/embeddings", tags=["Embeddings"])
        test_app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

        async_session_factory = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async def override_get_db():
            async with async_session_factory() as session:
                yield session

        test_app.dependency_overrides[get_db] = override_get_db

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_process_story_embedding_endpoint(self, async_client):
        """스토리 임베딩 생성 엔드포인트가 동작한다."""
        story_id = str(uuid.uuid4())

        response = await async_client.post(
            "/api/embeddings/story",
            json={
                "story_id": story_id,
                "summary": "어린 마법사 아리안은 마법 아카데미에 입학하여 친구들을 만난다.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "story_id" in data
        assert "chunk_count" in data
        assert data["chunk_count"] >= 1

    @pytest.mark.asyncio
    async def test_get_story_chunks_endpoint(self, async_client):
        """청크 수 조회 엔드포인트가 동작한다."""
        story_id = str(uuid.uuid4())

        # 먼저 임베딩 생성
        await async_client.post(
            "/api/embeddings/story",
            json={
                "story_id": story_id,
                "summary": "테스트 줄거리입니다.",
            },
        )

        # 청크 수 조회
        response = await async_client.get(f"/api/embeddings/story/{story_id}/chunks")

        assert response.status_code == 200
        data = response.json()
        assert "story_id" in data
        assert "chunk_count" in data


@skip_if_no_db
@skip_if_no_gemini
class TestChatEndpoints:
    """채팅 API 엔드포인트 테스트."""

    @pytest.fixture
    async def async_client(self, test_engine, setup_database):
        """비동기 테스트 클라이언트 (DB 포함)."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from src.models.database import get_db

        # lifespan 없이 앱 생성 (서비스 수동 초기화)
        test_app = FastAPI()

        # 서비스 초기화
        test_app.state.chat_service = ChatService()
        test_app.state.embedding_service = EmbeddingService()
        test_app.state.story_generation_service = StoryGenerationService()
        test_app.state.image_generation_service = ImageGenerationService()
        test_app.state.voice_service = VoiceService()

        # 라우터 등록
        test_app.include_router(health.router, tags=["Health"])
        test_app.include_router(embedding.router, prefix="/api/embeddings", tags=["Embeddings"])
        test_app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

        async_session_factory = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async def override_get_db():
            async with async_session_factory() as session:
                yield session

        test_app.dependency_overrides[get_db] = override_get_db

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=test_app),
            base_url="http://test",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_response(self, async_client):
        """채팅 엔드포인트가 응답을 반환한다."""
        response = await async_client.post(
            "/api/chat/response",
            json={
                "story_id": str(uuid.uuid4()),
                "character_name": "아리안",
                "character_role": "주인공",
                "character_personality": "지적이고 차분한 마법사",
                "story_title": "마법사의 여정",
                "story_summary": "어린 마법사 아리안이 마법 아카데미에서 성장하는 이야기",
                "user_message": "안녕하세요!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert len(data["response"]) > 0

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_session_id(self, async_client):
        """세션 ID로 대화를 이어간다."""
        story_id = str(uuid.uuid4())
        chat_request = {
            "story_id": story_id,
            "character_name": "아리안",
            "character_role": "주인공",
            "character_personality": "지적이고 차분한 마법사",
            "story_title": "마법사의 여정",
            "story_summary": "어린 마법사 아리안이 마법 아카데미에서 성장하는 이야기",
            "user_message": "안녕하세요!",
        }

        # 첫 번째 메시지
        response1 = await async_client.post("/api/chat/response", json=chat_request)
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # 두 번째 메시지 (같은 세션)
        chat_request["user_message"] = "오늘 날씨가 좋네요."
        chat_request["session_id"] = session_id

        response2 = await async_client.post("/api/chat/response", json=chat_request)

        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_chat_endpoint_validation_error(self, async_client):
        """잘못된 요청에 422를 반환한다."""
        response = await async_client.post(
            "/api/chat/response",
            json={
                "story_id": "test",
                # character_name 등 필수 필드 누락
            },
        )

        assert response.status_code == 422
