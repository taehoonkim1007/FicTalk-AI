from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_chat_service,
    get_embedding_service,
    get_image_generation_service,
    get_story_generation_service,
    get_voice_service,
)
from src.api.routes import chat, embedding, health, image_generation, story_generation, tts
from src.models.database import get_db
from src.models.schemas import (
    CharacterGenerationResponse,
    GeneratedCharacter,
    ImageGenerationResponse,
    SummaryGenerationResponse,
    VoiceAttributes,
    VoiceRecommendResponse,
    VoiceSettings,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(story_generation.router, prefix="/api/story-generation")
    app.include_router(image_generation.router, prefix="/api/image-generation")
    app.include_router(tts.router, prefix="/api/tts")
    app.include_router(chat.router, prefix="/api/chat")
    app.include_router(embedding.router, prefix="/api/embeddings")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthRoute:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "fictalk-ai"


class TestStoryGenerationRoutes:
    def test_generate_summary(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_summary = AsyncMock(
            return_value=SummaryGenerationResponse(summary="생성된 줄거리입니다.")
        )
        app.dependency_overrides[get_story_generation_service] = lambda: mock_service

        response = client.post(
            "/api/story-generation/summary",
            json={"title": "테스트 제목", "description": "테스트 설명"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"] == "생성된 줄거리입니다."

        app.dependency_overrides.clear()

    def test_generate_summary_missing_field(self, app, client):
        mock_service = MagicMock()
        app.dependency_overrides[get_story_generation_service] = lambda: mock_service

        response = client.post(
            "/api/story-generation/summary",
            json={"title": "테스트 제목"},
        )
        assert response.status_code == 422

        app.dependency_overrides.clear()

    def test_generate_characters(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_characters = AsyncMock(
            return_value=CharacterGenerationResponse(
                characters=[
                    GeneratedCharacter(
                        name="홍길동",
                        role="주인공",
                        description="설명",
                        personality="성격",
                        firstMessage="안녕",
                    )
                ]
            )
        )
        app.dependency_overrides[get_story_generation_service] = lambda: mock_service

        response = client.post(
            "/api/story-generation/characters",
            json={
                "title": "테스트 제목",
                "description": "테스트 설명",
                "summary": "테스트 줄거리",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "characters" in data
        assert len(data["characters"]) == 1
        assert data["characters"][0]["name"] == "홍길동"

        app.dependency_overrides.clear()


class TestImageGenerationRoutes:
    def test_generate_profile_image(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_profile_image = AsyncMock(
            return_value=ImageGenerationResponse(
                image_base64="base64data", prompt_used="prompt used"
            )
        )
        app.dependency_overrides[get_image_generation_service] = lambda: mock_service

        response = client.post(
            "/api/image-generation/profile-image",
            json={"description": "캐릭터 설명", "personality": "성격"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert "prompt_used" in data

        app.dependency_overrides.clear()

    def test_generate_cover_image(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_cover_image = AsyncMock(
            return_value=ImageGenerationResponse(
                image_base64="base64data", prompt_used="prompt used"
            )
        )
        app.dependency_overrides[get_image_generation_service] = lambda: mock_service

        response = client.post(
            "/api/image-generation/cover-image",
            json={
                "title": "제목",
                "description": "설명",
                "summary": "줄거리",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data

        app.dependency_overrides.clear()

    def test_generate_background_image(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_background_image = AsyncMock(
            return_value=ImageGenerationResponse(
                image_base64="base64data", prompt_used="prompt used"
            )
        )
        app.dependency_overrides[get_image_generation_service] = lambda: mock_service

        response = client.post(
            "/api/image-generation/background-image",
            json={
                "title": "제목",
                "description": "설명",
                "summary": "줄거리",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data

        app.dependency_overrides.clear()

    def test_generate_character_background_image(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_character_background_image = AsyncMock(
            return_value=ImageGenerationResponse(
                image_base64="base64data", prompt_used="prompt used"
            )
        )
        app.dependency_overrides[get_image_generation_service] = lambda: mock_service

        response = client.post(
            "/api/image-generation/character-background-image",
            json={"description": "캐릭터 설명", "personality": "성격"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data

        app.dependency_overrides.clear()


class TestTTSRoutes:
    def test_recommend_voice(self, app, client):
        mock_service = MagicMock()
        mock_service.recommend_voice = AsyncMock(
            return_value=VoiceRecommendResponse(
                voice_id="voice-123",
                voice_name="Test Voice",
                attributes=VoiceAttributes(
                    gender="male",
                    age="young",
                    accent="Korean",
                    tone=["calm"],
                    keywords=["narrator"],
                ),
                voice_settings=VoiceSettings(),
            )
        )
        app.dependency_overrides[get_voice_service] = lambda: mock_service

        response = client.post(
            "/api/tts/voice-id",
            json={"description": "캐릭터 설명", "personality": "성격"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "voice_id" in data
        assert "voice_name" in data
        assert "attributes" in data
        assert "voice_settings" in data

        app.dependency_overrides.clear()

    def test_generate_sample(self, app, client):
        mock_service = MagicMock()
        mock_service.generate_sample_audio = AsyncMock(return_value=b"audio_bytes")
        app.dependency_overrides[get_voice_service] = lambda: mock_service

        response = client.post(
            "/api/tts/sample",
            json={"voice_id": "voice-123", "text": "안녕하세요"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "audio_base64" in data

        app.dependency_overrides.clear()


class TestChatRoutes:
    """Chat API 라우트 테스트."""

    def _mock_db_generator(self):
        """Mock DB 세션 생성기."""
        mock_db = MagicMock()

        async def gen():
            yield mock_db

        return gen

    def test_generate_chat_response_success(self, app, client):
        """채팅 응답 생성 성공 테스트."""
        mock_service = MagicMock()
        mock_service.run = AsyncMock(
            return_value={
                "response": "안녕하세요! 반갑습니다.",
                "mode": "creative",
                "used_rag": False,
                "max_similarity": 0.0,
                "session_id": "session-123",
            }
        )
        app.dependency_overrides[get_chat_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/chat/response",
            json={
                "character_name": "홍길동",
                "character_role": "주인공",
                "character_personality": "용감하고 정의로운",
                "story_id": "story-123",
                "story_title": "홍길동전",
                "story_summary": "조선시대 의적 홍길동의 이야기",
                "messages": [{"role": "user", "content": "안녕"}],
                "user_message": "오늘 기분이 어때?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "안녕하세요! 반갑습니다."
        assert data["mode"] == "creative"
        assert data["used_rag"] is False
        assert data["session_id"] == "session-123"

        app.dependency_overrides.clear()

    def test_generate_chat_response_with_rag_mode(self, app, client):
        """RAG 모드 응답 테스트."""
        mock_service = MagicMock()
        mock_service.run = AsyncMock(
            return_value={
                "response": "줄거리에 따르면...",
                "mode": "rag",
                "used_rag": True,
                "max_similarity": 0.85,
                "session_id": "session-456",
            }
        )
        app.dependency_overrides[get_chat_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/chat/response",
            json={
                "character_name": "홍길동",
                "character_role": "주인공",
                "character_personality": "용감한",
                "story_id": "story-123",
                "story_title": "홍길동전",
                "story_summary": "의적 이야기",
                "user_message": "활빈당에 대해 알려줘",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "rag"
        assert data["used_rag"] is True
        assert data["max_similarity"] == 0.85

        app.dependency_overrides.clear()

    def test_generate_chat_response_with_session_id(self, app, client):
        """세션 ID를 사용한 채팅 테스트."""
        mock_service = MagicMock()
        mock_service.run = AsyncMock(
            return_value={
                "response": "이전 대화를 기억합니다.",
                "mode": "creative",
                "used_rag": False,
                "max_similarity": 0.0,
                "session_id": "existing-session",
            }
        )
        app.dependency_overrides[get_chat_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/chat/response",
            json={
                "character_name": "캐릭터",
                "character_role": "조연",
                "character_personality": "친절한",
                "story_id": "story-123",
                "story_title": "테스트",
                "story_summary": "테스트 줄거리",
                "user_message": "계속 이야기해줘",
                "session_id": "existing-session",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"

        app.dependency_overrides.clear()

    def test_generate_chat_response_error(self, app, client):
        """채팅 응답 생성 실패 시 500 에러 반환."""
        mock_service = MagicMock()
        mock_service.run = AsyncMock(side_effect=Exception("AI 서버 오류"))
        app.dependency_overrides[get_chat_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/chat/response",
            json={
                "character_name": "홍길동",
                "character_role": "주인공",
                "character_personality": "용감한",
                "story_id": "story-123",
                "story_title": "홍길동전",
                "story_summary": "의적 이야기",
                "user_message": "안녕",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "AI 서버 오류" in data["detail"]

        app.dependency_overrides.clear()

    def test_generate_chat_response_missing_field(self, app, client):
        """필수 필드 누락 시 422 에러 반환."""
        mock_service = MagicMock()
        app.dependency_overrides[get_chat_service] = lambda: mock_service

        response = client.post(
            "/api/chat/response",
            json={
                "character_name": "홍길동",
                # character_role 누락
                "story_id": "story-123",
            },
        )

        assert response.status_code == 422

        app.dependency_overrides.clear()


class TestEmbeddingRoutes:
    """Embedding API 라우트 테스트."""

    def _mock_db_generator(self):
        """Mock DB 세션 생성기."""
        mock_db = MagicMock()

        async def gen():
            yield mock_db

        return gen

    def test_process_story_embedding_success(self, app, client):
        """스토리 임베딩 처리 성공 테스트."""
        mock_service = MagicMock()
        mock_service.process_story_summary = AsyncMock(return_value=5)
        app.dependency_overrides[get_embedding_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/embeddings/story",
            json={
                "story_id": "story-uuid-123",
                "summary": "이것은 테스트 스토리의 줄거리입니다. " * 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["story_id"] == "story-uuid-123"
        assert data["chunk_count"] == 5
        assert "5개의 청크가 생성되었습니다" in data["message"]

        app.dependency_overrides.clear()

    def test_process_story_embedding_error(self, app, client):
        """스토리 임베딩 처리 실패 시 500 에러 반환."""
        mock_service = MagicMock()
        mock_service.process_story_summary = AsyncMock(side_effect=Exception("임베딩 생성 실패"))
        app.dependency_overrides[get_embedding_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.post(
            "/api/embeddings/story",
            json={
                "story_id": "story-uuid-123",
                "summary": "테스트 줄거리",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "임베딩 생성 실패" in data["detail"]

        app.dependency_overrides.clear()

    def test_get_story_chunks_success(self, app, client):
        """스토리 청크 수 조회 성공 테스트."""
        mock_service = MagicMock()
        mock_service.get_story_chunk_count = AsyncMock(return_value=3)
        app.dependency_overrides[get_embedding_service] = lambda: mock_service
        app.dependency_overrides[get_db] = self._mock_db_generator()

        response = client.get("/api/embeddings/story/story-uuid-123/chunks")

        assert response.status_code == 200
        data = response.json()
        assert data["story_id"] == "story-uuid-123"
        assert data["chunk_count"] == 3

        app.dependency_overrides.clear()

    def test_batch_process_no_stories(self, app, client):
        """일괄 처리할 스토리가 없는 경우 테스트."""
        mock_service = MagicMock()
        app.dependency_overrides[get_embedding_service] = lambda: mock_service

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db_gen():
            yield mock_db

        app.dependency_overrides[get_db] = mock_get_db_gen

        response = client.post("/api/embeddings/batch")

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 0
        assert data["skipped"] == 0
        assert data["failed"] == 0
        assert data["results"] == []

        app.dependency_overrides.clear()

    def test_batch_process_with_stories(self, app, client):
        """일괄 처리 성공 테스트."""
        mock_service = MagicMock()
        mock_service.process_story_summary = AsyncMock(return_value=3)
        app.dependency_overrides[get_embedding_service] = lambda: mock_service

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("story-1", "제목1", "줄거리1"),
            ("story-2", "제목2", "줄거리2"),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db_gen():
            yield mock_db

        app.dependency_overrides[get_db] = mock_get_db_gen

        response = client.post("/api/embeddings/batch")

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 2
        assert data["skipped"] == 0
        assert data["failed"] == 0
        assert len(data["results"]) == 2

        app.dependency_overrides.clear()

    def test_batch_process_skips_empty_summary(self, app, client):
        """줄거리가 없는 스토리는 건너뛰기 테스트."""
        mock_service = MagicMock()
        mock_service.process_story_summary = AsyncMock(return_value=3)
        app.dependency_overrides[get_embedding_service] = lambda: mock_service

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("story-1", "제목1", ""),  # 빈 줄거리
            ("story-2", "제목2", "   "),  # 공백만 있는 줄거리
            ("story-3", "제목3", "유효한 줄거리"),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db_gen():
            yield mock_db

        app.dependency_overrides[get_db] = mock_get_db_gen

        response = client.post("/api/embeddings/batch")

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 1
        assert data["skipped"] == 2
        assert data["failed"] == 0

        app.dependency_overrides.clear()

    def test_batch_process_handles_failure(self, app, client):
        """일괄 처리 중 일부 실패 테스트."""
        mock_service = MagicMock()
        mock_service.process_story_summary = AsyncMock(side_effect=[3, Exception("처리 실패")])
        app.dependency_overrides[get_embedding_service] = lambda: mock_service

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("story-1", "제목1", "줄거리1"),
            ("story-2", "제목2", "줄거리2"),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        async def mock_get_db_gen():
            yield mock_db

        app.dependency_overrides[get_db] = mock_get_db_gen

        response = client.post("/api/embeddings/batch")

        assert response.status_code == 200
        data = response.json()
        assert data["processed"] == 1
        assert data["failed"] == 1

        app.dependency_overrides.clear()
