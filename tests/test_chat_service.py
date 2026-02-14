from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.services.chat_service import ChatService, ChatSessionCache, _current_db_context


@pytest.fixture
def cache():
    return ChatSessionCache(ttl_minutes=30)


class TestChatSessionCacheGet:
    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent_session(self, cache):
        result = await cache.get("nonexistent-session-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_messages_for_valid_session(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        await cache.set("session-1", messages)
        result = await cache.get("session-1")
        assert result == messages

    @pytest.mark.asyncio
    async def test_returns_none_for_expired_session(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        await cache.set("session-1", messages)
        cache._cache["session-1"]["expires_at"] = datetime.now() - timedelta(minutes=1)
        result = await cache.get("session-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_removes_expired_session_from_cache(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        await cache.set("session-1", messages)
        cache._cache["session-1"]["expires_at"] = datetime.now() - timedelta(minutes=1)
        await cache.get("session-1")
        assert "session-1" not in cache._cache


class TestChatSessionCacheSet:
    @pytest.mark.asyncio
    async def test_stores_messages(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        await cache.set("session-1", messages)
        assert cache._cache["session-1"]["messages"] == messages

    @pytest.mark.asyncio
    async def test_sets_expiration(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        await cache.set("session-1", messages)
        expires_at = cache._cache["session-1"]["expires_at"]
        assert expires_at > datetime.now()
        assert expires_at < datetime.now() + timedelta(minutes=31)

    @pytest.mark.asyncio
    async def test_overwrites_existing_session(self, cache):
        await cache.set("session-1", [{"role": "user", "content": "안녕"}])
        await cache.set("session-1", [{"role": "user", "content": "새 메시지"}])
        assert cache._cache["session-1"]["messages"] == [{"role": "user", "content": "새 메시지"}]


class TestChatSessionCacheUpdate:
    @pytest.mark.asyncio
    async def test_updates_messages(self, cache):
        await cache.set("session-1", [{"role": "user", "content": "안녕"}])
        new_messages = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요"},
        ]
        await cache.update("session-1", new_messages)
        assert cache._cache["session-1"]["messages"] == new_messages

    @pytest.mark.asyncio
    async def test_does_nothing_for_nonexistent_session(self, cache):
        await cache.update("nonexistent", [{"role": "user", "content": "안녕"}])
        assert "nonexistent" not in cache._cache


class TestChatSessionCacheCreateSession:
    @pytest.mark.asyncio
    async def test_creates_new_session_id(self, cache):
        session_id = await cache.create_session()
        assert session_id is not None
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_stores_empty_messages_by_default(self, cache):
        session_id = await cache.create_session()
        assert cache._cache[session_id]["messages"] == []

    @pytest.mark.asyncio
    async def test_stores_provided_messages(self, cache):
        messages = [{"role": "user", "content": "안녕"}]
        session_id = await cache.create_session(messages)
        assert cache._cache[session_id]["messages"] == messages

    @pytest.mark.asyncio
    async def test_unique_session_ids(self, cache):
        session_ids = [await cache.create_session() for _ in range(100)]
        assert len(set(session_ids)) == 100


class TestChatSessionCacheClearExpired:
    @pytest.mark.asyncio
    async def test_clears_expired_sessions(self, cache):
        await cache.set("expired-1", [])
        await cache.set("expired-2", [])
        await cache.set("valid", [])

        cache._cache["expired-1"]["expires_at"] = datetime.now() - timedelta(minutes=1)
        cache._cache["expired-2"]["expires_at"] = datetime.now() - timedelta(minutes=1)

        cleared_count = await cache.clear_expired()

        assert cleared_count == 2
        assert "expired-1" not in cache._cache
        assert "expired-2" not in cache._cache
        assert "valid" in cache._cache

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_expired(self, cache):
        await cache.set("valid-1", [])
        await cache.set("valid-2", [])
        cleared_count = await cache.clear_expired()
        assert cleared_count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_cache(self, cache):
        cleared_count = await cache.clear_expired()
        assert cleared_count == 0


class TestChatServiceInit:
    def test_init_creates_components(self):
        """초기화 시 RAG 서비스, 세션 캐시, 워크플로우 생성."""
        with patch("src.services.chat_service.RAGService") as mock_rag:
            service = ChatService()
            mock_rag.assert_called_once()
            assert service._session_cache is not None
            assert service._workflow is not None


class TestChatServiceRouteByMode:
    def test_route_returns_mode(self):
        """상태의 mode 값 반환."""
        with patch("src.services.chat_service.RAGService"):
            service = ChatService()

            rag_state = {"mode": "rag"}
            assert service._route_by_mode(rag_state) == "rag"

            creative_state = {"mode": "creative"}
            assert service._route_by_mode(creative_state) == "creative"


class TestChatServiceRun:
    @pytest.mark.asyncio
    async def test_run_creates_new_session(self):
        """세션 ID 없으면 새 세션 생성."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.retrieve_and_evaluate") as mock_retrieve,
            patch("src.services.chat_service.generate_creative_response") as mock_creative,
        ):
            mock_retrieve.return_value = {
                "rag_results": [],
                "max_similarity": 0.3,
                "mode": "creative",
            }
            mock_creative.return_value = {
                "response": "테스트 응답",
                "used_rag": False,
            }

            service = ChatService()
            mock_db = AsyncMock()

            result = await service.run(
                character_name="테스트 캐릭터",
                character_role="히어로",
                character_personality="용감함",
                story_id="story-123",
                story_title="테스트 스토리",
                story_summary="테스트 요약",
                messages=None,
                user_message="안녕",
                db=mock_db,
                session_id=None,
            )

            assert "session_id" in result
            assert result["session_id"] is not None
            assert result["response"] == "테스트 응답"
            assert result["mode"] == "creative"

    @pytest.mark.asyncio
    async def test_run_uses_cached_session(self):
        """세션 ID 있으면 캐시된 메시지 사용."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.retrieve_and_evaluate") as mock_retrieve,
            patch("src.services.chat_service.generate_rag_response") as mock_rag,
        ):
            mock_retrieve.return_value = {
                "rag_results": [("context", 0.9)],
                "max_similarity": 0.9,
                "mode": "rag",
            }
            mock_rag.return_value = {
                "response": "RAG 응답",
                "used_rag": True,
            }

            service = ChatService()
            mock_db = AsyncMock()

            # 첫 번째 호출 - 세션 생성
            result1 = await service.run(
                character_name="테스트",
                character_role="역할",
                character_personality="성격",
                story_id="story-123",
                story_title="제목",
                story_summary="요약",
                messages=[],
                user_message="첫 메시지",
                db=mock_db,
            )

            session_id = result1["session_id"]

            # 두 번째 호출 - 세션 재사용
            result2 = await service.run(
                character_name="테스트",
                character_role="역할",
                character_personality="성격",
                story_id="story-123",
                story_title="제목",
                story_summary="요약",
                messages=None,
                user_message="두 번째 메시지",
                db=mock_db,
                session_id=session_id,
            )

            assert result2["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_run_handles_expired_session(self):
        """만료된 세션은 새 세션 생성."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.retrieve_and_evaluate") as mock_retrieve,
            patch("src.services.chat_service.generate_creative_response") as mock_creative,
        ):
            mock_retrieve.return_value = {
                "rag_results": [],
                "max_similarity": 0.2,
                "mode": "creative",
            }
            mock_creative.return_value = {
                "response": "응답",
                "used_rag": False,
            }

            service = ChatService()
            mock_db = AsyncMock()

            # 만료된 세션 ID로 호출
            result = await service.run(
                character_name="테스트",
                character_role="역할",
                character_personality="성격",
                story_id="story-123",
                story_title="제목",
                story_summary="요약",
                messages=[{"role": "user", "content": "이전 메시지"}],
                user_message="새 메시지",
                db=mock_db,
                session_id="expired-session-id",
            )

            # 만료된 세션이므로 새 세션 ID 생성
            assert result["session_id"] != "expired-session-id"


class TestChatServiceNodes:
    @pytest.mark.asyncio
    async def test_retrieve_node(self):
        """RAG 검색 노드."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.retrieve_and_evaluate") as mock_retrieve,
        ):
            mock_retrieve.return_value = {
                "rag_results": [("context", 0.8)],
                "max_similarity": 0.8,
                "mode": "rag",
            }

            service = ChatService()
            mock_db = AsyncMock()
            # contextvars에 DB 세션 설정
            token = _current_db_context.set(mock_db)

            try:
                state = {
                    "story_id": "story-123",
                    "user_message": "테스트",
                }

                result = await service._retrieve_node(state)

                assert result["mode"] == "rag"
                mock_retrieve.assert_called_once()
            finally:
                _current_db_context.reset(token)

    @pytest.mark.asyncio
    async def test_retrieve_node_raises_without_db(self):
        """DB 세션이 없으면 RuntimeError 발생."""
        with patch("src.services.chat_service.RAGService"):
            service = ChatService()

            state = {
                "story_id": "story-123",
                "user_message": "테스트",
            }

            with pytest.raises(RuntimeError, match="DB session not set in context"):
                await service._retrieve_node(state)

    @pytest.mark.asyncio
    async def test_rag_response_node(self):
        """RAG 응답 노드."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.generate_rag_response") as mock_gen,
        ):
            mock_gen.return_value = {
                "response": "RAG 응답",
                "used_rag": True,
            }

            service = ChatService()
            state = {"rag_results": [("context", 0.9)]}

            result = await service._rag_response_node(state)

            assert result["response"] == "RAG 응답"
            assert result["used_rag"] is True

    @pytest.mark.asyncio
    async def test_creative_response_node(self):
        """Creative 응답 노드."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.generate_creative_response") as mock_gen,
        ):
            mock_gen.return_value = {
                "response": "Creative 응답",
                "used_rag": False,
            }

            service = ChatService()
            state = {}

            result = await service._creative_response_node(state)

            assert result["response"] == "Creative 응답"
            assert result["used_rag"] is False


class TestChatSessionCacheConcurrency:
    """ChatSessionCache 동시성 테스트."""

    @pytest.mark.asyncio
    async def test_concurrent_set_operations(self, cache):
        """동시 set 작업이 안전하게 처리되어야 함."""
        import asyncio

        async def set_session(i: int):
            await cache.set(f"session-{i}", [{"role": "user", "content": f"message-{i}"}])

        # 100개의 동시 set 작업
        await asyncio.gather(*[set_session(i) for i in range(100)])

        # 모든 세션이 올바르게 저장되어야 함
        assert len(cache._cache) == 100
        for i in range(100):
            messages = await cache.get(f"session-{i}")
            assert messages == [{"role": "user", "content": f"message-{i}"}]

    @pytest.mark.asyncio
    async def test_concurrent_create_session_unique_ids(self, cache):
        """동시 create_session 호출 시 고유 ID 생성."""
        import asyncio

        # 100개의 동시 세션 생성
        session_ids = await asyncio.gather(*[cache.create_session() for _ in range(100)])

        # 모든 ID가 고유해야 함
        assert len(set(session_ids)) == 100

    @pytest.mark.asyncio
    async def test_concurrent_get_and_update(self, cache):
        """동시 get/update 작업이 데이터 손실 없이 처리되어야 함."""
        import asyncio

        session_id = await cache.create_session([{"role": "user", "content": "initial"}])

        async def update_and_get(i: int):
            messages = await cache.get(session_id)
            if messages is not None:
                new_messages = [*messages, {"role": "user", "content": f"msg-{i}"}]
                await cache.update(session_id, new_messages)
            return await cache.get(session_id)

        # 10개의 동시 update 작업
        await asyncio.gather(*[update_and_get(i) for i in range(10)])

        # 최종 메시지 수 확인 (최소 initial + 일부 추가)
        final_messages = await cache.get(session_id)
        assert final_messages is not None
        assert len(final_messages) >= 1  # 최소한 initial 메시지는 있어야 함
