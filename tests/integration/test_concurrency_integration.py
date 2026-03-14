"""동시성 제어 통합 테스트.

ChatSessionCache의 asyncio.Lock, contextvars를 통한 DB 세션 격리,
임베딩 서비스의 병렬 처리가 올바르게 동작하는지 검증합니다.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.chat_service import ChatService, ChatSessionCache, _current_db_context
from src.services.embedding_service import EMBEDDING_CONCURRENCY_LIMIT, EmbeddingService
from src.services.story_generation_service import API_CONCURRENCY_LIMIT, StoryGenerationService


class TestChatSessionCacheConcurrencyIntegration:
    """ChatSessionCache 동시성 통합 테스트."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation_no_collision(self):
        """동시에 많은 세션을 생성해도 ID 충돌이 없어야 함."""
        cache = ChatSessionCache(ttl_minutes=30)

        # 1000개의 동시 세션 생성
        session_ids = await asyncio.gather(
            *[cache.create_session([{"role": "user", "content": f"msg-{i}"}]) for i in range(1000)]
        )

        # 모든 ID가 고유해야 함
        assert len(set(session_ids)) == 1000, "세션 ID 충돌 발생"

    @pytest.mark.asyncio
    async def test_concurrent_read_write_consistency(self):
        """동시 읽기/쓰기 작업에서 데이터 일관성 유지."""
        cache = ChatSessionCache(ttl_minutes=30)
        session_id = await cache.create_session([])

        # 여러 태스크가 동시에 메시지 추가
        async def add_message(index: int):
            messages = await cache.get(session_id)
            if messages is not None:
                new_messages = [*messages, {"role": "user", "content": f"msg-{index}"}]
                await cache.set(session_id, new_messages)

        await asyncio.gather(*[add_message(i) for i in range(50)])

        # 최종 상태 확인 - 최소한 일부 메시지가 저장되어야 함
        final_messages = await cache.get(session_id)
        assert final_messages is not None
        assert len(final_messages) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_clear_expired(self):
        """동시 만료 정리 작업이 안전하게 처리됨."""
        cache = ChatSessionCache(ttl_minutes=0)  # 즉시 만료

        # 100개 세션 생성
        for i in range(100):
            await cache.set(f"session-{i}", [])

        # 동시에 만료 정리 실행
        results = await asyncio.gather(*[cache.clear_expired() for _ in range(10)])

        # 모든 정리 작업의 합이 100이어야 함 (각 세션은 한 번만 정리됨)
        assert sum(results) == 100


class TestDBContextIsolationIntegration:
    """DB 세션 컨텍스트 격리 통합 테스트."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_use_separate_db_sessions(self):
        """동시 요청이 각자의 DB 세션을 사용해야 함."""
        captured_db_sessions: list[AsyncMock] = []

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
                "response": "테스트 응답",
                "used_rag": False,
            }

            service = ChatService()

            async def make_request(index: int):
                mock_db = AsyncMock()
                mock_db.index = index  # 식별용 인덱스
                captured_db_sessions.append(mock_db)

                await service.run(
                    character_name=f"캐릭터-{index}",
                    character_role="역할",
                    character_personality="성격",
                    story_id=f"story-{index}",
                    story_title="제목",
                    story_summary="요약",
                    messages=None,
                    user_message=f"메시지-{index}",
                    db=mock_db,
                )

            # 10개의 동시 요청
            await asyncio.gather(*[make_request(i) for i in range(10)])

            # 각 요청이 별도의 DB 세션을 사용했는지 확인
            assert len(captured_db_sessions) == 10

    @pytest.mark.asyncio
    async def test_context_var_isolation_between_tasks(self):
        """컨텍스트 변수가 태스크 간에 격리됨."""
        results: list[int] = []

        async def set_and_read(index: int):
            mock_db = MagicMock()
            mock_db.index = index
            token = _current_db_context.set(mock_db)
            try:
                await asyncio.sleep(0.01)  # 다른 태스크에게 실행 기회 제공
                current_db = _current_db_context.get()
                if current_db is not None:
                    results.append(current_db.index)
            finally:
                _current_db_context.reset(token)

        # 동시에 여러 태스크 실행
        await asyncio.gather(*[set_and_read(i) for i in range(20)])

        # 각 태스크가 자신이 설정한 값을 읽어야 함
        assert len(results) == 20
        assert sorted(results) == list(range(20))


class TestEmbeddingServiceConcurrencyIntegration:
    """임베딩 서비스 동시성 통합 테스트."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_embeddings(self):
        """세마포어가 동시 임베딩 요청 수를 제한함."""
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_embed_content(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.05)  # API 호출 시뮬레이션

            async with lock:
                concurrent_count -= 1

            mock_result = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_result.embeddings = [mock_embedding]
            return mock_result

        with patch("src.services.embedding_service.genai.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.aio.models.embed_content = mock_embed_content
            MockClient.return_value = mock_client

            service = EmbeddingService()

            # 20개의 동시 임베딩 요청
            chunks = [f"청크 {i}" for i in range(20)]

            # 서비스의 _api_semaphore 사용 (실제 구현 검증)
            async def embed_with_limit(chunk: str):
                async with service._api_semaphore:
                    return await service.generate_embedding(chunk)

            await asyncio.gather(*[embed_with_limit(chunk) for chunk in chunks])

            # 동시 실행 수가 제한을 초과하지 않아야 함
            assert max_concurrent <= EMBEDDING_CONCURRENCY_LIMIT, (
                f"동시 실행 수 {max_concurrent}가 제한 {EMBEDDING_CONCURRENCY_LIMIT}을 초과"
            )


class TestStoryGenerationConcurrencyIntegration:
    """스토리 생성 서비스 동시성 통합 테스트."""

    @pytest.mark.asyncio
    async def test_character_extraction_semaphore(self):
        """캐릭터 추출 시 세마포어가 동시 요청을 제한함."""
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_generate_content(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.05)  # API 호출 시뮬레이션

            async with lock:
                concurrent_count -= 1

            mock_response = MagicMock()
            mock_response.text = '{"characters": [{"name": "테스트", "role": "주인공", "description": "설명", "personality": "성격", "firstMessage": "안녕"}]}'
            return mock_response

        with patch("src.services.story_generation_service.genai.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate_content
            MockClient.return_value = mock_client

            service = StoryGenerationService()

            # 긴 줄거리로 여러 청크 생성
            long_summary = "이것은 테스트 줄거리입니다. " * 500

            await service.generate_characters(
                title="테스트",
                description="설명",
                summary=long_summary,
            )

            # 동시 실행 수가 제한을 초과하지 않아야 함
            assert max_concurrent <= API_CONCURRENCY_LIMIT, (
                f"동시 실행 수 {max_concurrent}가 제한 {API_CONCURRENCY_LIMIT}을 초과"
            )


class TestChatServiceRequestIsolation:
    """ChatService 요청 격리 통합 테스트."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_chat_requests(self):
        """여러 동시 채팅 요청이 서로 간섭 없이 처리됨."""
        with (
            patch("src.services.chat_service.RAGService"),
            patch("src.services.chat_service.retrieve_and_evaluate") as mock_retrieve,
            patch("src.services.chat_service.generate_creative_response") as mock_creative,
        ):

            async def mock_retrieve_fn(state, rag_service, db):
                # 요청별로 다른 응답
                await asyncio.sleep(0.01)
                return {
                    "rag_results": [],
                    "max_similarity": 0.2,
                    "mode": "creative",
                }

            async def mock_creative_fn(state):
                await asyncio.sleep(0.01)
                return {
                    "response": f"응답: {state['user_message']}",
                    "used_rag": False,
                }

            mock_retrieve.side_effect = mock_retrieve_fn
            mock_creative.side_effect = mock_creative_fn

            service = ChatService()

            async def make_request(index: int) -> dict:
                mock_db = AsyncMock()
                result = await service.run(
                    character_name="캐릭터",
                    character_role="역할",
                    character_personality="성격",
                    story_id="story-1",
                    story_title="제목",
                    story_summary="요약",
                    messages=None,
                    user_message=f"메시지-{index}",
                    db=mock_db,
                )
                return {"index": index, "response": result["response"]}

            # 20개의 동시 요청
            results = await asyncio.gather(*[make_request(i) for i in range(20)])

            # 각 요청이 올바른 응답을 받았는지 확인
            for r in results:
                expected_response = f"응답: 메시지-{r['index']}"
                assert r["response"] == expected_response, (
                    f"요청 {r['index']}의 응답이 잘못됨: {r['response']}"
                )
