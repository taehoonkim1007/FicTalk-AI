from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.embedding_service import EmbeddingService


@pytest.fixture
def service():
    service = EmbeddingService.__new__(EmbeddingService)
    service.CHUNK_SIZE = 500
    service.CHUNK_OVERLAP = 50
    return service


class TestEmbeddingServiceInit:
    def test_init_creates_client(self):
        """초기화 시 genai client 생성."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            service = EmbeddingService()
            mock_client.assert_called_once()
            assert service._client is not None


class TestChunkText:
    def test_empty_text_returns_empty_list(self, service):
        chunks = service.chunk_text("")
        assert chunks == []

    def test_short_text_returns_single_chunk(self, service):
        text = "짧은 텍스트입니다."
        chunks = service.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size_returns_single_chunk(self, service):
        text = "A" * 500
        chunks = service.chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_splits_into_multiple_chunks(self, service):
        text = "A" * 1000
        chunks = service.chunk_text(text)
        assert len(chunks) >= 2

    def test_chunks_have_overlap(self, service):
        text = "A" * 1000
        chunks = service.chunk_text(text)
        if len(chunks) >= 2:
            total_length = sum(len(c) for c in chunks)
            assert total_length > len(text)

    def test_splits_at_sentence_boundary(self, service):
        text = "첫 번째 문장입니다. " * 30 + "마지막 문장입니다."
        chunks = service.chunk_text(text)
        for chunk in chunks[:-1]:
            assert chunk.endswith(".") or chunk.endswith("!")

    def test_handles_korean_delimiters(self, service):
        text = "한글 문장입니다。" * 100
        chunks = service.chunk_text(text)
        assert len(chunks) >= 2

    def test_handles_newlines_as_delimiter(self, service):
        text = ("줄바꿈 문장입니다\n" * 100).strip()
        chunks = service.chunk_text(text)
        assert len(chunks) >= 2

    def test_strips_whitespace_from_chunks(self, service):
        text = "  문장1.  문장2.  " * 50
        chunks = service.chunk_text(text)
        for chunk in chunks:
            assert chunk == chunk.strip()

    def test_no_empty_chunks(self, service):
        text = "문장입니다. " * 100
        chunks = service.chunk_text(text)
        for chunk in chunks:
            assert chunk.strip() != ""


class TestGenerateEmbedding:
    @pytest.mark.asyncio
    async def test_successful_embedding_generation(self):
        """임베딩 생성 성공."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_result = MagicMock()
            mock_result.embeddings = [mock_embedding]
            mock_client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_result)

            service = EmbeddingService()
            embedding = await service.generate_embedding("테스트 텍스트")

            assert len(embedding) == 768
            assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self):
        """임베딩 생성 실패 시 RuntimeError."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_client.return_value.aio.models.embed_content = AsyncMock(
                side_effect=Exception("API Error")
            )

            service = EmbeddingService()

            with pytest.raises(RuntimeError) as exc_info:
                await service.generate_embedding("테스트 텍스트")

            assert "임베딩 생성에 실패했습니다" in str(exc_info.value)


class TestGenerateQueryEmbedding:
    @pytest.mark.asyncio
    async def test_successful_query_embedding(self):
        """쿼리 임베딩 생성 성공."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.2] * 768
            mock_result = MagicMock()
            mock_result.embeddings = [mock_embedding]
            mock_client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_result)

            service = EmbeddingService()
            embedding = await service.generate_query_embedding("검색 쿼리")

            assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_query_embedding_failure(self):
        """쿼리 임베딩 생성 실패 시 RuntimeError."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_client.return_value.aio.models.embed_content = AsyncMock(
                side_effect=Exception("API Error")
            )

            service = EmbeddingService()

            with pytest.raises(RuntimeError) as exc_info:
                await service.generate_query_embedding("검색 쿼리")

            assert "쿼리 임베딩 생성에 실패했습니다" in str(exc_info.value)


class TestProcessStorySummary:
    @pytest.mark.asyncio
    async def test_process_story_with_chunks(self):
        """스토리 요약 처리 및 청크 저장."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_result = MagicMock()
            mock_result.embeddings = [mock_embedding]
            mock_client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_result)

            mock_db = AsyncMock()

            service = EmbeddingService()
            count = await service.process_story_summary(
                db=mock_db,
                story_id="story-123",
                summary="짧은 요약 텍스트입니다.",
            )

            assert count == 1
            mock_db.execute.assert_called()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_empty_summary(self):
        """빈 요약은 0 반환."""
        with patch("src.services.embedding_service.genai.Client"):
            mock_db = AsyncMock()

            service = EmbeddingService()
            count = await service.process_story_summary(
                db=mock_db,
                story_id="story-123",
                summary="",
            )

            assert count == 0

    @pytest.mark.asyncio
    async def test_process_long_summary_creates_multiple_chunks(self):
        """긴 요약은 여러 청크 생성."""
        with patch("src.services.embedding_service.genai.Client") as mock_client:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_result = MagicMock()
            mock_result.embeddings = [mock_embedding]
            mock_client.return_value.aio.models.embed_content = AsyncMock(return_value=mock_result)

            mock_db = AsyncMock()

            service = EmbeddingService()
            long_summary = "이것은 아주 긴 스토리 요약입니다. " * 100

            count = await service.process_story_summary(
                db=mock_db,
                story_id="story-123",
                summary=long_summary,
            )

            assert count >= 2


class TestGetStoryChunkCount:
    @pytest.mark.asyncio
    async def test_get_chunk_count(self):
        """청크 수 조회."""
        with patch("src.services.embedding_service.genai.Client"):
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 5
            mock_db.execute.return_value = mock_result

            service = EmbeddingService()
            count = await service.get_story_chunk_count(mock_db, "story-123")

            assert count == 5

    @pytest.mark.asyncio
    async def test_get_chunk_count_returns_zero_when_none(self):
        """청크 없으면 0 반환."""
        with patch("src.services.embedding_service.genai.Client"):
            mock_db = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = None
            mock_db.execute.return_value = mock_result

            service = EmbeddingService()
            count = await service.get_story_chunk_count(mock_db, "story-123")

            assert count == 0
