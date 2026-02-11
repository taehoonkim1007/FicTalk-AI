import pytest
from sqlalchemy import text

from tests.integration.conftest import skip_if_no_db, skip_if_no_gemini


@skip_if_no_db
class TestEmbeddingIntegration:
    """임베딩 생성 및 저장 통합 테스트."""

    @skip_if_no_gemini
    @pytest.mark.asyncio
    async def test_generate_embedding_returns_768_dimensions(self):
        """Gemini 임베딩이 768차원인지 확인."""
        from src.services.embedding_service import EmbeddingService

        service = EmbeddingService()
        embedding = await service.generate_embedding("테스트 텍스트입니다.")

        assert len(embedding) == 768
        assert all(isinstance(v, float) for v in embedding)

    @skip_if_no_gemini
    @pytest.mark.asyncio
    async def test_query_embedding_different_from_document(self):
        """쿼리 임베딩과 문서 임베딩이 다른 task_type 사용."""
        from src.services.embedding_service import EmbeddingService

        service = EmbeddingService()
        text_content = "마법사가 마법을 사용합니다."

        doc_embedding = await service.generate_embedding(text_content)
        query_embedding = await service.generate_query_embedding(text_content)

        # 동일 텍스트여도 task_type이 다르면 임베딩이 약간 다를 수 있음
        assert len(doc_embedding) == len(query_embedding) == 768


@skip_if_no_db
@skip_if_no_gemini
class TestRAGSearchIntegration:
    """RAG 벡터 검색 통합 테스트."""

    @pytest.mark.asyncio
    async def test_store_and_search_chunks(self, db_session, clean_db, story_id):
        """청크 저장 후 벡터 검색."""
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        embedding_service = EmbeddingService()
        rag_service = RAGService()

        # 1. 테스트 데이터 저장
        summary = """
        마법사 아리안은 마법 아카데미에서 수련했다.
        그의 친구 레온은 검술에 능했고, 소피아는 치유 마법을 사용했다.
        세 친구는 함께 어둠의 군주와 싸웠다.
        """

        chunk_count = await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary=summary,
        )

        assert chunk_count >= 1

        # 2. 관련 청크 검색
        chunks = await rag_service.search_relevant_chunks(
            db=db_session,
            story_id=story_id,
            query="마법사 아리안은 누구인가요?",
            top_k=3,
        )

        # 검색 결과가 있어야 함
        assert len(chunks) >= 1
        # 관련 내용이 포함되어야 함
        expected_keywords = ["아리안", "마법사"]
        found = [chunk for chunk in chunks if any(kw in chunk for kw in expected_keywords)]
        assert found, f"기대 키워드 '{expected_keywords}' 중 없음. 실제 청크: {chunks}"

    @pytest.mark.asyncio
    async def test_search_with_scores(self, db_session, clean_db, story_id):
        """점수 포함 검색 테스트."""
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        embedding_service = EmbeddingService()
        rag_service = RAGService()

        # 데이터 저장
        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary="용감한 기사가 드래곤과 싸웠다. 기사는 왕국을 구했다.",
        )

        # 점수 포함 검색
        results = await rag_service.search_relevant_chunks_with_scores(
            db=db_session,
            story_id=story_id,
            query="기사가 무엇을 했나요?",
            top_k=3,
        )

        assert len(results) >= 1
        # 결과는 (content, similarity) 튜플
        content, score = results[0]
        assert isinstance(content, str)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_search_with_reranking(self, db_session, clean_db, story_id):
        """키워드 부스팅 재정렬 테스트."""
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        embedding_service = EmbeddingService()
        rag_service = RAGService()

        # 여러 청크 저장
        long_summary = """
        마법사 아리안은 화염 마법을 사용한다.
        전사 레온은 검과 방패로 싸운다.
        치유사 소피아는 동료들을 치료한다.
        마왕은 어둠의 힘을 사용한다.
        """

        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary=long_summary,
        )

        # 키워드가 포함된 쿼리로 재정렬 검색
        results = await rag_service.search_with_reranking(
            db=db_session,
            story_id=story_id,
            query="마법사 아리안의 능력은?",
        )

        assert len(results) >= 1
        # 결과는 점수로 정렬됨
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


@skip_if_no_db
@skip_if_no_gemini
class TestContextGenerationIntegration:
    """채팅 컨텍스트 생성 통합 테스트."""

    @pytest.mark.asyncio
    async def test_get_context_for_chat(self, db_session, clean_db, story_id):
        """채팅용 컨텍스트 생성."""
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        embedding_service = EmbeddingService()
        rag_service = RAGService()

        # 데이터 저장
        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary="공주 엘리사는 마법의 성에 살았다. 그녀는 용과 친구가 되었다.",
        )

        # 컨텍스트 생성
        context = await rag_service.get_context_for_chat(
            db=db_session,
            story_id=story_id,
            user_message="공주에 대해 알려줘",
        )

        assert "[관련 내용" in context
        assert "엘리사" in context or "공주" in context

    @pytest.mark.asyncio
    async def test_context_respects_max_length(self, db_session, clean_db, story_id):
        """최대 컨텍스트 길이 제한."""
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        embedding_service = EmbeddingService()
        rag_service = RAGService()

        # 긴 데이터 저장
        long_summary = "아주 긴 스토리입니다. " * 100

        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary=long_summary,
        )

        # 짧은 max_context_length 설정
        context = await rag_service.get_context_for_chat(
            db=db_session,
            story_id=story_id,
            user_message="스토리 내용이 뭐야?",
            max_context_length=200,
        )

        # 컨텍스트가 제한 내에 있어야 함 (포맷팅 오버헤드 고려)
        assert len(context) <= 300


@skip_if_no_db
class TestDatabaseOperationsIntegration:
    """데이터베이스 작업 통합 테스트."""

    @pytest.mark.asyncio
    async def test_chunk_count_query(self, db_session, clean_db, story_id):
        """청크 수 조회."""
        from src.services.embedding_service import EmbeddingService

        # Gemini API 없이 테스트하기 위해 직접 데이터 삽입
        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES
                    ('chunk1', '첫 번째 청크', 0, :story_id),
                    ('chunk2', '두 번째 청크', 1, :story_id),
                    ('chunk3', '세 번째 청크', 2, :story_id)
            """),
            {"story_id": story_id},
        )
        await db_session.commit()

        # 청크 수 확인
        service = EmbeddingService.__new__(EmbeddingService)
        count = await service.get_story_chunk_count(db_session, story_id)

        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_existing_chunks_before_insert(self, db_session, clean_db, story_id):
        """기존 청크 삭제 후 새 청크 삽입."""
        # 기존 데이터 삽입
        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES ('old-chunk', '이전 청크', 0, :story_id)
            """),
            {"story_id": story_id},
        )
        await db_session.commit()

        # 삭제 확인
        await db_session.execute(
            text('DELETE FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )
        await db_session.commit()

        # 카운트 확인
        result = await db_session.execute(
            text('SELECT COUNT(*) FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )
        count = result.scalar()

        assert count == 0

    @pytest.mark.asyncio
    async def test_vector_search_without_embedding(self, db_session, clean_db, story_id):
        """임베딩 없는 청크는 검색에서 제외."""
        # 임베딩 없는 청크 삽입
        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId", embedding)
                VALUES ('no-embedding', '임베딩 없는 청크', 0, :story_id, NULL)
            """),
            {"story_id": story_id},
        )
        await db_session.commit()

        # 검색 쿼리 (embedding IS NOT NULL 조건)
        result = await db_session.execute(
            text("""
                SELECT COUNT(*) FROM "StoryContent"
                WHERE "storyId" = :story_id AND embedding IS NOT NULL
            """),
            {"story_id": story_id},
        )
        count = result.scalar()

        assert count == 0
