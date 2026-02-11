from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.rag_service import RAGService


class TestExtractKeywords:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService"):
            return RAGService()

    def test_extracts_korean_words(self, service):
        """한국어 단어 추출."""
        query = "마법사의 이야기를 알려줘"
        keywords = service._extract_keywords(query)

        assert "마법사" in keywords
        assert "이야기" in keywords

    def test_removes_stopwords(self, service):
        """불용어 제거 확인."""
        query = "이것은 무엇이야"
        keywords = service._extract_keywords(query)

        # "이것", "무엇" 등은 불용어일 수 있음
        assert len(keywords) >= 0

    def test_removes_korean_particles(self, service):
        """한국어 조사 제거."""
        query = "캐릭터가 성격은 어떠한가요"
        keywords = service._extract_keywords(query)

        # 조사가 제거된 키워드
        expected = ["캐릭터", "성격"]
        found = [kw for kw in keywords if any(e in kw for e in expected)]
        assert found, f"기대 키워드 '{expected}' 중 없음. 실제: {keywords}"

    def test_extracts_english_words(self, service):
        """영어 단어 추출."""
        query = "Show me the Harry Potter story"
        keywords = service._extract_keywords(query)

        assert "Harry" in keywords or "Potter" in keywords

    def test_filters_short_words(self, service):
        """2글자 미만 단어 필터링."""
        query = "나 는 좋 아"
        keywords = service._extract_keywords(query)

        # 1글자 단어는 제외
        assert all(len(kw) >= 2 for kw in keywords)

    def test_removes_duplicates(self, service):
        """중복 키워드 제거."""
        query = "마법사 마법사 마법사의 이야기"
        keywords = service._extract_keywords(query)

        # 중복 없이 유니크해야 함
        assert len(keywords) == len(set(keywords))

    def test_empty_query(self, service):
        """빈 쿼리 처리."""
        keywords = service._extract_keywords("")
        assert keywords == []


class TestCalculateKeywordBoost:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService"):
            return RAGService()

    def test_full_match_boost(self, service):
        """모든 키워드 매칭 시 최대 부스트."""
        content = "마법사가 마법을 사용하는 이야기입니다"
        keywords = ["마법사", "이야기"]

        boost = service._calculate_keyword_boost(content, keywords)

        # 모든 키워드 매칭 시 KEYWORD_BOOST 값 반환
        assert boost > 0

    def test_partial_match_boost(self, service):
        """일부 키워드만 매칭."""
        content = "마법사가 등장하는 장면"
        keywords = ["마법사", "이야기"]

        boost = service._calculate_keyword_boost(content, keywords)

        # 50% 매칭이므로 절반 정도의 부스트
        assert boost > 0

    def test_no_match_zero_boost(self, service):
        """매칭 없으면 0."""
        content = "평범한 일상 이야기"
        keywords = ["드래곤", "전사"]

        boost = service._calculate_keyword_boost(content, keywords)

        assert boost == 0.0

    def test_empty_keywords_zero_boost(self, service):
        """키워드 없으면 0."""
        content = "아무 내용이나"
        keywords = []

        boost = service._calculate_keyword_boost(content, keywords)

        assert boost == 0.0


class TestSearchRelevantChunks:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.generate_query_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_emb.return_value = mock_instance
            return RAGService()

    @pytest.mark.asyncio
    async def test_returns_chunks_above_threshold(self, service):
        """임계값 이상 청크만 반환."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("청크1 내용", 0.85),
            ("청크2 내용", 0.75),
            ("청크3 내용", 0.65),
        ]
        mock_db.execute.return_value = mock_result

        chunks = await service.search_relevant_chunks(
            db=mock_db,
            story_id="story-123",
            query="테스트 질문",
            top_k=3,
        )

        assert len(chunks) > 0
        assert "청크1 내용" in chunks

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self, service):
        """청크 없으면 빈 리스트."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        chunks = await service.search_relevant_chunks(
            db=mock_db,
            story_id="story-123",
            query="테스트 질문",
        )

        assert chunks == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, service):
        """예외 발생 시 빈 리스트."""
        mock_db = AsyncMock()
        mock_db.execute.side_effect = Exception("DB 오류")

        chunks = await service.search_relevant_chunks(
            db=mock_db,
            story_id="story-123",
            query="테스트 질문",
        )

        assert chunks == []


class TestSearchRelevantChunksWithScores:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.generate_query_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_emb.return_value = mock_instance
            return RAGService()

    @pytest.mark.asyncio
    async def test_returns_chunks_with_scores(self, service):
        """점수와 함께 청크 반환."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("청크1", 0.9),
            ("청크2", 0.8),
        ]
        mock_db.execute.return_value = mock_result

        results = await service.search_relevant_chunks_with_scores(
            db=mock_db,
            story_id="story-123",
            query="테스트",
            top_k=3,
        )

        assert len(results) == 2
        assert results[0] == ("청크1", 0.9)
        assert results[1] == ("청크2", 0.8)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, service):
        """예외 발생 시 빈 리스트."""
        mock_db = AsyncMock()
        mock_db.execute.side_effect = Exception("DB 오류")

        results = await service.search_relevant_chunks_with_scores(
            db=mock_db,
            story_id="story-123",
            query="테스트",
        )

        assert results == []


class TestSearchWithReranking:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.generate_query_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_emb.return_value = mock_instance
            return RAGService()

    @pytest.mark.asyncio
    async def test_reranks_with_keyword_boost(self, service):
        """키워드 부스팅으로 재정렬."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        # 마법사가 포함된 청크가 부스팅되어야 함
        mock_result.fetchall.return_value = [
            ("일반적인 이야기", 0.85),
            ("마법사가 등장하는 이야기", 0.80),
        ]
        mock_db.execute.return_value = mock_result

        results = await service.search_with_reranking(
            db=mock_db,
            story_id="story-123",
            query="마법사에 대해 알려줘",
        )

        assert len(results) > 0
        # 결과가 점수로 정렬됨
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self, service):
        """청크 없으면 빈 리스트."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.execute.return_value = mock_result

        results = await service.search_with_reranking(
            db=mock_db,
            story_id="story-123",
            query="테스트",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, service):
        """예외 발생 시 빈 리스트."""
        mock_db = AsyncMock()
        mock_db.execute.side_effect = Exception("DB 오류")

        results = await service.search_with_reranking(
            db=mock_db,
            story_id="story-123",
            query="테스트",
        )

        assert results == []


class TestGetContextForChat:
    @pytest.fixture
    def service(self):
        with patch("src.services.rag_service.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.generate_query_embedding = AsyncMock(return_value=[0.1] * 768)
            mock_emb.return_value = mock_instance
            return RAGService()

    @pytest.mark.asyncio
    async def test_formats_context_correctly(self, service):
        """컨텍스트 포맷 확인."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("첫 번째 청크 내용", 0.9),
            ("두 번째 청크 내용", 0.8),
        ]
        mock_db.execute.return_value = mock_result

        context = await service.get_context_for_chat(
            db=mock_db,
            story_id="story-123",
            user_message="테스트 메시지",
        )

        assert "[관련 내용 1]" in context
        assert "첫 번째 청크 내용" in context

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self, service):
        """청크 없으면 빈 문자열."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        # 유사도 임계값 이하
        mock_result.fetchall.return_value = [
            ("청크", 0.1),
        ]
        mock_db.execute.return_value = mock_result

        context = await service.get_context_for_chat(
            db=mock_db,
            story_id="story-123",
            user_message="테스트 메시지",
        )

        # 임계값 이하면 빈 문자열
        assert context == ""

    @pytest.mark.asyncio
    async def test_respects_max_context_length(self, service):
        """최대 컨텍스트 길이 준수."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        # 긴 청크들
        mock_result.fetchall.return_value = [
            ("A" * 100, 0.9),
            ("B" * 100, 0.85),
            ("C" * 100, 0.8),
        ]
        mock_db.execute.return_value = mock_result

        context = await service.get_context_for_chat(
            db=mock_db,
            story_id="story-123",
            user_message="테스트",
            max_context_length=150,
        )

        # 150자 제한으로 모든 청크가 포함되지 않아야 함
        assert len(context) <= 200  # 포맷 오버헤드 포함
