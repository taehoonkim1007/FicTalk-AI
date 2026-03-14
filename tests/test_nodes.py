from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graphs.nodes import (
    _build_chat_history,
    _evaluate_context_relevance,
    generate_creative_response,
    generate_rag_response,
    retrieve_and_evaluate,
)


class TestBuildChatHistory:
    """_build_chat_history 함수 테스트."""

    def test_converts_user_role(self):
        """user 역할이 올바르게 변환된다."""
        messages = [{"role": "user", "content": "안녕하세요"}]
        result = _build_chat_history(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].text == "안녕하세요"

    def test_converts_assistant_to_model(self):
        """assistant 역할이 model로 변환된다."""
        messages = [{"role": "assistant", "content": "반갑습니다"}]
        result = _build_chat_history(messages)

        assert len(result) == 1
        assert result[0].role == "model"
        assert result[0].parts[0].text == "반갑습니다"

    def test_limits_to_last_5_messages(self):
        """메시지가 5개를 초과하면 최근 5개만 반환한다."""
        messages = [{"role": "user", "content": f"메시지 {i}"} for i in range(10)]
        result = _build_chat_history(messages)

        assert len(result) == 5
        assert result[0].parts[0].text == "메시지 5"
        assert result[4].parts[0].text == "메시지 9"

    def test_handles_empty_messages(self):
        """빈 메시지 리스트를 처리한다."""
        result = _build_chat_history([])
        assert len(result) == 0

    def test_handles_mixed_roles(self):
        """user와 assistant가 혼합된 대화를 처리한다."""
        messages = [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
            {"role": "user", "content": "추가 질문"},
        ]
        result = _build_chat_history(messages)

        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "model"
        assert result[2].role == "user"

    def test_handles_empty_content(self):
        """빈 content를 처리한다."""
        messages = [{"role": "user", "content": ""}]
        result = _build_chat_history(messages)

        assert len(result) == 1
        assert result[0].parts[0].text == ""

    def test_handles_missing_content(self):
        """content가 없는 경우 빈 문자열로 처리한다."""
        messages = [{"role": "user"}]
        result = _build_chat_history(messages)

        assert len(result) == 1
        assert result[0].parts[0].text == ""


class TestEvaluateContextRelevance:
    """_evaluate_context_relevance 함수 테스트."""

    @pytest.mark.asyncio
    async def test_returns_true_for_yes_response(self):
        """'yes'로 시작하는 응답은 True를 반환한다."""
        mock_response = MagicMock()
        mock_response.text = "Yes, the context is relevant."

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_no_response(self):
        """'no'로 시작하는 응답은 False를 반환한다."""
        mock_response = MagicMock()
        mock_response.text = "No, the context is not relevant."

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self):
        """응답이 없으면 None을 반환한다."""
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문", max_retries=0)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """예외 발생 시 None을 반환한다."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("API 오류"))

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문", max_retries=0)

        assert result is None

    @pytest.mark.asyncio
    async def test_extracts_text_from_candidates(self):
        """response.text가 None일 때 candidates에서 텍스트를 추출한다."""
        mock_part = MagicMock()
        mock_part.text = "yes"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = None

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문")

        assert result is True

    @pytest.mark.asyncio
    async def test_case_insensitive_yes_detection(self):
        """대소문자 구분 없이 'yes'를 감지한다."""
        mock_response = MagicMock()
        mock_response.text = "YES, absolutely relevant."

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await _evaluate_context_relevance("컨텍스트", "질문")

        assert result is True


class TestRetrieveAndEvaluate:
    """retrieve_and_evaluate 함수 테스트."""

    @pytest.fixture
    def base_state(self):
        return {
            "character_name": "홍길동",
            "character_role": "주인공",
            "character_personality": "용감한",
            "story_id": "story-123",
            "story_title": "홍길동전",
            "story_summary": "조선시대 의적 이야기",
            "user_message": "활빈당에 대해 알려줘",
            "messages": [],
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
            "response": "",
            "used_rag": False,
        }

    @pytest.mark.asyncio
    async def test_returns_creative_mode_when_no_results(self, base_state):
        """RAG 검색 결과가 없으면 creative 모드를 반환한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(return_value=[])
        mock_db = AsyncMock()

        result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "creative"
        assert result["rag_results"] == []
        assert result["max_similarity"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_creative_mode_when_low_similarity(self, base_state):
        """유사도가 낮으면 creative 모드를 반환한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(
            return_value=[("청크 내용", 0.3)]  # 낮은 유사도
        )
        mock_db = AsyncMock()

        result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "creative"
        assert result["max_similarity"] == 0.3

    @pytest.mark.asyncio
    async def test_returns_rag_mode_when_context_relevant(self, base_state):
        """컨텍스트가 관련성이 있으면 rag 모드를 반환한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(
            return_value=[("활빈당은 의적 집단입니다.", 0.85)]
        )
        mock_db = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = "yes"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "rag"
        assert result["max_similarity"] == 0.85

    @pytest.mark.asyncio
    async def test_returns_creative_mode_when_context_irrelevant(self, base_state):
        """컨텍스트가 관련성이 없으면 creative 모드를 반환한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(return_value=[("관련 없는 내용", 0.85)])
        mock_db = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = "no"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "creative"

    @pytest.mark.asyncio
    async def test_fallback_to_rag_when_evaluation_fails_high_similarity(self, base_state):
        """평가 실패 시 높은 유사도면 rag 모드로 폴백한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(
            return_value=[("내용", 0.7)]  # FALLBACK_RAG_THRESHOLD 이상
        )
        mock_db = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "rag"

    @pytest.mark.asyncio
    async def test_fallback_to_creative_when_evaluation_fails_low_similarity(self, base_state):
        """평가 실패 시 낮은 유사도면 creative 모드로 폴백한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(
            return_value=[("내용", 0.55)]  # FALLBACK_RAG_THRESHOLD 미만
        )
        mock_db = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("src.graphs.nodes._get_genai_client", return_value=mock_client):
            result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "creative"

    @pytest.mark.asyncio
    async def test_returns_creative_mode_on_exception(self, base_state):
        """예외 발생 시 creative 모드를 반환한다."""
        mock_rag_service = MagicMock()
        mock_rag_service.search_with_reranking = AsyncMock(side_effect=Exception("DB 오류"))
        mock_db = AsyncMock()

        result = await retrieve_and_evaluate(base_state, mock_rag_service, mock_db)

        assert result["mode"] == "creative"
        assert result["rag_results"] == []


class TestGenerateRagResponse:
    """generate_rag_response 함수 테스트."""

    @pytest.fixture
    def rag_state(self):
        return {
            "character_name": "홍길동",
            "character_role": "주인공",
            "character_personality": "용감하고 정의로운",
            "story_id": "story-123",
            "story_title": "홍길동전",
            "story_summary": "조선시대 의적 이야기",
            "user_message": "활빈당에 대해 알려줘",
            "messages": [{"role": "user", "content": "안녕"}],
            "rag_results": [("활빈당은 의적 집단입니다.", 0.85)],
            "max_similarity": 0.85,
            "mode": "rag",
            "response": "",
            "used_rag": False,
        }

    @pytest.mark.asyncio
    async def test_generates_response_with_rag_context(self, rag_state):
        """RAG 컨텍스트를 포함한 응답을 생성한다."""
        mock_response = MagicMock()
        mock_response.text = "활빈당은 가난한 사람들을 돕는 의적 집단이지."
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"))]

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with (
            patch("src.graphs.nodes._get_genai_client", return_value=mock_client),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {
                "response": "활빈당은 가난한 사람들을 돕는 의적 집단이지.",
                "used_rag": True,
            }
            result = await generate_rag_response(rag_state)

        assert result["used_rag"] is True
        assert "활빈당" in result["response"]

    @pytest.mark.asyncio
    async def test_filters_low_similarity_results(self, rag_state):
        """유사도가 0.5 미만인 결과는 필터링된다."""
        rag_state["rag_results"] = [
            ("높은 유사도 청크", 0.85),
            ("낮은 유사도 청크", 0.3),
        ]

        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {"response": "응답", "used_rag": True}
            await generate_rag_response(rag_state)

    @pytest.mark.asyncio
    async def test_removes_character_name_prefix(self, rag_state):
        """응답에서 캐릭터 이름 접두사를 제거한다."""
        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {
                "response": "나는 활빈당의 수장이다.",
                "used_rag": True,
            }
            result = await generate_rag_response(rag_state)

        assert not result["response"].startswith("홍길동:")

    @pytest.mark.asyncio
    async def test_handles_empty_rag_results(self, rag_state):
        """RAG 결과가 비어있어도 응답을 생성한다."""
        rag_state["rag_results"] = []

        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {"response": "응답", "used_rag": True}
            result = await generate_rag_response(rag_state)

        assert result["used_rag"] is True


class TestGenerateCreativeResponse:
    """generate_creative_response 함수 테스트."""

    @pytest.fixture
    def creative_state(self):
        return {
            "character_name": "홍길동",
            "character_role": "주인공",
            "character_personality": "용감하고 정의로운",
            "story_id": "story-123",
            "story_title": "홍길동전",
            "story_summary": "조선시대 의적 이야기",
            "user_message": "오늘 날씨가 어때?",
            "messages": [],
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
            "response": "",
            "used_rag": False,
        }

    @pytest.mark.asyncio
    async def test_generates_creative_response(self, creative_state):
        """창의적 응답을 생성한다."""
        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {
                "response": "오늘은 하늘이 맑구나!",
                "used_rag": False,
            }
            result = await generate_creative_response(creative_state)

        assert result["used_rag"] is False
        assert "하늘" in result["response"]

    @pytest.mark.asyncio
    async def test_removes_character_name_prefix(self, creative_state):
        """응답에서 캐릭터 이름 접두사를 제거한다."""
        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {
                "response": "날씨가 좋구나.",
                "used_rag": False,
            }
            result = await generate_creative_response(creative_state)

        assert not result["response"].startswith("홍길동:")

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self, creative_state):
        """이전 대화가 없어도 응답을 생성한다."""
        creative_state["messages"] = []

        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {"response": "응답", "used_rag": False}
            result = await generate_creative_response(creative_state)

        assert result["used_rag"] is False

    @pytest.mark.asyncio
    async def test_handles_none_personality(self, creative_state):
        """성격이 None이어도 응답을 생성한다."""
        creative_state["character_personality"] = None

        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {"response": "응답", "used_rag": False}
            result = await generate_creative_response(creative_state)

        assert result["used_rag"] is False

    @pytest.mark.asyncio
    async def test_handles_none_story_summary(self, creative_state):
        """줄거리가 None이어도 응답을 생성한다."""
        creative_state["story_summary"] = None

        with (
            patch("src.graphs.nodes._get_genai_client"),
            patch("src.graphs.nodes.retry_api_call_dict") as mock_retry,
        ):
            mock_retry.return_value = {"response": "응답", "used_rag": False}
            result = await generate_creative_response(creative_state)

        assert result["used_rag"] is False
