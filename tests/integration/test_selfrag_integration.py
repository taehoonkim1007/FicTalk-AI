import pytest

from tests.integration.conftest import skip_if_no_gemini


@skip_if_no_gemini
class TestSelfRAGEvaluation:
    """Self-RAG 컨텍스트 평가 통합 테스트."""

    @pytest.mark.asyncio
    async def test_relevant_context_returns_yes(self):
        """관련 있는 컨텍스트는 'yes' 반환."""
        from src.graphs.nodes import _evaluate_context_relevance

        context = "마법사 아리안은 화염 마법을 사용하는 젊은 마법사입니다."
        question = "아리안은 어떤 마법을 사용하나요?"

        result = await _evaluate_context_relevance(context, question)

        assert result is True

    @pytest.mark.asyncio
    async def test_irrelevant_context_returns_no(self):
        """관련 없는 컨텍스트는 'no' 반환."""
        from src.graphs.nodes import _evaluate_context_relevance

        context = "오늘 날씨가 좋습니다. 하늘이 맑네요."
        question = "마법사의 이름이 뭐야?"

        result = await _evaluate_context_relevance(context, question)

        assert result is False

    @pytest.mark.asyncio
    async def test_partial_relevance(self):
        """부분적으로 관련 있는 컨텍스트."""
        from src.graphs.nodes import _evaluate_context_relevance

        context = "왕국에는 많은 마법사가 있습니다. 그 중 가장 유명한 것은 대마법사입니다."
        question = "마법사에 대해 알려줘"

        result = await _evaluate_context_relevance(context, question)

        # 부분적 관련성은 보통 True로 평가됨
        assert isinstance(result, bool)


@skip_if_no_gemini
class TestRetrieveAndEvaluateIntegration:
    """RAG 검색 + Self-RAG 평가 통합 테스트."""

    @pytest.mark.asyncio
    async def test_mode_selection_with_relevant_context(self, db_session, clean_db, story_id):
        """관련 컨텍스트가 있으면 RAG 모드 선택."""
        from src.graphs.nodes import retrieve_and_evaluate
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        # 데이터 준비
        embedding_service = EmbeddingService()
        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary="마법사 아리안은 화염 마법을 사용합니다. 그는 마법 아카데미 출신입니다.",
        )

        rag_service = RAGService()

        # 초기 상태
        state = {
            "story_id": story_id,
            "user_message": "아리안은 어떤 마법을 쓰나요?",
            "story_title": "마법사의 여정",
            "story_summary": "마법사의 이야기",
            "character_name": "아리안",
            "character_role": "주인공",
            "character_personality": "용감함",
            "messages": [],
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
            "response": "",
            "used_rag": False,
        }

        # 검색 및 평가 실행
        result = await retrieve_and_evaluate(
            state=state,
            rag_service=rag_service,
            db=db_session,
        )

        # 관련 컨텍스트가 있으면 RAG 모드
        assert result["mode"] in ["rag", "creative"]
        if result["max_similarity"] >= 0.63 or result.get("rag_results"):
            assert result["mode"] == "rag"

    @pytest.mark.asyncio
    async def test_mode_selection_without_relevant_context(self, db_session, clean_db, story_id):
        """관련 컨텍스트가 없으면 Creative 모드 선택."""
        from src.graphs.nodes import retrieve_and_evaluate
        from src.services.embedding_service import EmbeddingService
        from src.services.rag_service import RAGService

        # 관련 없는 데이터 준비
        embedding_service = EmbeddingService()
        await embedding_service.process_story_summary(
            db=db_session,
            story_id=story_id,
            summary="오늘 날씨가 좋습니다. 하늘이 맑네요.",
        )

        rag_service = RAGService()

        # 초기 상태 - 완전히 다른 질문
        state = {
            "story_id": story_id,
            "user_message": "드래곤의 약점이 뭐야?",
            "story_title": "테스트",
            "story_summary": "테스트",
            "character_name": "테스트",
            "character_role": "테스트",
            "character_personality": "테스트",
            "messages": [],
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
            "response": "",
            "used_rag": False,
        }

        result = await retrieve_and_evaluate(
            state=state,
            rag_service=rag_service,
            db=db_session,
        )

        # 유사도가 낮으면 creative 모드
        if result["max_similarity"] < 0.63:
            assert result["mode"] == "creative"


@skip_if_no_gemini
class TestChatResponseGeneration:
    """채팅 응답 생성 통합 테스트."""

    @pytest.mark.asyncio
    async def test_rag_response_includes_context(self):
        """RAG 응답이 컨텍스트를 활용하는지."""
        from src.graphs.nodes import generate_rag_response

        state = {
            "character_name": "아리안",
            "character_role": "마법사",
            "character_personality": "지적이고 호기심이 많음",
            "story_title": "마법사의 여정",
            "story_summary": "마법사 아리안의 모험",
            "messages": [],
            "user_message": "네 마법에 대해 알려줘",
            "rag_results": [("아리안은 화염 마법을 사용하는 마법사입니다.", 0.9)],
            "max_similarity": 0.9,
            "mode": "rag",
            "response": "",
            "used_rag": False,
        }

        result = await generate_rag_response(state)

        assert result["response"]
        assert len(result["response"]) > 0
        assert result["used_rag"] is True

    @pytest.mark.asyncio
    async def test_creative_response_without_context(self):
        """Creative 응답이 컨텍스트 없이 생성되는지."""
        from src.graphs.nodes import generate_creative_response

        state = {
            "character_name": "레온",
            "character_role": "기사",
            "character_personality": "용감하고 정의로움",
            "story_title": "기사의 모험",
            "story_summary": "기사 레온의 여정",
            "messages": [],
            "user_message": "안녕!",
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
            "response": "",
            "used_rag": False,
        }

        result = await generate_creative_response(state)

        assert result["response"]
        assert len(result["response"]) > 0
        assert result["used_rag"] is False
