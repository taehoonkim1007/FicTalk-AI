import asyncio
import logging

from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.constants.error_messages import ERROR_CHAT_GENERATION
from src.common.constants.operation_names import OP_CREATIVE_RESPONSE, OP_RAG_RESPONSE
from src.common.constants.settings import FALLBACK_RAG_THRESHOLD, MIN_SIMILARITY_FOR_EVALUATION
from src.config import settings
from src.graphs.state import ChatState
from src.services.rag_service import RAGService
from src.utils.prompts import (
    CHAT_SYSTEM_PROMPT,
    CREATIVE_MODE_PROMPT,
    SELF_RAG_EVALUATION_PROMPT,
)
from src.utils.retry import retry_api_call_dict

logger = logging.getLogger(__name__)


def _get_genai_client() -> genai.Client:
    """Google GenAI 클라이언트 생성."""
    return genai.Client(api_key=settings.google_api_key)


def _build_chat_history(messages: list[dict]) -> list[types.Content]:
    """대화 히스토리를 GenAI 형식으로 변환.

    백엔드에서 'assistant'로 전송되는 role을 Gemini API의 'model'로 변환합니다.
    """
    history = []
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    for msg in recent_messages:
        role = "user" if msg.get("role") == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg.get("content", ""))]))
    return history


async def _evaluate_context_relevance(
    context: str,
    question: str,
    max_retries: int = 2,
) -> bool | None:
    """Self-RAG: LLM이 컨텍스트가 질문에 답할 수 있는지 평가.

    Returns:
        True: 컨텍스트로 답변 가능 (RAG 모드)
        False: 컨텍스트로 답변 불가 (Creative 모드)
        None: 평가 실패 (폴백 로직 사용)
    """
    client = _get_genai_client()

    prompt = SELF_RAG_EVALUATION_PROMPT.format(
        context=context,
        question=question,
    )

    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )

            # 응답 텍스트 추출 시도
            response_text = None
            if response.text is not None:
                response_text = response.text
            elif response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    logger.warning(f"Self-RAG 응답 종료 이유: {candidate.finish_reason}")

            if response_text is None:
                if attempt < max_retries:
                    logger.warning(f"Self-RAG 평가: 응답 없음, 재시도 {attempt + 1}/{max_retries}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                logger.warning("Self-RAG 평가: 응답 없음 (최대 재시도 도달)")
                return None

            answer = response_text.strip().lower()
            is_relevant = answer.startswith("yes")

            logger.info(
                f"Self-RAG 평가: '{answer}' → {'RAG 모드' if is_relevant else 'Creative 모드'}"
            )
            return is_relevant

        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Self-RAG 평가 오류, 재시도 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            logger.error(f"Self-RAG 평가 실패: {e}")
            return None

    return None


async def retrieve_and_evaluate(
    state: ChatState,
    rag_service: RAGService,
    db: AsyncSession,
) -> dict:
    """RAG 검색 및 Self-RAG 평가 노드.

    1. 사용자 메시지로 관련 청크를 검색
    2. 유사도가 충분하면 LLM이 컨텍스트 관련성 평가 (Self-RAG)
    3. 평가 결과에 따라 RAG/Creative 모드 결정
    """
    try:
        # RAG 검색 (Reranking 적용)
        results = await rag_service.search_with_reranking(
            db=db,
            story_id=state["story_id"],
            query=state["user_message"],
        )

        if not results:
            logger.info("RAG 검색 결과 없음, Creative 모드로 전환")
            return {
                "rag_results": [],
                "max_similarity": 0.0,
                "mode": "creative",
            }

        max_sim = max(score for _, score in results)
        logger.info(f"RAG 검색 완료: {len(results)}개 청크, 최대 유사도: {max_sim:.3f}")

        # 유사도가 너무 낮으면 평가 없이 Creative 모드
        if max_sim < MIN_SIMILARITY_FOR_EVALUATION:
            logger.info(f"유사도 {max_sim:.3f} < {MIN_SIMILARITY_FOR_EVALUATION}, Creative 모드")
            return {
                "rag_results": results,
                "max_similarity": max_sim,
                "mode": "creative",
            }

        # Self-RAG: LLM이 컨텍스트 관련성 평가
        context_for_eval = "\n\n".join([content for content, _ in results])
        is_relevant = await _evaluate_context_relevance(
            context=context_for_eval,
            question=state["user_message"],
        )

        # Self-RAG 평가 결과에 따라 모드 결정
        if is_relevant is None:
            # 평가 실패 시 유사도 기반 폴백
            mode = "rag" if max_sim >= FALLBACK_RAG_THRESHOLD else "creative"
            logger.info(f"Self-RAG 평가 실패, 유사도 폴백: {max_sim:.3f} → {mode}")
        else:
            mode = "rag" if is_relevant else "creative"
            logger.info(f"Self-RAG 결과: 모드 = {mode}")

        return {
            "rag_results": results,
            "max_similarity": max_sim,
            "mode": mode,
        }

    except Exception as e:
        logger.error(f"RAG 검색/평가 실패: {e}")
        return {
            "rag_results": [],
            "max_similarity": 0.0,
            "mode": "creative",
        }


async def generate_rag_response(state: ChatState) -> dict:
    """RAG 모드 응답 생성 노드.

    검색된 컨텍스트를 활용하여 줄거리 기반의 정확한 응답을 생성합니다.
    Rate limit 발생 시 지수 백오프로 재시도합니다.
    """
    client = _get_genai_client()

    # RAG 컨텍스트 구성
    context_parts = []
    for i, (content, score) in enumerate(state["rag_results"], 1):
        if score >= 0.5:  # 최소 유사도 필터
            context_parts.append(f"[관련 내용 {i}]\n{content}")

    rag_context = "\n\n".join(context_parts) if context_parts else ""

    # 시스템 프롬프트 구성
    system_prompt = CHAT_SYSTEM_PROMPT.format(
        character_name=state["character_name"],
        character_role=state["character_role"],
        character_personality=state["character_personality"] or "특별한 성격 설정 없음",
        story_title=state["story_title"],
        story_summary=state["story_summary"] or "줄거리 정보 없음",
    )

    # 대화 히스토리 구성
    chat_history = _build_chat_history(state["messages"])

    # 사용자 메시지에 RAG 컨텍스트 포함
    message_to_send = state["user_message"]
    if rag_context:
        message_to_send = f"[참고 정보]\n{rag_context}\n\n[사용자 메시지]\n{state['user_message']}"
        logger.info(f"RAG 모드: {len(rag_context)}자 컨텍스트 포함")

    # 현재 메시지 추가
    chat_history.append(types.Content(role="user", parts=[types.Part(text=message_to_send)]))

    # API 호출 함수 정의
    async def _call_api() -> dict:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )

        # finish_reason 로깅 (디버깅용)
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason.name != "STOP":
                logger.warning(f"RAG 응답 종료 이유: {finish_reason.name}")

        response_text = response.text.strip()

        # 캐릭터 이름 접두사 제거
        if response_text.startswith(f"{state['character_name']}:"):
            response_text = response_text[len(f"{state['character_name']}:") :].strip()

        return {"response": response_text, "used_rag": True}

    # 재시도 로직 적용
    result = await retry_api_call_dict(
        _call_api,
        operation_name=OP_RAG_RESPONSE,
        error_message=ERROR_CHAT_GENERATION,
    )

    logger.info(f"RAG 모드 응답 생성 완료: {state['character_name']}")

    return result


async def generate_creative_response(state: ChatState) -> dict:
    """Creative 모드 응답 생성 노드.

    줄거리에 없는 질문에 대해 캐릭터의 성격과 세계관을 기반으로
    창의적으로 상상하여 응답합니다.
    Rate limit 발생 시 지수 백오프로 재시도합니다.
    """
    client = _get_genai_client()

    # Creative 모드 전용 시스템 프롬프트
    system_prompt = CREATIVE_MODE_PROMPT.format(
        character_name=state["character_name"],
        character_role=state["character_role"],
        character_personality=state["character_personality"] or "특별한 성격 설정 없음",
        story_title=state["story_title"],
        story_summary=state["story_summary"] or "줄거리 정보 없음",
    )

    # 대화 히스토리 구성
    chat_history = _build_chat_history(state["messages"])

    # 현재 메시지 추가
    chat_history.append(types.Content(role="user", parts=[types.Part(text=state["user_message"])]))

    # API 호출 함수 정의
    async def _call_api() -> dict:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.8,
                max_output_tokens=2048,
            ),
        )

        # finish_reason 로깅 (디버깅용)
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason.name != "STOP":
                logger.warning(f"Creative 응답 종료 이유: {finish_reason.name}")

        response_text = response.text.strip()

        # 캐릭터 이름 접두사 제거
        if response_text.startswith(f"{state['character_name']}:"):
            response_text = response_text[len(f"{state['character_name']}:") :].strip()

        return {"response": response_text, "used_rag": False}

    # 재시도 로직 적용
    result = await retry_api_call_dict(
        _call_api,
        operation_name=OP_CREATIVE_RESPONSE,
        error_message=ERROR_CHAT_GENERATION,
    )

    logger.info(f"Creative 모드 응답 생성 완료: {state['character_name']}")

    return result
