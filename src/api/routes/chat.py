from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_chat_service
from src.models.database import get_db
from src.models.schemas import ChatRequest, ChatResponse
from src.services.chat_service import ChatService

router = APIRouter()


@router.post("/response", response_model=ChatResponse)
async def generate_chat_response(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """캐릭터 채팅 응답 생성 API (LangGraph 기반 RAG/Creative 모드).

    캐릭터 정보와 대화 내역을 기반으로 캐릭터로서의 응답을 생성합니다.

    **동작 방식:**
    - **1단계**: RAG 검색 수행 (유사도 < 0.5인 경우 즉시 Creative 모드)
    - **2단계**: 검색된 정보의 적합성을 LLM이 스스로 평가 (Self-RAG)
        - 적합함 → RAG 모드 (줄거리 기반)
        - 부적합함 → Creative 모드 (캐릭터 성격 기반)
    - **3단계**: 평가 실패 시 유사도 0.63 기준으로 폴백 결정

    **파라미터:**
    - **character_name**: 캐릭터 이름
    - **character_role**: 캐릭터 역할 (주인공, 조연 등)
    - **character_personality**: 캐릭터 성격
    - **story_id**: 스토리 ID (RAG 검색용)
    - **story_title**: 소속 스토리 제목
    - **story_summary**: 스토리 줄거리
    - **messages**: 이전 대화 내역 (session_id가 있으면 생략 가능)
    - **user_message**: 사용자가 보낸 새 메시지
    - **session_id**: 세션 ID (캐시된 대화 사용 시)

    **응답:**
    - **response**: 캐릭터의 응답
    - **mode**: 응답 모드 ('rag' 또는 'creative')
    - **used_rag**: RAG 컨텍스트 사용 여부
    - **max_similarity**: RAG 검색 최대 유사도 점수
    - **session_id**: 세션 ID (다음 요청에 사용)
    """

    try:
        messages = (
            [{"role": m.role, "content": m.content} for m in request.messages]
            if request.messages
            else None
        )

        result = await service.run(
            character_name=request.character_name,
            character_role=request.character_role,
            character_personality=request.character_personality,
            story_id=request.story_id,
            story_title=request.story_title,
            story_summary=request.story_summary,
            messages=messages,
            user_message=request.user_message,
            db=db,
            session_id=request.session_id,
        )

        return ChatResponse(
            response=result["response"],
            mode=result["mode"],
            used_rag=result["used_rag"],
            max_similarity=result["max_similarity"],
            session_id=result["session_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
