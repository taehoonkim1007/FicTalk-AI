from fastapi import APIRouter, HTTPException

from src.models.schemas import ChatResponseRequest, ChatResponseResponse
from src.services.chat_service import ChatService

router = APIRouter()

# 싱글톤 인스턴스 (캐시 유지)
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """ChatService 싱글톤 인스턴스 반환."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


@router.post("/response", response_model=ChatResponseResponse)
async def generate_chat_response(request: ChatResponseRequest) -> ChatResponseResponse:
    """캐릭터 채팅 응답 생성 API.

    캐릭터 정보와 대화 내역을 기반으로 캐릭터로서의 응답을 생성합니다.

    - **character_name**: 캐릭터 이름
    - **character_role**: 캐릭터 역할 (주인공, 조연 등)
    - **character_personality**: 캐릭터 성격
    - **story_title**: 소속 스토리 제목
    - **story_summary**: 스토리 줄거리
    - **messages**: 이전 대화 내역
    - **user_message**: 사용자가 보낸 새 메시지
    """
    service = get_chat_service()
    try:
        return await service.generate_response(
            character_name=request.character_name,
            character_role=request.character_role,
            character_personality=request.character_personality,
            story_title=request.story_title,
            story_summary=request.story_summary,
            messages=request.messages,
            user_message=request.user_message,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
