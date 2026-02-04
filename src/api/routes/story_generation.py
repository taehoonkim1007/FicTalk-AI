from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_story_generation_service
from src.models.schemas import (
    CharacterGenerationRequest,
    CharacterGenerationResponse,
    SummaryGenerationRequest,
    SummaryGenerationResponse,
)
from src.services.story_generation_service import StoryGenerationService

router = APIRouter()


@router.post("/summary", response_model=SummaryGenerationResponse)
async def generate_summary(
    request: SummaryGenerationRequest,
    service: StoryGenerationService = Depends(get_story_generation_service),
) -> SummaryGenerationResponse:
    """줄거리 생성 API.

    제목과 한줄 요약을 기반으로 5단계 구조의 줄거리를 생성합니다.
    - **title**: 스토리 제목 (최대 100자)
    - **description**: 스토리 한줄 요약 (최대 400자)
    """
    try:
        return await service.generate_summary(request.title, request.description)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/characters", response_model=CharacterGenerationResponse)
async def generate_characters(
    request: CharacterGenerationRequest,
    service: StoryGenerationService = Depends(get_story_generation_service),
) -> CharacterGenerationResponse:
    """캐릭터 생성 API.

    스토리 정보를 기반으로 캐릭터들을 생성합니다.
    - **title**: 스토리 제목 (최대 100자)
    - **description**: 스토리 한줄 요약 (최대 400자)
    - **summary**: 스토리 줄거리 (최대 4000자)

    AI가 줄거리 → 한줄요약 → 제목 순으로 캐릭터를 파악하여 1~6명을 생성합니다.
    역할은 "주인공" 또는 "조연"으로 분류됩니다.
    """
    try:
        return await service.generate_characters(
            request.title, request.description, request.summary
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
