from fastapi import APIRouter, HTTPException

from src.models.schemas import (
    CharacterImageRequest,
    ImageGenerationResponse,
    StoryImageRequest,
)
from src.services.image_generation_service import ImageGenerationService

router = APIRouter()


@router.post("/profile-image", response_model=ImageGenerationResponse)
async def generate_profile_image(request: CharacterImageRequest) -> ImageGenerationResponse:
    """프로필 이미지 생성 API.

    캐릭터 정보를 기반으로 프로필 이미지를 생성합니다.
    - **description**: 캐릭터 설명 (최대 1000자)
    - **personality**: 캐릭터 성격 (최대 500자)

    Returns:
        - **image_base64**: 생성된 이미지 (base64 인코딩, PNG 포맷)
        - **prompt_used**: 이미지 생성에 사용된 프롬프트
    """
    service = ImageGenerationService()
    try:
        return await service.generate_profile_image(
            description=request.description,
            personality=request.personality,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cover-image", response_model=ImageGenerationResponse)
async def generate_cover_image(request: StoryImageRequest) -> ImageGenerationResponse:
    """스토리 커버 이미지 생성 API.

    스토리 정보를 기반으로 커버 이미지를 생성합니다.
    - **title**: 스토리 제목 (최대 200자)
    - **description**: 스토리 설명 (최대 500자)
    - **summary**: 스토리 요약 (최대 4000자)

    Returns:
        - **image_base64**: 생성된 이미지 (base64 인코딩, PNG 포맷)
        - **prompt_used**: 이미지 생성에 사용된 프롬프트
    """
    service = ImageGenerationService()
    try:
        return await service.generate_cover_image(
            title=request.title,
            description=request.description,
            summary=request.summary,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/background-image", response_model=ImageGenerationResponse)
async def generate_background_image(request: StoryImageRequest) -> ImageGenerationResponse:
    """캐릭터 채팅 배경 이미지 생성 API.

    스토리 정보를 기반으로 채팅 배경 이미지를 생성합니다.
    - **title**: 스토리 제목 (최대 200자)
    - **description**: 스토리 설명 (최대 500자)
    - **summary**: 스토리 요약 (최대 4000자)

    Returns:
        - **image_base64**: 생성된 이미지 (base64 인코딩, PNG 포맷)
        - **prompt_used**: 이미지 생성에 사용된 프롬프트
    """
    service = ImageGenerationService()
    try:
        return await service.generate_background_image(
            title=request.title,
            description=request.description,
            summary=request.summary,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/character-background-image", response_model=ImageGenerationResponse)
async def generate_character_background_image(
    request: CharacterImageRequest,
) -> ImageGenerationResponse:
    """캐릭터별 채팅 배경 이미지 생성 API.

    캐릭터 정보를 기반으로 채팅 배경 이미지를 생성합니다.
    - **description**: 캐릭터 설명 (최대 1000자)
    - **personality**: 캐릭터 성격 (최대 500자)

    Returns:
        - **image_base64**: 생성된 이미지 (base64 인코딩, PNG 포맷)
        - **prompt_used**: 이미지 생성에 사용된 프롬프트
    """
    service = ImageGenerationService()
    try:
        return await service.generate_character_background_image(
            description=request.description,
            personality=request.personality,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
