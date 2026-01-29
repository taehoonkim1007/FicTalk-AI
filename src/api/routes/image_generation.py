from fastapi import APIRouter, HTTPException

from src.models.schemas import ProfileImageRequest, ProfileImageResponse
from src.services.image_generation_service import ImageGenerationService

router = APIRouter()


@router.post("/profile-image", response_model=ProfileImageResponse)
async def generate_profile_image(request: ProfileImageRequest) -> ProfileImageResponse:
    """프로필 이미지 생성 API.

    캐릭터 정보를 기반으로 프로필 이미지를 생성합니다.
    - **name**: 캐릭터 이름 (최대 100자)
    - **role**: 역할 - 주인공 또는 조연 (최대 50자)
    - **description**: 캐릭터 설명 (최대 1000자)
    - **personality**: 캐릭터 성격 (선택, 최대 500자)

    Returns:
        - **image_base64**: 생성된 이미지 (base64 인코딩, PNG 포맷)
        - **prompt_used**: 이미지 생성에 사용된 프롬프트
    """
    service = ImageGenerationService()
    try:
        return await service.generate_profile_image(
            name=request.name,
            role=request.role,
            description=request.description,
            personality=request.personality,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
