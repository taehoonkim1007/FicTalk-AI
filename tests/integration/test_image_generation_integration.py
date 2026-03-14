import base64

import pytest

from src.services.image_generation_service import ImageGenerationService
from tests.integration.conftest import skip_if_no_gemini


@skip_if_no_gemini
class TestImageGenerationIntegration:
    """이미지 생성 API 반환값 검증 테스트."""

    @pytest.fixture
    def service(self) -> ImageGenerationService:
        """이미지 생성 서비스 인스턴스."""
        return ImageGenerationService()

    @pytest.mark.asyncio
    async def test_generate_profile_image_returns_valid_base64(
        self, service: ImageGenerationService
    ):
        """프로필 이미지가 유효한 base64를 반환한다."""
        result = await service.generate_profile_image(
            description="푸른 눈의 젊은 마법사",
            personality="지적이고 차분한 성격",
        )

        assert result is not None
        assert result.image_base64 is not None
        assert len(result.image_base64) > 100
        assert result.prompt_used is not None

        # base64 디코딩 가능 확인
        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_cover_image_returns_valid_base64(self, service: ImageGenerationService):
        """커버 이미지가 유효한 base64를 반환한다."""
        result = await service.generate_cover_image(
            title="마법사의 여정",
            description="어린 마법사가 세계를 구하는 이야기",
            summary="주인공 아리안은 마법 아카데미에서 수련하며 성장한다.",
        )

        assert result is not None
        assert result.image_base64 is not None
        assert len(result.image_base64) > 100
        assert result.prompt_used is not None

        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_background_image_returns_valid_base64(
        self, service: ImageGenerationService
    ):
        """배경 이미지가 유효한 base64를 반환한다."""
        result = await service.generate_background_image(
            title="마법학교",
            description="마법을 배우는 학생들의 이야기",
            summary="고대 마법 도서관에서 벌어지는 모험",
        )

        assert result is not None
        assert result.image_base64 is not None
        assert len(result.image_base64) > 100
        assert result.prompt_used is not None

        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_character_background_image_returns_valid_base64(
        self, service: ImageGenerationService
    ):
        """캐릭터 배경 이미지가 유효한 base64를 반환한다."""
        result = await service.generate_character_background_image(
            description="푸른 로브를 입은 젊은 마법사",
            personality="신비롭고 지적인 분위기",
        )

        assert result is not None
        assert result.image_base64 is not None
        assert len(result.image_base64) > 100
        assert result.prompt_used is not None

        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) > 0
