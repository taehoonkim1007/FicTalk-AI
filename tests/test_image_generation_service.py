from unittest.mock import MagicMock, patch

import pytest

from src.services.image_generation_service import ImageGenerationService


class TestImageGenerationServiceInit:
    def test_client_initialization(self):
        """클라이언트 초기화 확인."""
        with patch("src.services.image_generation_service.genai.Client") as mock_client:
            service = ImageGenerationService()
            mock_client.assert_called_once()
            assert service.client is not None


class TestExtractImageFromResponse:
    @pytest.fixture
    def service(self):
        with patch("src.services.image_generation_service.genai.Client"):
            return ImageGenerationService()

    def test_extract_bytes_image(self, service):
        """bytes 이미지 데이터 추출 및 base64 인코딩."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"test_image_bytes"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        result = service._extract_image_from_response(mock_response)

        assert result == "dGVzdF9pbWFnZV9ieXRlcw=="  # base64 of "test_image_bytes"

    def test_extract_string_image(self, service):
        """이미 base64 인코딩된 문자열 이미지 데이터."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = "already_base64_encoded"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        result = service._extract_image_from_response(mock_response)

        assert result == "already_base64_encoded"

    def test_no_candidates_raises_error(self, service):
        """candidates가 없으면 ValueError 발생."""
        mock_response = MagicMock()
        mock_response.candidates = []

        with pytest.raises(ValueError) as exc_info:
            service._extract_image_from_response(mock_response)

        assert "No candidates in response" in str(exc_info.value)

    def test_no_image_data_raises_error(self, service):
        """이미지 데이터가 없으면 ValueError 발생."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = None
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        with pytest.raises(ValueError) as exc_info:
            service._extract_image_from_response(mock_response)

        assert "No image data in response" in str(exc_info.value)

    def test_empty_parts_raises_error(self, service):
        """parts가 비어있으면 ValueError 발생."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[]))]

        with pytest.raises(ValueError) as exc_info:
            service._extract_image_from_response(mock_response)

        assert "No image data in response" in str(exc_info.value)


class TestGenerateProfileImage:
    @pytest.fixture
    def service(self):
        with patch("src.services.image_generation_service.genai.Client"):
            return ImageGenerationService()

    @pytest.mark.asyncio
    async def test_generate_profile_image_success(self, service):
        """프로필 이미지 생성 성공."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"profile_image_bytes"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        with patch.object(service.client.models, "generate_content", return_value=mock_response):
            result = await service.generate_profile_image(
                description="파란 눈의 소녀",
                personality="밝고 활발한 성격",
            )

        assert result.image_base64 is not None
        assert "파란 눈의 소녀" in result.prompt_used


class TestGenerateCoverImage:
    @pytest.fixture
    def service(self):
        with patch("src.services.image_generation_service.genai.Client"):
            return ImageGenerationService()

    @pytest.mark.asyncio
    async def test_generate_cover_image_success(self, service):
        """커버 이미지 생성 성공."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"cover_image_bytes"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        with patch.object(service.client.models, "generate_content", return_value=mock_response):
            result = await service.generate_cover_image(
                title="마법의 숲",
                description="마법사와 요정의 이야기",
                summary="한 소년이 마법의 숲에서 모험을 시작한다",
            )

        assert result.image_base64 is not None
        assert result.prompt_used is not None


class TestGenerateBackgroundImage:
    @pytest.fixture
    def service(self):
        with patch("src.services.image_generation_service.genai.Client"):
            return ImageGenerationService()

    @pytest.mark.asyncio
    async def test_generate_background_image_success(self, service):
        """배경 이미지 생성 성공."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"background_image_bytes"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        with patch.object(service.client.models, "generate_content", return_value=mock_response):
            result = await service.generate_background_image(
                title="달빛 정원",
                description="신비로운 정원 이야기",
                summary="보름달이 뜬 밤의 정원 풍경",
            )

        assert result.image_base64 is not None
        assert result.prompt_used is not None


class TestGenerateCharacterBackgroundImage:
    @pytest.fixture
    def service(self):
        with patch("src.services.image_generation_service.genai.Client"):
            return ImageGenerationService()

    @pytest.mark.asyncio
    async def test_generate_character_background_image_success(self, service):
        """캐릭터 배경 이미지 생성 성공."""
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"character_bg_bytes"
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        with patch.object(service.client.models, "generate_content", return_value=mock_response):
            result = await service.generate_character_background_image(
                description="검은 머리 마법사",
                personality="차분하고 지적인 성격",
            )

        assert result.image_base64 is not None
        assert result.prompt_used is not None
