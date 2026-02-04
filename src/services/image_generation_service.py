import asyncio
import base64
import logging

from google import genai
from google.genai import types

from src.common.constants.error_messages import (
    ERROR_BACKGROUND_IMAGE,
    ERROR_CHARACTER_BACKGROUND_IMAGE,
    ERROR_COVER_IMAGE,
    ERROR_PROFILE_IMAGE,
)
from src.common.constants.gemini import GEMINI_IMAGE_MODEL
from src.common.constants.operation_names import (
    OP_BACKGROUND_IMAGE,
    OP_CHARACTER_BACKGROUND_IMAGE,
    OP_COVER_IMAGE,
    OP_PROFILE_IMAGE,
)
from src.config import settings
from src.models.schemas import ImageGenerationResponse
from src.utils.prompts import (
    BACKGROUND_IMAGE_PROMPT,
    CHARACTER_BACKGROUND_IMAGE_PROMPT,
    COVER_IMAGE_PROMPT,
    PROFILE_IMAGE_PROMPT,
)
from src.utils.retry import retry_api_call_tuple

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Google Gemini 2.5 Flash Image를 사용한 이미지 생성 서비스.

    Rate limit 발생 시 지수 백오프로 재시도합니다.
    """

    MODEL_NAME = GEMINI_IMAGE_MODEL

    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.google_api_key)

    async def generate_profile_image(
        self,
        description: str,
        personality: str,
    ) -> ImageGenerationResponse:
        """캐릭터 프로필 이미지 생성.

        Args:
            description: 캐릭터 설명
            personality: 캐릭터 성격

        Returns:
            ImageGenerationResponse: base64 인코딩된 이미지와 사용된 프롬프트
        """
        prompt = PROFILE_IMAGE_PROMPT.format(description=description, personality=personality)
        logger.info("프로필 이미지 생성 시작")

        async def _call_api() -> tuple[str, str]:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            )
            image_base64 = self._extract_image_from_response(response)
            return (image_base64, prompt)

        image_base64, prompt_used = await retry_api_call_tuple(
            _call_api,
            operation_name=OP_PROFILE_IMAGE,
            error_message=ERROR_PROFILE_IMAGE,
        )

        logger.info(f"프로필 이미지 생성 완료, base64 길이: {len(image_base64)}")
        return ImageGenerationResponse(image_base64=image_base64, prompt_used=prompt_used)

    async def generate_cover_image(
        self,
        title: str,
        description: str,
        summary: str,
    ) -> ImageGenerationResponse:
        """스토리 커버 이미지 생성.

        Args:
            title: 스토리 제목
            description: 스토리 설명
            summary: 스토리 요약

        Returns:
            ImageGenerationResponse: base64 인코딩된 이미지와 사용된 프롬프트
        """
        prompt = COVER_IMAGE_PROMPT.format(title=title, description=description, summary=summary)
        logger.info(f"커버 이미지 생성 시작: {title}")

        async def _call_api() -> tuple[str, str]:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            )
            image_base64 = self._extract_image_from_response(response)
            return (image_base64, prompt)

        image_base64, prompt_used = await retry_api_call_tuple(
            _call_api,
            operation_name=OP_COVER_IMAGE,
            error_message=ERROR_COVER_IMAGE,
        )

        logger.info(f"커버 이미지 생성 완료: {title}, base64 길이: {len(image_base64)}")
        return ImageGenerationResponse(image_base64=image_base64, prompt_used=prompt_used)

    async def generate_background_image(
        self,
        title: str,
        description: str,
        summary: str,
    ) -> ImageGenerationResponse:
        """스토리 채팅 배경 이미지 생성.

        Args:
            title: 스토리 제목
            description: 스토리 설명
            summary: 스토리 요약

        Returns:
            ImageGenerationResponse: base64 인코딩된 이미지와 사용된 프롬프트
        """
        prompt = BACKGROUND_IMAGE_PROMPT.format(
            title=title, description=description, summary=summary
        )
        logger.info(f"배경 이미지 생성 시작: {title}")

        async def _call_api() -> tuple[str, str]:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            )
            image_base64 = self._extract_image_from_response(response)
            return (image_base64, prompt)

        image_base64, prompt_used = await retry_api_call_tuple(
            _call_api,
            operation_name=OP_BACKGROUND_IMAGE,
            error_message=ERROR_BACKGROUND_IMAGE,
        )

        logger.info(f"배경 이미지 생성 완료: {title}, base64 길이: {len(image_base64)}")
        return ImageGenerationResponse(image_base64=image_base64, prompt_used=prompt_used)

    async def generate_character_background_image(
        self,
        description: str,
        personality: str,
    ) -> ImageGenerationResponse:
        """캐릭터별 채팅 배경 이미지 생성.

        Args:
            description: 캐릭터 설명
            personality: 캐릭터 성격

        Returns:
            ImageGenerationResponse: base64 인코딩된 이미지와 사용된 프롬프트
        """
        prompt = CHARACTER_BACKGROUND_IMAGE_PROMPT.format(
            description=description, personality=personality
        )
        logger.info("캐릭터 배경 이미지 생성 시작")

        async def _call_api() -> tuple[str, str]:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            )
            image_base64 = self._extract_image_from_response(response)
            return (image_base64, prompt)

        image_base64, prompt_used = await retry_api_call_tuple(
            _call_api,
            operation_name=OP_CHARACTER_BACKGROUND_IMAGE,
            error_message=ERROR_CHARACTER_BACKGROUND_IMAGE,
        )

        logger.info(f"캐릭터 배경 이미지 생성 완료, base64 길이: {len(image_base64)}")
        return ImageGenerationResponse(image_base64=image_base64, prompt_used=prompt_used)

    def _extract_image_from_response(self, response) -> str:
        """Gemini API 응답에서 이미지 데이터 추출."""
        if not response.candidates:
            raise ValueError("응답에 candidates가 없습니다")

        candidate = response.candidates[0]
        image_data = None

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data
                    break

        if image_data is None:
            raise ValueError("이미지 생성 결과가 없습니다")

        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode("utf-8")
        return image_data
