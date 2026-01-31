import asyncio
import base64
import logging

from google import genai
from google.genai import types

from src.config import settings
from src.models.schemas import ImageGenerationResponse
from src.utils.prompts import (
    BACKGROUND_IMAGE_PROMPT,
    CHARACTER_BACKGROUND_IMAGE_PROMPT,
    COVER_IMAGE_PROMPT,
    PROFILE_IMAGE_PROMPT,
)

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Google Gemini 2.5 Flash Image를 사용한 이미지 생성 서비스."""

    MODEL_NAME = "gemini-2.5-flash-image"

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

        try:
            # 동기 메서드를 람다로 감싸서 별도 스레드에서 실행
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
            )

            # 디버그 로깅
            logger.info(f"응답 타입: {type(response)}")
            logger.info(f"candidates 수: {len(response.candidates) if response.candidates else 0}")

            if not response.candidates:
                raise ValueError("응답에 candidates가 없습니다")

            candidate = response.candidates[0]
            logger.info(f"candidate 타입: {type(candidate)}")
            logger.info(f"content 타입: {type(candidate.content)}")
            parts_count = 0
            if candidate.content and candidate.content.parts:
                parts_count = len(candidate.content.parts)
            logger.info(f"parts 수: {parts_count}")

            # 응답에서 이미지 추출
            image_data = None
            if candidate.content and candidate.content.parts:
                for i, part in enumerate(candidate.content.parts):
                    logger.info(
                        f"part[{i}] 타입: {type(part)}, inline_data: {part.inline_data is not None}"
                    )
                    if part.inline_data is not None:
                        image_data = part.inline_data.data
                        logger.info(
                            f"이미지 데이터 타입: {type(image_data)}, 크기: {len(image_data) if image_data else 0}"
                        )
                        break

            if image_data is None:
                raise ValueError("이미지 생성 결과가 없습니다")

            # 이미지 데이터가 이미 bytes인 경우 직접 인코딩
            if isinstance(image_data, bytes):
                image_base64 = base64.b64encode(image_data).decode("utf-8")
            else:
                # 이미 base64 문자열인 경우
                image_base64 = image_data

            logger.info(f"프로필 이미지 생성 완료, base64 길이: {len(image_base64)}")
            return ImageGenerationResponse(
                image_base64=image_base64,
                prompt_used=prompt,
            )

        except Exception as e:
            import traceback

            logger.error(f"프로필 이미지 생성 실패: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"프로필 이미지 생성에 실패했습니다: {e}") from e

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
        summary_excerpt = summary[:1000] if len(summary) > 1000 else summary
        prompt = COVER_IMAGE_PROMPT.format(
            title=title, description=description, summary=summary_excerpt
        )
        logger.info(f"커버 이미지 생성 시작: {title}")

        try:
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
            logger.info(f"커버 이미지 생성 완료: {title}, base64 길이: {len(image_base64)}")

            return ImageGenerationResponse(
                image_base64=image_base64,
                prompt_used=prompt,
            )

        except Exception as e:
            import traceback

            logger.error(f"커버 이미지 생성 실패: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"커버 이미지 생성에 실패했습니다: {e}") from e

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
        summary_excerpt = summary[:1000] if len(summary) > 1000 else summary
        prompt = BACKGROUND_IMAGE_PROMPT.format(
            title=title, description=description, summary=summary_excerpt
        )
        logger.info(f"배경 이미지 생성 시작: {title}")

        try:
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
            logger.info(f"배경 이미지 생성 완료: {title}, base64 길이: {len(image_base64)}")

            return ImageGenerationResponse(
                image_base64=image_base64,
                prompt_used=prompt,
            )

        except Exception as e:
            import traceback

            logger.error(f"배경 이미지 생성 실패: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"배경 이미지 생성에 실패했습니다: {e}") from e

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

        try:
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
            logger.info(f"캐릭터 배경 이미지 생성 완료, base64 길이: {len(image_base64)}")

            return ImageGenerationResponse(
                image_base64=image_base64,
                prompt_used=prompt,
            )

        except Exception as e:
            import traceback

            logger.error(f"캐릭터 배경 이미지 생성 실패: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"캐릭터 배경 이미지 생성에 실패했습니다: {e}") from e

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
