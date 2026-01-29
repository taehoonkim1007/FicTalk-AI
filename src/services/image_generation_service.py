import asyncio
import base64
import logging

from google import genai
from google.genai import types

from src.config import settings
from src.models.schemas import ProfileImageResponse

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """Google Gemini 2.5 Flash Image를 사용한 이미지 생성 서비스."""

    MODEL_NAME = "gemini-2.5-flash-image"

    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.google_api_key)

    async def generate_profile_image(
        self,
        name: str,
        role: str,
        description: str,
        personality: str,
    ) -> ProfileImageResponse:
        """캐릭터 프로필 이미지 생성.

        Args:
            name: 캐릭터 이름
            role: 역할 (주인공 또는 조연)
            description: 캐릭터 설명
            personality: 캐릭터 성격

        Returns:
            ProfileImageResponse: base64 인코딩된 이미지와 사용된 프롬프트
        """
        prompt = self._build_profile_image_prompt(name, role, description, personality)
        logger.info(f"프로필 이미지 생성 시작: {name}")

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

            logger.info(f"프로필 이미지 생성 완료: {name}, base64 길이: {len(image_base64)}")
            return ProfileImageResponse(
                image_base64=image_base64,
                prompt_used=prompt,
            )

        except Exception as e:
            import traceback

            logger.error(f"프로필 이미지 생성 실패: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"프로필 이미지 생성에 실패했습니다: {e}") from e

    def _build_profile_image_prompt(
        self,
        name: str,
        role: str,
        description: str,
        personality: str,
    ) -> str:
        """프로필 이미지 생성용 프롬프트 구성."""
        prompt_parts = [
            "Character portrait illustration for a fiction story:",
            "",
            f"Character Name: {name}",
            f"Role: {role}",
            f"Description: {description}",
            f"Personality: {personality}",
            "",
            "Art Style Requirements:",
            "- Portrait format (head and shoulders)",
            "- Clean, simple gradient background",
            "- High quality digital illustration",
            "- Semi-realistic anime art style",
            "- Expressive facial features matching the personality",
            "- Soft lighting with subtle shadows",
            "- Suitable for a fiction character profile card",
        ]

        return "\n".join(prompt_parts)
