import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from src.common.constants.error_messages import ERROR_CHARACTER_EXTRACTION, ERROR_SUMMARY_GENERATION
from src.common.constants.gemini import (
    CHARACTER_CHUNK_SIZE,
    CHARACTER_MAX_OUTPUT_TOKENS,
    CHARACTER_TEMPERATURE,
    GEMINI_MODEL,
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_TEMPERATURE,
)
from src.common.constants.operation_names import OP_CHARACTER_EXTRACTION, OP_SUMMARY_GENERATION
from src.common.constants.voice_keywords import CHARACTER_TITLES
from src.config import settings
from src.models.schemas import (
    CharacterGenerationResponse,
    GeneratedCharacter,
    SummaryGenerationResponse,
)
from src.utils.prompts import CHARACTER_GENERATION_PROMPT, SUMMARY_GENERATION_PROMPT
from src.utils.retry import retry_api_call

logger = logging.getLogger(__name__)

# 동시 Gemini API 요청 수 제한 (Rate Limit 방지)
API_CONCURRENCY_LIMIT = 3


class StoryGenerationService:
    """Google Gemini를 사용한 스토리 생성 서비스."""

    MODEL_NAME = GEMINI_MODEL

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.google_api_key)
        self._api_semaphore = asyncio.Semaphore(API_CONCURRENCY_LIMIT)

    async def generate_summary(self, title: str, description: str) -> SummaryGenerationResponse:
        """제목과 한줄요약으로 줄거리 생성."""
        prompt = SUMMARY_GENERATION_PROMPT.format(title=title, description=description)

        async def _call_api() -> SummaryGenerationResponse:
            response = await self._client.aio.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=SUMMARY_TEMPERATURE,
                    max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
                ),
            )
            summary_text = (response.text or "").strip()
            logger.info(f"줄거리 생성 완료: {len(summary_text)}자")
            return SummaryGenerationResponse(summary=summary_text)

        return await retry_api_call(
            _call_api,
            operation_name=OP_SUMMARY_GENERATION,
            error_message=ERROR_SUMMARY_GENERATION,
        )

    async def generate_characters(
        self, title: str, description: str, summary: str
    ) -> CharacterGenerationResponse:
        """청킹 방식으로 캐릭터 생성 (토큰 제한 대응).

        세마포어로 동시 API 요청 수가 제한되어 Rate Limit을 방지합니다.
        """
        try:
            # 1단계: 줄거리 분할
            chunks = self._split_into_chunks(summary, max_chars=CHARACTER_CHUNK_SIZE)

            # 2단계: 각 청크에서 캐릭터 정보 추출 (세마포어로 동시 요청 제한)
            async def extract_with_limit(chunk: str) -> list[GeneratedCharacter]:
                async with self._api_semaphore:
                    return await self._extract_characters_from_chunk(title, description, chunk)

            logger.info(f"캐릭터 추출 시작: {len(chunks)}개 청크 병렬 처리")
            chunk_results = await asyncio.gather(*[extract_with_limit(chunk) for chunk in chunks])

            # 3단계: 결과 병합 + 중복 제거
            merged_characters = self._merge_characters(chunk_results)

            if not merged_characters:
                raise ValueError("생성된 캐릭터가 없습니다")

            logger.info(f"총 {len(merged_characters)}명의 캐릭터 생성 완료")
            return CharacterGenerationResponse(characters=merged_characters)

        except Exception as e:
            logger.error(f"캐릭터 생성 실패: {e}")
            raise RuntimeError(f"캐릭터 생성에 실패했습니다: {e}") from e

    def _split_into_chunks(self, text: str, max_chars: int = 2000) -> list[str]:
        """문장 단위로 텍스트 분할."""
        if len(text) <= max_chars:
            return [text]

        # 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_json_from_response(self, text: str) -> dict:
        """AI 응답에서 JSON 추출 (마크다운 래핑 처리)."""
        text = text.strip()

        # 마크다운 코드 블록 제거 (```json ... ``` 또는 ``` ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            # 첫 줄 제거 (```json 또는 ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 마지막 줄이 ```이면 제거
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)

    async def _extract_characters_from_chunk(
        self, title: str, description: str, chunk: str
    ) -> list[GeneratedCharacter]:
        """단일 청크에서 캐릭터 정보 추출."""
        prompt = CHARACTER_GENERATION_PROMPT.format(
            title=title, description=description, summary=chunk
        )

        async def _call_api() -> list[GeneratedCharacter]:
            response = await self._client.aio.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=CHARACTER_TEMPERATURE,
                    max_output_tokens=CHARACTER_MAX_OUTPUT_TOKENS,
                ),
            )
            response_text = response.text or ""
            try:
                data = self._extract_json_from_response(response_text)
                # {"characters": [...]} 또는 [...] 둘 다 처리
                characters_list = data if isinstance(data, list) else data.get("characters", [])
                return [GeneratedCharacter(**char) for char in characters_list]
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}, 응답: {response_text[:300]}...")
                raise

        try:
            return await retry_api_call(
                _call_api,
                operation_name=OP_CHARACTER_EXTRACTION,
                error_message=ERROR_CHARACTER_EXTRACTION,
            )
        except Exception as e:
            logger.error(f"청크 캐릭터 추출 실패: {e}")
            return []

    def _merge_characters(
        self, chunk_results: list[list[GeneratedCharacter]]
    ) -> list[GeneratedCharacter]:
        """여러 청크의 캐릭터 결과를 병합 (이름 기준 중복 제거)."""
        seen_names: list[str] = []
        merged: list[GeneratedCharacter] = []

        for characters in chunk_results:
            for char in characters:
                normalized = self._normalize_character_name(char.name)
                if not self._is_duplicate_name(normalized, seen_names):
                    seen_names.append(normalized)
                    merged.append(char)

        return merged

    def _normalize_character_name(self, name: str) -> str:
        """캐릭터 이름 정규화 (호칭 제거, 공백 정리)."""
        # 공백 제거 및 소문자화
        normalized = name.strip().lower().replace(" ", "")

        # 호칭/직위 제거
        for title in CHARACTER_TITLES:
            normalized = normalized.replace(title, "")

        return normalized.strip()

    def _is_duplicate_name(self, name: str, existing_names: list[str]) -> bool:
        """중복 이름인지 확인 (부분 일치 포함)."""
        if not name:
            return True  # 빈 이름은 중복으로 처리

        for existing in existing_names:
            # 완전 일치
            if name == existing:
                return True
            # 한쪽이 다른 쪽에 포함 (부분 일치)
            if len(name) >= 2 and len(existing) >= 2 and (name in existing or existing in name):
                return True

        return False
