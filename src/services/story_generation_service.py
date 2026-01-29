import asyncio
import json
import logging
import re

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.config import settings
from src.models.schemas import (
    CharacterGenerationResponse,
    GeneratedCharacter,
    SummaryGenerationResponse,
)
from src.utils.prompts import CHARACTER_GENERATION_PROMPT, SUMMARY_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class StoryGenerationService:
    """Google Gemini를 사용한 스토리 생성 서비스."""

    MODEL_NAME = "gemini-2.0-flash-lite"

    def __init__(self) -> None:
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=GenerationConfig(
                temperature=0.8,
                max_output_tokens=4096,
            ),
        )

    async def generate_summary(self, title: str, description: str) -> SummaryGenerationResponse:
        """제목과 한줄요약으로 줄거리 생성."""
        prompt = SUMMARY_GENERATION_PROMPT.format(title=title, description=description)

        try:
            response = await self.model.generate_content_async(prompt)
            return SummaryGenerationResponse(summary=response.text.strip())
        except Exception as e:
            logger.error(f"줄거리 생성 실패: {e}")
            raise RuntimeError(f"줄거리 생성에 실패했습니다: {e}") from e

    async def generate_characters(
        self, title: str, description: str, summary: str
    ) -> CharacterGenerationResponse:
        """청킹 방식으로 캐릭터 생성 (토큰 제한 대응)."""
        try:
            # 1단계: 줄거리 분할 (2000자 기준)
            chunks = self._split_into_chunks(summary, max_chars=2000)
            logger.info(f"줄거리 {len(summary)}자를 {len(chunks)}개 청크로 분할")

            # 2단계: 각 청크에서 캐릭터 정보 추출 (병렬)
            extraction_tasks = [
                self._extract_characters_from_chunk(title, description, chunk) for chunk in chunks
            ]
            chunk_results = await asyncio.gather(*extraction_tasks)

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

        response_text = ""
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            response_text = response.text
            data = self._extract_json_from_response(response_text)
            return [GeneratedCharacter(**char) for char in data.get("characters", [])]
        except json.JSONDecodeError as e:
            logger.error(f"청크 JSON 파싱 실패: {e}, 응답: {response_text[:500]}")
            return []
        except Exception as e:
            logger.error(f"청크 캐릭터 추출 실패: {e}")
            return []

    def _merge_characters(
        self, chunk_results: list[list[GeneratedCharacter]]
    ) -> list[GeneratedCharacter]:
        """여러 청크의 캐릭터 결과를 병합 (이름 기준 중복 제거)."""
        seen_names: set[str] = set()
        merged: list[GeneratedCharacter] = []

        for characters in chunk_results:
            for char in characters:
                normalized_name = char.name.strip().lower()
                if normalized_name not in seen_names:
                    seen_names.add(normalized_name)
                    merged.append(char)

        return merged
