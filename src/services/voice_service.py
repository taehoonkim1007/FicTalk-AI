import asyncio
import json
import logging
import re
from typing import Any

import httpx
from google import genai
from google.genai import types

from src.common.constants.elevenlabs import (
    DEFAULT_MALE_VOICE_ID,
    DEFAULT_VOICE_ID,
    ELEVENLABS_API_BASE,
    ELEVENLABS_SEARCH_TIMEOUT,
    ELEVENLABS_TTS_TIMEOUT,
    SCORE_ACCENT_MATCH,
    SCORE_AGE_MATCH,
    SCORE_GENDER_MATCH,
    SCORE_GENDER_MISMATCH,
    SCORE_KEYWORD_MATCH,
    SCORE_TONE_MATCH,
    SCORE_USE_CASE_MATCH,
    TTS_MODEL_ID,
    VOICE_SIMILARITY_BOOST_MAX,
    VOICE_SIMILARITY_BOOST_MIN,
    VOICE_SPEED_MAX,
    VOICE_SPEED_MIN,
    VOICE_STABILITY_MAX,
    VOICE_STABILITY_MIN,
    VOICE_STYLE_MAX,
    VOICE_STYLE_MIN,
)
from src.common.constants.gemini import (
    GEMINI_MODEL,
    VOICE_ANALYSIS_MAX_OUTPUT_TOKENS,
    VOICE_ANALYSIS_MAX_RETRIES,
    VOICE_ANALYSIS_RETRY_DELAY,
    VOICE_ANALYSIS_TEMPERATURE,
)
from src.common.constants.voice_keywords import FEMALE_KEYWORDS, MALE_KEYWORDS
from src.config import settings
from src.models.schemas import VoiceAttributes, VoiceRecommendResponse, VoiceSettings
from src.utils.prompts import VOICE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class VoiceService:
    """ElevenLabs Voice 매핑 서비스."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.google_api_key)
        self.elevenlabs_headers = {
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        }

    async def analyze_character_voice(
        self, description: str, personality: str
    ) -> tuple[VoiceAttributes, VoiceSettings]:
        """AI로 캐릭터의 목소리 특성 및 설정 분석."""
        prompt = VOICE_ANALYSIS_PROMPT.format(description=description, personality=personality)

        try:
            data = None
            for attempt in range(VOICE_ANALYSIS_MAX_RETRIES):
                try:
                    response = await self._client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=VOICE_ANALYSIS_TEMPERATURE,
                            max_output_tokens=VOICE_ANALYSIS_MAX_OUTPUT_TOKENS,
                        ),
                    )

                    response_text = response.text
                    if not response_text:
                        logger.warning("Voice 분석: AI 응답이 비어있음, 재시도...")
                        continue

                    response_text = response_text.strip()
                    data = self._parse_json_response(response_text)
                    logger.info(f"Voice 분석 완료: gender={data.get('gender')}")
                    break

                except json.JSONDecodeError as e:
                    logger.warning(f"Voice 분석 JSON 파싱 실패 (시도 {attempt + 1}): {e}")
                    if attempt == VOICE_ANALYSIS_MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(VOICE_ANALYSIS_RETRY_DELAY)

            if data is None:
                raise ValueError("모든 재시도 실패")

            attributes = VoiceAttributes(
                gender=data.get("gender", "neutral"),
                age=data.get("age", "young"),
                accent=data.get("accent", "Korean"),
                tone=data.get("tone", ["calm"]),
                keywords=data.get("keywords", ["narrator"]),
            )

            voice_settings_data = data.get("voice_settings", {})
            voice_settings = VoiceSettings(
                stability=self._clamp(
                    voice_settings_data.get("stability", 0.5),
                    VOICE_STABILITY_MIN,
                    VOICE_STABILITY_MAX,
                ),
                similarity_boost=self._clamp(
                    voice_settings_data.get("similarity_boost", 0.75),
                    VOICE_SIMILARITY_BOOST_MIN,
                    VOICE_SIMILARITY_BOOST_MAX,
                ),
                style=self._clamp(
                    voice_settings_data.get("style", 0.0),
                    VOICE_STYLE_MIN,
                    VOICE_STYLE_MAX,
                ),
                speed=self._clamp(
                    voice_settings_data.get("speed", 1.0),
                    VOICE_SPEED_MIN,
                    VOICE_SPEED_MAX,
                ),
            )

            return attributes, voice_settings

        except Exception as e:
            logger.error(f"Voice 분석 실패: {e}")
            # 폴백: 텍스트에서 직접 성별 추론
            fallback_gender = self._infer_gender_from_text(description)
            logger.info(f"Voice 분석 폴백: gender={fallback_gender}")
            return (
                VoiceAttributes(
                    gender=fallback_gender,
                    age="young",
                    accent="Korean",
                    tone=["calm"],
                    keywords=["narrator"],
                ),
                VoiceSettings(),
            )

    def _parse_json_response(self, text: str) -> dict:
        """AI 응답에서 JSON 파싱 (여러 방법 시도)."""

        # 1. 그대로 파싱 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 마크다운 코드 블록 제거 후 시도
        cleaned = text
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                cleaned = match.group(1)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # 3. trailing comma 제거 후 시도
        cleaned = re.sub(r",\s*}", "}", text)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 4. JSON 객체 부분만 추출
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 모든 시도 실패
        raise json.JSONDecodeError("모든 파싱 방법 실패", text, 0)

    def _infer_gender_from_text(self, text: str) -> str:
        """텍스트에서 성별을 직접 추론 (AI 실패 시 폴백)."""
        text_lower = text.lower()
        for keyword in FEMALE_KEYWORDS:
            if keyword in text_lower:
                return "female"
        for keyword in MALE_KEYWORDS:
            if keyword in text_lower:
                return "male"
        return "neutral"

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """값을 min_val과 max_val 사이로 제한."""
        return max(min_val, min(max_val, value))

    async def search_matching_voice(self, attributes: VoiceAttributes) -> dict[str, Any]:
        """ElevenLabs에서 매칭되는 voice 검색 (사용자 라이브러리)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ELEVENLABS_API_BASE}/voices",
                    headers=self.elevenlabs_headers,
                    timeout=ELEVENLABS_SEARCH_TIMEOUT,
                )
                response.raise_for_status()
                voices_data = response.json()

            voices = voices_data.get("voices", [])

            if not voices:
                logger.warning("Voice 검색: 음성을 찾을 수 없음")
                return self._get_fallback_voice(attributes)

            # 성별 일치하는 음성 필터링
            matching_voices = []
            for voice in voices:
                labels = voice.get("labels", {})
                voice_gender = labels.get("gender", "").lower()
                if voice_gender == attributes.gender:
                    matching_voices.append(voice)

            if not matching_voices:
                logger.warning(f"Voice 검색: 성별({attributes.gender}) 일치 음성 없음")
                return self._get_fallback_voice(attributes)

            # 매칭 점수 계산
            scored_voices = []
            for voice in matching_voices:
                score = self._calculate_voice_score(voice, attributes)
                scored_voices.append((score, voice))

            # 점수 순으로 정렬
            scored_voices.sort(key=lambda x: x[0], reverse=True)

            best_voice = scored_voices[0][1]
            logger.info(f"Voice 검색 완료: {best_voice['name']}")
            return {
                "voice_id": best_voice["voice_id"],
                "voice_name": best_voice["name"],
            }

        except Exception as e:
            logger.error(f"Voice 검색 실패: {e}")
            return self._get_fallback_voice(attributes)

    def _calculate_voice_score(self, voice: dict[str, Any], attributes: VoiceAttributes) -> int:
        """Voice와 속성 간 매칭 점수 계산."""
        score = 0
        labels = voice.get("labels", {})

        # 성별 매칭 (가장 중요)
        voice_gender = labels.get("gender", "").lower()
        if attributes.gender != "neutral":
            if voice_gender == attributes.gender:
                score += SCORE_GENDER_MATCH
            elif voice_gender and voice_gender != attributes.gender:
                score += SCORE_GENDER_MISMATCH
            # 성별 라벨이 없는 경우: 보너스/페널티 없음 (0점)

        # 나이대 매칭
        voice_age = labels.get("age", "").lower()
        age_mapping = {
            "child": ["child", "young"],
            "young": ["young", "middle_aged"],
            "middle": ["middle_aged", "old"],
            "old": ["old", "middle_aged"],
        }
        if attributes.age in age_mapping and voice_age in age_mapping[attributes.age]:
            score += SCORE_AGE_MATCH

        # 억양 매칭
        voice_accent = labels.get("accent", "").lower()
        if attributes.accent.lower() in voice_accent:
            score += SCORE_ACCENT_MATCH

        # use_case 매칭
        voice_use_case = labels.get("use_case", "").lower()
        use_case_keywords = ["narration", "characters", "conversational"]
        for keyword in use_case_keywords:
            if keyword in voice_use_case:
                score += SCORE_USE_CASE_MATCH

        # 키워드 매칭 (voice description에서)
        voice_description = labels.get("description", "").lower()
        for keyword in attributes.keywords:
            if keyword.lower() in voice_description:
                score += SCORE_KEYWORD_MATCH

        # tone 매칭
        for tone in attributes.tone:
            if tone.lower() in voice_description:
                score += SCORE_TONE_MATCH

        return score

    def _get_fallback_voice(self, attributes: VoiceAttributes) -> dict[str, Any]:
        """매칭 실패 시 기본 voice 반환."""
        if attributes.gender == "male":
            return {
                "voice_id": DEFAULT_MALE_VOICE_ID,
                "voice_name": "Adam (Default)",
            }
        return {
            "voice_id": DEFAULT_VOICE_ID,
            "voice_name": "Rachel (Default)",
        }

    async def recommend_voice(self, description: str, personality: str) -> VoiceRecommendResponse:
        """캐릭터에 적합한 voice 추천 (분석 + 검색 통합)."""
        # 1. AI로 캐릭터 특성 및 설정 분석
        attributes, voice_settings = await self.analyze_character_voice(description, personality)
        logger.info(f"캐릭터 분석 결과: {attributes}, 설정: {voice_settings}")

        # 2. ElevenLabs에서 매칭되는 voice 검색
        voice_result = await self.search_matching_voice(attributes)
        logger.info(f"추천 Voice: {voice_result}")

        return VoiceRecommendResponse(
            voice_id=voice_result["voice_id"],
            voice_name=voice_result["voice_name"],
            attributes=attributes,
            voice_settings=voice_settings,
        )

    async def generate_sample_audio(
        self, voice_id: str, text: str, voice_settings: VoiceSettings | None = None
    ) -> bytes:
        """ElevenLabs TTS API로 샘플 음성 생성."""
        settings_to_use = voice_settings or VoiceSettings()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": TTS_MODEL_ID,
                        "voice_settings": {
                            "stability": settings_to_use.stability,
                            "similarity_boost": settings_to_use.similarity_boost,
                            "style": settings_to_use.style,
                            "speed": settings_to_use.speed,
                        },
                    },
                    timeout=ELEVENLABS_TTS_TIMEOUT,
                )
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"샘플 음성 생성 실패: {e}")
            raise RuntimeError(f"샘플 음성 생성에 실패했습니다: {e}") from e

    def _extract_json_from_response(self, text: str) -> dict:
        """AI 응답에서 JSON 추출."""
        text = text.strip()

        # 마크다운 코드 블록 제거
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)
