import json
import logging
from typing import Any

import google.generativeai as genai
import httpx
from google.generativeai.types import GenerationConfig

from src.config import settings
from src.models.schemas import VoiceAttributes, VoiceRecommendResponse, VoiceSettings
from src.utils.prompts import VOICE_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class VoiceService:
    """ElevenLabs Voice 매핑 서비스."""

    MODEL_NAME = "gemini-2.0-flash-lite"
    ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

    # 기본 폴백 Voice ID (매칭 실패 시)
    DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - 기본 여성 음성
    DEFAULT_MALE_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam - 기본 남성 음성

    def __init__(self) -> None:
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=GenerationConfig(
                temperature=0.3,  # 일관성을 위해 낮은 temperature
                max_output_tokens=512,
            ),
        )
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
            response = await self.model.generate_content_async(prompt)
            data = self._extract_json_from_response(response.text)

            attributes = VoiceAttributes(
                gender=data.get("gender", "neutral"),
                age=data.get("age", "young"),
                accent=data.get("accent", "Korean"),
                tone=data.get("tone", ["calm"]),
                keywords=data.get("keywords", ["narrator"]),
            )

            voice_settings_data = data.get("voice_settings", {})
            voice_settings = VoiceSettings(
                stability=self._clamp(voice_settings_data.get("stability", 0.5), 0.0, 1.0),
                similarity_boost=self._clamp(
                    voice_settings_data.get("similarity_boost", 0.75), 0.0, 1.0
                ),
                style=self._clamp(voice_settings_data.get("style", 0.0), 0.0, 1.0),
                speed=self._clamp(voice_settings_data.get("speed", 1.0), 0.7, 1.2),
            )

            return attributes, voice_settings
        except Exception as e:
            logger.error(f"Voice 특성 분석 실패: {e}")
            return (
                VoiceAttributes(
                    gender="neutral",
                    age="young",
                    accent="Korean",
                    tone=["calm"],
                    keywords=["narrator"],
                ),
                VoiceSettings(),
            )

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """값을 min_val과 max_val 사이로 제한."""
        return max(min_val, min(max_val, value))

    async def search_matching_voice(self, attributes: VoiceAttributes) -> dict[str, Any]:
        """ElevenLabs Voice Library에서 매칭되는 voice 검색."""
        try:
            async with httpx.AsyncClient() as client:
                # Voice Library 검색
                response = await client.get(
                    f"{self.ELEVENLABS_API_BASE}/voices",
                    headers=self.elevenlabs_headers,
                    params={"show_legacy": "false"},
                    timeout=30.0,
                )
                response.raise_for_status()
                voices_data = response.json()

            voices = voices_data.get("voices", [])
            if not voices:
                logger.warning("ElevenLabs에서 voice를 찾을 수 없음")
                return self._get_fallback_voice(attributes)

            # 매칭 점수 계산
            scored_voices = []
            for voice in voices:
                score = self._calculate_voice_score(voice, attributes)
                scored_voices.append((score, voice))

            # 점수 순으로 정렬
            scored_voices.sort(key=lambda x: x[0], reverse=True)

            if scored_voices and scored_voices[0][0] > 0:
                best_voice = scored_voices[0][1]
                return {
                    "voice_id": best_voice["voice_id"],
                    "voice_name": best_voice["name"],
                }

            return self._get_fallback_voice(attributes)

        except Exception as e:
            logger.error(f"ElevenLabs Voice 검색 실패: {e}")
            return self._get_fallback_voice(attributes)

    def _calculate_voice_score(self, voice: dict[str, Any], attributes: VoiceAttributes) -> int:
        """Voice와 속성 간 매칭 점수 계산."""
        score = 0
        labels = voice.get("labels", {})

        # 성별 매칭 (가장 중요)
        voice_gender = labels.get("gender", "").lower()
        if attributes.gender != "neutral":
            if voice_gender == attributes.gender:
                score += 30
            elif voice_gender:
                score -= 20  # 성별 불일치 페널티

        # 나이대 매칭
        voice_age = labels.get("age", "").lower()
        age_mapping = {
            "child": ["child", "young"],
            "young": ["young", "middle_aged"],
            "middle": ["middle_aged", "old"],
            "old": ["old", "middle_aged"],
        }
        if attributes.age in age_mapping and voice_age in age_mapping[attributes.age]:
            score += 15

        # 억양 매칭
        voice_accent = labels.get("accent", "").lower()
        if attributes.accent.lower() in voice_accent:
            score += 10

        # use_case 매칭
        voice_use_case = labels.get("use_case", "").lower()
        use_case_keywords = ["narration", "characters", "conversational"]
        for keyword in use_case_keywords:
            if keyword in voice_use_case:
                score += 5

        # 키워드 매칭 (voice description에서)
        voice_description = labels.get("description", "").lower()
        for keyword in attributes.keywords:
            if keyword.lower() in voice_description:
                score += 3

        # tone 매칭
        for tone in attributes.tone:
            if tone.lower() in voice_description:
                score += 2

        return score

    def _get_fallback_voice(self, attributes: VoiceAttributes) -> dict[str, Any]:
        """매칭 실패 시 기본 voice 반환."""
        if attributes.gender == "male":
            return {
                "voice_id": self.DEFAULT_MALE_VOICE_ID,
                "voice_name": "Adam (Default)",
            }
        return {
            "voice_id": self.DEFAULT_VOICE_ID,
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
                    f"{self.ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": settings_to_use.stability,
                            "similarity_boost": settings_to_use.similarity_boost,
                            "style": settings_to_use.style,
                            "speed": settings_to_use.speed,
                        },
                    },
                    timeout=60.0,
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
