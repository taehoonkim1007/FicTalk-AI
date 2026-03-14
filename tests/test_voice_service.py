import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.constants.elevenlabs import (
    DEFAULT_MALE_VOICE_ID,
    DEFAULT_VOICE_ID,
    SCORE_ACCENT_MATCH,
    SCORE_AGE_MATCH,
    SCORE_GENDER_MATCH,
    SCORE_GENDER_MISMATCH,
    SCORE_KEYWORD_MATCH,
    SCORE_TONE_MATCH,
    SCORE_USE_CASE_MATCH,
)
from src.models.schemas import VoiceAttributes, VoiceSettings
from src.services.voice_service import VoiceService


@pytest.fixture
def service():
    service = VoiceService.__new__(VoiceService)
    return service


class TestVoiceServiceInit:
    def test_init_creates_client(self):
        """초기화 시 genai client와 headers 설정."""
        with patch("src.services.voice_service.genai.Client") as mock_client:
            service = VoiceService()
            mock_client.assert_called_once()
            assert "xi-api-key" in service.elevenlabs_headers
            assert service.elevenlabs_headers["Content-Type"] == "application/json"


class TestParseJsonResponse:
    def test_plain_json(self, service):
        text = '{"gender": "male"}'
        result = service._parse_json_response(text)
        assert result == {"gender": "male"}

    def test_markdown_json_block(self, service):
        text = '```json\n{"gender": "female"}\n```'
        result = service._parse_json_response(text)
        assert result == {"gender": "female"}

    def test_trailing_comma_removal(self, service):
        text = '{"gender": "male",}'
        result = service._parse_json_response(text)
        assert result == {"gender": "male"}

    def test_trailing_comma_in_array(self, service):
        text = '{"tone": ["calm", "warm",]}'
        result = service._parse_json_response(text)
        assert result == {"tone": ["calm", "warm"]}

    def test_extracts_json_object(self, service):
        text = 'Here is the result: {"gender": "male"} Let me explain...'
        result = service._parse_json_response(text)
        assert result == {"gender": "male"}

    def test_invalid_json_raises(self, service):
        text = "This is not JSON at all"
        with pytest.raises(json.JSONDecodeError):
            service._parse_json_response(text)


class TestInferGenderFromText:
    def test_female_keywords(self, service):
        assert service._infer_gender_from_text("그녀는 아름다운 공주입니다") == "female"
        assert service._infer_gender_from_text("젊은 소녀가 등장합니다") == "female"
        assert service._infer_gender_from_text("마녀가 숲에 살고 있었다") == "female"

    def test_male_keywords(self, service):
        assert service._infer_gender_from_text("용감한 기사가 나타났다") == "male"
        assert service._infer_gender_from_text("젊은 왕자가 여행을 떠났다") == "male"
        assert service._infer_gender_from_text("그는 위대한 황제였다") == "male"

    def test_neutral_when_no_keywords(self, service):
        assert service._infer_gender_from_text("신비로운 존재가 나타났다") == "neutral"

    def test_case_insensitive(self, service):
        assert service._infer_gender_from_text("PRINCESS") == "neutral"  # 영어는 목록에 없음

    def test_word_boundary_no_false_positive(self, service):
        assert service._infer_gender_from_text("공작소에서 일한다") == "neutral"
        assert service._infer_gender_from_text("왕따를 당했다") == "neutral"

    def test_word_boundary_correct_match(self, service):
        assert service._infer_gender_from_text("용감한 기사가 싸웠다") == "male"
        assert service._infer_gender_from_text("위대한 왕이 있었다") == "male"
        assert service._infer_gender_from_text("공작 부인이 왔다") == "female"


class TestClamp:
    def test_value_within_range(self, service):
        assert service._clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_below_min(self, service):
        assert service._clamp(-0.5, 0.0, 1.0) == 0.0

    def test_value_above_max(self, service):
        assert service._clamp(1.5, 0.0, 1.0) == 1.0

    def test_value_at_min(self, service):
        assert service._clamp(0.0, 0.0, 1.0) == 0.0

    def test_value_at_max(self, service):
        assert service._clamp(1.0, 0.0, 1.0) == 1.0


class TestCalculateVoiceScore:
    def test_gender_match_bonus(self, service):
        voice = {"labels": {"gender": "male"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH

    def test_gender_mismatch_penalty(self, service):
        voice = {"labels": {"gender": "female"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score == SCORE_GENDER_MISMATCH

    def test_neutral_gender_no_bonus_penalty(self, service):
        voice = {"labels": {"gender": "female"}}
        attributes = VoiceAttributes(
            gender="neutral", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= 0

    def test_age_match_bonus(self, service):
        voice = {"labels": {"gender": "male", "age": "young"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH + SCORE_AGE_MATCH

    def test_accent_match_bonus(self, service):
        voice = {"labels": {"gender": "male", "accent": "Korean native"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH + SCORE_ACCENT_MATCH


class TestGetFallbackVoice:
    def test_male_fallback(self, service):
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        result = service._get_fallback_voice(attributes)
        assert result["voice_id"] == DEFAULT_MALE_VOICE_ID
        assert "Adam" in result["voice_name"]

    def test_female_fallback(self, service):
        attributes = VoiceAttributes(
            gender="female", age="young", accent="Korean", tone=[], keywords=[]
        )
        result = service._get_fallback_voice(attributes)
        assert result["voice_id"] == DEFAULT_VOICE_ID
        assert "Rachel" in result["voice_name"]

    def test_neutral_fallback(self, service):
        attributes = VoiceAttributes(
            gender="neutral", age="young", accent="Korean", tone=[], keywords=[]
        )
        result = service._get_fallback_voice(attributes)
        assert result["voice_id"] == DEFAULT_VOICE_ID


class TestCalculateVoiceScoreExtended:
    def test_use_case_match_bonus(self, service):
        """use_case 매칭 보너스."""
        voice = {"labels": {"gender": "male", "use_case": "narration"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH + SCORE_USE_CASE_MATCH

    def test_keyword_match_bonus(self, service):
        """키워드 매칭 보너스."""
        voice = {"labels": {"gender": "male", "description": "calm narrator voice"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=[], keywords=["calm"]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH + SCORE_KEYWORD_MATCH

    def test_tone_match_bonus(self, service):
        """tone 매칭 보너스."""
        voice = {"labels": {"gender": "male", "description": "warm friendly voice"}}
        attributes = VoiceAttributes(
            gender="male", age="young", accent="Korean", tone=["warm"], keywords=[]
        )
        score = service._calculate_voice_score(voice, attributes)
        assert score >= SCORE_GENDER_MATCH + SCORE_TONE_MATCH


class TestAnalyzeCharacterVoice:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        """AI 분석 성공."""
        with patch("src.services.voice_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(
                {
                    "gender": "female",
                    "age": "young",
                    "accent": "Korean",
                    "tone": ["calm", "gentle"],
                    "keywords": ["princess"],
                    "voice_settings": {
                        "stability": 0.6,
                        "similarity_boost": 0.8,
                        "style": 0.2,
                        "speed": 1.0,
                    },
                }
            )
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            attributes, settings = await service.analyze_character_voice(
                description="아름다운 공주", personality="차분하고 온화함"
            )

            assert attributes.gender == "female"
            assert attributes.age == "young"
            assert settings.stability == 0.6

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """AI 실패 시 폴백."""
        with patch("src.services.voice_service.genai.Client") as mock_client:
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                side_effect=Exception("API Error")
            )

            service = VoiceService()
            attributes, settings = await service.analyze_character_voice(
                description="용감한 기사", personality="용맹함"
            )

            assert attributes.gender == "male"  # 텍스트에서 추론
            assert isinstance(settings, VoiceSettings)

    @pytest.mark.asyncio
    async def test_empty_response_retry(self):
        """빈 응답 시 재시도."""
        with patch("src.services.voice_service.genai.Client") as mock_client:
            mock_empty_response = MagicMock()
            mock_empty_response.text = ""

            mock_valid_response = MagicMock()
            mock_valid_response.text = json.dumps({"gender": "male"})

            mock_client.return_value.aio.models.generate_content = AsyncMock(
                side_effect=[mock_empty_response, mock_valid_response]
            )

            service = VoiceService()
            attributes, _ = await service.analyze_character_voice(
                description="테스트", personality="테스트"
            )

            assert attributes.gender == "male"


class TestSearchMatchingVoice:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        """음성 검색 성공."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "voices": [
                    {
                        "voice_id": "voice1",
                        "name": "Test Voice",
                        "labels": {"gender": "female", "age": "young"},
                    }
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            attributes = VoiceAttributes(
                gender="female", age="young", accent="Korean", tone=[], keywords=[]
            )
            result = await service.search_matching_voice(attributes)

            assert result["voice_id"] == "voice1"
            assert result["voice_name"] == "Test Voice"

    @pytest.mark.asyncio
    async def test_no_voices_fallback(self):
        """음성 없을 때 폴백."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_response = MagicMock()
            mock_response.json.return_value = {"voices": []}
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            attributes = VoiceAttributes(
                gender="male", age="young", accent="Korean", tone=[], keywords=[]
            )
            result = await service.search_matching_voice(attributes)

            assert result["voice_id"] == DEFAULT_MALE_VOICE_ID

    @pytest.mark.asyncio
    async def test_no_gender_match_fallback(self):
        """성별 일치 없을 때 폴백."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "voices": [
                    {
                        "voice_id": "voice1",
                        "name": "Female Voice",
                        "labels": {"gender": "female"},
                    }
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            attributes = VoiceAttributes(
                gender="male", age="young", accent="Korean", tone=[], keywords=[]
            )
            result = await service.search_matching_voice(attributes)

            assert result["voice_id"] == DEFAULT_MALE_VOICE_ID

    @pytest.mark.asyncio
    async def test_api_error_fallback(self):
        """API 오류 시 폴백."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("API Error")
            )

            service = VoiceService()
            attributes = VoiceAttributes(
                gender="female", age="young", accent="Korean", tone=[], keywords=[]
            )
            result = await service.search_matching_voice(attributes)

            assert result["voice_id"] == DEFAULT_VOICE_ID


class TestRecommendVoice:
    @pytest.mark.asyncio
    async def test_recommend_voice_integration(self):
        """음성 추천 통합 테스트."""
        with patch("src.services.voice_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(
                {
                    "gender": "female",
                    "age": "young",
                    "accent": "Korean",
                    "tone": ["gentle"],
                    "keywords": ["princess"],
                    "voice_settings": {"stability": 0.5},
                }
            )
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            with patch("httpx.AsyncClient") as mock_httpx:
                mock_voice_response = MagicMock()
                mock_voice_response.json.return_value = {
                    "voices": [
                        {
                            "voice_id": "voice123",
                            "name": "Princess Voice",
                            "labels": {"gender": "female", "age": "young"},
                        }
                    ]
                }
                mock_voice_response.raise_for_status = MagicMock()
                mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_voice_response
                )

                service = VoiceService()
                result = await service.recommend_voice(
                    description="아름다운 공주", personality="온화함"
                )

                assert result.voice_id == "voice123"
                assert result.attributes.gender == "female"


class TestGenerateSampleAudio:
    @pytest.mark.asyncio
    async def test_successful_audio_generation(self):
        """샘플 음성 생성 성공."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_response = MagicMock()
            mock_response.content = b"audio_bytes"
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            result = await service.generate_sample_audio(
                voice_id="voice123",
                text="테스트 음성",
                voice_settings=VoiceSettings(stability=0.5),
            )

            assert result == b"audio_bytes"

    @pytest.mark.asyncio
    async def test_audio_generation_without_settings(self):
        """기본 설정으로 샘플 음성 생성."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_response = MagicMock()
            mock_response.content = b"audio_data"
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            service = VoiceService()
            result = await service.generate_sample_audio(
                voice_id="voice123",
                text="테스트",
            )

            assert result == b"audio_data"

    @pytest.mark.asyncio
    async def test_audio_generation_failure(self):
        """샘플 음성 생성 실패 시 RuntimeError."""
        with (
            patch("src.services.voice_service.genai.Client"),
            patch("httpx.AsyncClient") as mock_httpx,
        ):
            mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("TTS Error")
            )

            service = VoiceService()

            with pytest.raises(RuntimeError) as exc_info:
                await service.generate_sample_audio(
                    voice_id="voice123",
                    text="테스트",
                )

            assert "샘플 음성 생성에 실패했습니다" in str(exc_info.value)


class TestExtractJsonFromResponse:
    def test_plain_json(self, service):
        """일반 JSON 파싱."""
        text = '{"key": "value"}'
        result = service._extract_json_from_response(text)
        assert result == {"key": "value"}

    def test_markdown_code_block(self, service):
        """마크다운 코드 블록 제거."""
        text = '```\n{"key": "value"}\n```'
        result = service._extract_json_from_response(text)
        assert result == {"key": "value"}

    def test_markdown_with_json_label(self, service):
        """json 라벨이 있는 마크다운 블록."""
        text = '```json\n{"key": "value"}\n```'
        result = service._extract_json_from_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self, service):
        """잘못된 JSON은 예외 발생."""
        text = "not a json"
        with pytest.raises(json.JSONDecodeError):
            service._extract_json_from_response(text)
