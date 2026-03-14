import pytest

from src.services.voice_service import VoiceService
from tests.integration.conftest import skip_if_no_elevenlabs, skip_if_no_gemini


@skip_if_no_gemini
class TestVoiceAnalysisIntegration:
    """Gemini 음성 분석 통합 테스트."""

    @pytest.fixture
    def service(self) -> VoiceService:
        """음성 서비스 인스턴스."""
        return VoiceService()

    @pytest.mark.asyncio
    async def test_analyze_character_voice_returns_attributes(self, service: VoiceService):
        """캐릭터 음성 분석이 속성을 반환한다."""
        attributes, settings = await service.analyze_character_voice(
            description="20대 초반의 젊은 여성 마법사",
            personality="지적이고 차분하며 신비로운 분위기",
        )

        assert attributes is not None
        assert attributes.gender is not None
        assert attributes.gender in ["male", "female", "neutral"]
        assert attributes.age is not None
        assert attributes.accent is not None

        assert settings is not None
        assert settings.stability is not None
        assert settings.similarity_boost is not None

    @pytest.mark.asyncio
    async def test_analyze_male_character_voice(self, service: VoiceService):
        """남성 캐릭터 음성 분석."""
        attributes, _ = await service.analyze_character_voice(
            description="40대 중반의 강인한 기사단장",
            personality="용맹하고 정의로우며 카리스마 있는 리더",
        )

        assert attributes is not None
        assert attributes.gender in ["male", "neutral"]

    @pytest.mark.asyncio
    async def test_analyze_female_character_voice(self, service: VoiceService):
        """여성 캐릭터 음성 분석."""
        attributes, _ = await service.analyze_character_voice(
            description="10대 후반의 아름다운 공주",
            personality="순수하고 밝으며 용기 있는 소녀",
        )

        assert attributes is not None
        assert attributes.gender in ["female", "neutral"]


@skip_if_no_elevenlabs
class TestVoiceSearchIntegration:
    """ElevenLabs 음성 검색 통합 테스트."""

    @pytest.fixture
    def service(self) -> VoiceService:
        """음성 서비스 인스턴스."""
        return VoiceService()

    @pytest.mark.asyncio
    async def test_search_matching_voice_returns_voice_id(self, service: VoiceService):
        """음성 검색이 voice_id를 반환한다."""
        from src.models.schemas import VoiceAttributes

        attributes = VoiceAttributes(
            gender="female",
            age="young",
            accent="korean",
            tone=["soft", "calm"],
            keywords=["gentle", "warm"],
        )

        result = await service.search_matching_voice(attributes)

        assert result is not None
        assert "voice_id" in result
        assert len(result["voice_id"]) > 0
        assert "voice_name" in result

    @pytest.mark.asyncio
    async def test_search_male_voice(self, service: VoiceService):
        """남성 음성 검색."""
        from src.models.schemas import VoiceAttributes

        attributes = VoiceAttributes(
            gender="male",
            age="middle_aged",
            accent="american",
            tone=["deep", "authoritative"],
            keywords=["narrator", "strong"],
        )

        result = await service.search_matching_voice(attributes)

        assert result is not None
        assert "voice_id" in result


@skip_if_no_gemini
@skip_if_no_elevenlabs
class TestVoiceRecommendationIntegration:
    """음성 추천 전체 플로우 통합 테스트."""

    @pytest.fixture
    def service(self) -> VoiceService:
        """음성 서비스 인스턴스."""
        return VoiceService()

    @pytest.mark.asyncio
    async def test_recommend_voice_full_flow(self, service: VoiceService):
        """음성 추천 전체 플로우가 동작한다."""
        result = await service.recommend_voice(
            description="20대 초반의 젊은 여성 마법사",
            personality="지적이고 차분하며 신비로운 분위기",
        )

        assert result is not None
        assert result.voice_id is not None
        assert len(result.voice_id) > 0
        assert result.voice_name is not None
        assert result.attributes is not None
        assert result.attributes.gender is not None

    @pytest.mark.asyncio
    async def test_recommend_voice_includes_settings(self, service: VoiceService):
        """음성 추천이 설정값을 포함한다."""
        result = await service.recommend_voice(
            description="40대 중반의 강인한 기사단장",
            personality="용맹하고 정의로운 리더",
        )

        assert result.voice_settings is not None
        assert result.voice_settings.stability is not None
        assert result.voice_settings.similarity_boost is not None
        assert 0 <= result.voice_settings.stability <= 1
        assert 0 <= result.voice_settings.similarity_boost <= 1


@skip_if_no_elevenlabs
class TestTTSGenerationIntegration:
    """TTS 오디오 생성 통합 테스트."""

    @pytest.fixture
    def service(self) -> VoiceService:
        """음성 서비스 인스턴스."""
        return VoiceService()

    @pytest.mark.asyncio
    async def test_generate_sample_audio_returns_bytes(self, service: VoiceService):
        """샘플 오디오 생성이 바이트를 반환한다."""
        from src.models.schemas import VoiceSettings

        # ElevenLabs 기본 음성 사용
        result = await service.generate_sample_audio(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel (기본 음성)
            text="안녕하세요, 저는 테스트 음성입니다.",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                speed=1.0,
            ),
        )

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_sample_audio_without_settings(self, service: VoiceService):
        """설정 없이 샘플 오디오를 생성한다."""
        result = await service.generate_sample_audio(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            text="테스트 음성입니다.",
            voice_settings=None,
        )

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
