import pytest

from src.services.story_generation_service import StoryGenerationService
from tests.integration.conftest import skip_if_no_gemini


@skip_if_no_gemini
class TestSummaryGeneration:
    """줄거리 생성 통합 테스트."""

    @pytest.fixture
    def service(self) -> StoryGenerationService:
        """스토리 생성 서비스 인스턴스."""
        return StoryGenerationService()

    @pytest.mark.asyncio
    async def test_generate_summary_returns_text(self, service: StoryGenerationService):
        """줄거리 생성이 텍스트를 반환한다."""
        result = await service.generate_summary(
            title="용사의 모험",
            description="평범한 소년이 용사가 되어 마왕을 물리치는 이야기",
        )

        assert result is not None
        assert result.summary is not None
        assert len(result.summary) > 100  # 최소 100자 이상
        assert isinstance(result.summary, str)

    @pytest.mark.asyncio
    async def test_generate_summary_includes_story_elements(self, service: StoryGenerationService):
        """생성된 줄거리가 스토리 요소를 포함한다."""
        result = await service.generate_summary(
            title="마법학교의 비밀",
            description="마법학교에 입학한 소녀가 학교의 비밀을 파헤치는 이야기",
        )

        # 기본적인 스토리 요소가 포함되어야 함
        expected_keywords = ["마법", "학교", "소녀", "비밀", "입학"]
        found = [kw for kw in expected_keywords if kw in result.summary]
        assert found, f"기대 키워드 '{expected_keywords}' 중 없음. 실제: {result.summary[:200]}"

    @pytest.mark.asyncio
    async def test_generate_summary_handles_various_genres(self, service: StoryGenerationService):
        """다양한 장르의 줄거리를 생성할 수 있다."""
        # 로맨스
        romance_result = await service.generate_summary(
            title="첫사랑의 기억",
            description="대학생 남녀의 풋풋한 사랑 이야기",
        )
        assert len(romance_result.summary) > 50

        # SF
        sf_result = await service.generate_summary(
            title="2150년 지구",
            description="인류가 화성으로 이주한 미래 세계",
        )
        assert len(sf_result.summary) > 50

    @pytest.mark.asyncio
    async def test_generate_summary_with_korean_content(self, service: StoryGenerationService):
        """한글 제목과 설명으로 한글 줄거리를 생성한다."""
        result = await service.generate_summary(
            title="조선시대 궁중 비화",
            description="왕실의 숨겨진 비밀을 파헤치는 궁녀의 이야기",
        )

        # 한글이 포함되어 있어야 함
        korean_chars = [char for char in result.summary if "\uac00" <= char <= "\ud7a3"]
        assert korean_chars, f"한글이 포함되어 있지 않음. 실제: {result.summary[:200]}"


@skip_if_no_gemini
class TestCharacterGeneration:
    """캐릭터 생성 통합 테스트."""

    @pytest.fixture
    def service(self) -> StoryGenerationService:
        """스토리 생성 서비스 인스턴스."""
        return StoryGenerationService()

    @pytest.mark.asyncio
    async def test_generate_characters_returns_list(self, service: StoryGenerationService):
        """캐릭터 생성이 리스트를 반환한다."""
        summary = """
        어린 마법사 아리안은 마법 아카데미에 입학하여 친구 레온과 소피아를 만난다.
        그들은 함께 어둠의 마왕 발록을 물리치기 위한 모험을 떠난다.
        여정 중 현명한 스승 가르도프와 신비로운 요정 루미아의 도움을 받는다.
        """

        result = await service.generate_characters(
            title="마법사의 모험",
            description="마법 아카데미 이야기",
            summary=summary,
        )

        assert result is not None
        assert result.characters is not None
        assert len(result.characters) >= 1  # 최소 1명 이상

    @pytest.mark.asyncio
    async def test_generated_characters_have_required_fields(self, service: StoryGenerationService):
        """생성된 캐릭터가 필수 필드를 가진다."""
        summary = """
        용감한 기사 알렉스는 공주 엘라를 구하기 위해 모험을 떠난다.
        마법사 멀린의 도움을 받아 드래곤 스모그와 맞서 싸운다.
        """

        result = await service.generate_characters(
            title="기사의 모험",
            description="공주를 구하는 기사 이야기",
            summary=summary,
        )

        for character in result.characters:
            assert character.name is not None
            assert len(character.name) > 0
            assert character.description is not None
            assert len(character.description) > 0
            assert character.personality is not None
            assert len(character.personality) > 0

    @pytest.mark.asyncio
    async def test_generate_characters_removes_duplicates(self, service: StoryGenerationService):
        """중복된 캐릭터 이름을 제거한다."""
        # 같은 캐릭터가 여러 번 언급되는 줄거리
        summary = """
        주인공 민수는 학교에서 가장 인기 있는 학생이다.
        민수의 친구 지훈은 항상 민수 곁에 있다.
        민수와 지훈은 함께 축구를 한다.
        어느 날 민수는 전학생 수진을 만난다.
        """

        result = await service.generate_characters(
            title="학교 이야기",
            description="학교에서 벌어지는 우정 이야기",
            summary=summary,
        )

        # 이름 중복 확인
        names = [char.name for char in result.characters]
        unique_names = set(names)
        assert len(names) == len(unique_names), "중복된 캐릭터가 있습니다"

    @pytest.mark.asyncio
    async def test_generate_characters_from_long_summary(self, service: StoryGenerationService):
        """긴 줄거리에서 캐릭터를 추출한다."""
        # 청킹이 필요한 긴 줄거리
        summary = """
        먼 옛날, 평화로운 왕국 아르테미아에 어린 왕자 레오가 태어났다.
        레오는 어릴 때부터 마법에 재능을 보였고, 왕실 마법사 엘더의 가르침을 받았다.
        레오의 여동생 공주 리아는 치유 마법에 뛰어난 재능을 가졌다.

        왕국의 기사단장 마르코는 레오에게 검술을 가르쳤다.
        마르코의 딸 에밀리아는 레오의 소꿉친구로, 함께 자랐다.

        어느 날, 북쪽의 어둠의 군주 모르도르가 왕국을 침략했다.
        모르도르의 부하 흑기사 칼론은 왕궁을 습격했다.
        왕과 왕비는 레오와 리아를 도망시키고 성을 지켰다.

        레오와 리아는 숲속의 은둔자 현자 가이우스를 만났다.
        가이우스는 그들에게 고대의 예언에 대해 알려주었다.
        예언에 따르면, 선택받은 자만이 어둠을 물리칠 수 있다고 했다.

        여정 중 그들은 도적단의 리더 잭과 마주쳤다.
        잭은 처음에는 적이었지만, 나중에 동료가 되었다.
        잭의 부하 중 로빈이라는 소년도 함께 합류했다.

        마침내 그들은 모르도르의 성에 도착했다.
        최종 결전에서 레오는 자신의 진정한 힘을 깨달았다.
        """

        result = await service.generate_characters(
            title="왕국의 전설",
            description="어둠의 군주에 맞서는 왕자의 모험",
            summary=summary,
        )

        assert len(result.characters) >= 3  # 최소 3명 이상 추출
        # 주요 캐릭터가 포함되어야 함
        character_names = [char.name.lower() for char in result.characters]
        found = [name for name in character_names if "레오" in name]
        assert found, f"'레오' 캐릭터가 없음. 실제 캐릭터: {character_names}"


@skip_if_no_gemini
class TestStoryGenerationEdgeCases:
    """스토리 생성 엣지 케이스 테스트."""

    @pytest.fixture
    def service(self) -> StoryGenerationService:
        """스토리 생성 서비스 인스턴스."""
        return StoryGenerationService()

    @pytest.mark.asyncio
    async def test_generate_summary_with_short_description(self, service: StoryGenerationService):
        """짧은 설명으로도 줄거리를 생성한다."""
        result = await service.generate_summary(
            title="미스터리",
            description="탐정 이야기",
        )

        assert result.summary is not None
        assert len(result.summary) > 20

    @pytest.mark.asyncio
    async def test_generate_characters_with_minimal_summary(self, service: StoryGenerationService):
        """최소한의 줄거리에서 캐릭터를 추출한다."""
        summary = "영웅 아서가 드래곤을 물리친다."

        result = await service.generate_characters(
            title="영웅전",
            description="드래곤과 싸우는 영웅",
            summary=summary,
        )

        assert result.characters is not None
        # 최소 1명은 추출되어야 함
        assert len(result.characters) >= 1

    @pytest.mark.asyncio
    async def test_generate_summary_with_english_input(self, service: StoryGenerationService):
        """영어 입력도 처리할 수 있다."""
        result = await service.generate_summary(
            title="The Last Knight",
            description="A story about the last knight in a dying kingdom",
        )

        assert result.summary is not None
        assert len(result.summary) > 50
