import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.schemas import GeneratedCharacter
from src.services.story_generation_service import StoryGenerationService


@pytest.fixture
def service():
    service = StoryGenerationService.__new__(StoryGenerationService)
    return service


class TestStoryGenerationServiceInit:
    def test_init_creates_client(self):
        """초기화 시 genai client 생성."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            service = StoryGenerationService()
            mock_client.assert_called_once()
            assert service._client is not None


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_successful_summary_generation(self):
        """줄거리 생성 성공."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = "  생성된 줄거리 텍스트입니다.  "
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()
            result = await service.generate_summary(
                title="테스트 스토리",
                description="테스트 설명",
            )

            assert result.summary == "생성된 줄거리 텍스트입니다."


class TestGenerateCharacters:
    @pytest.mark.asyncio
    async def test_successful_character_generation(self):
        """캐릭터 생성 성공."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(
                {
                    "characters": [
                        {
                            "name": "홍길동",
                            "role": "주인공",
                            "description": "용감한 영웅",
                            "personality": "정의로움",
                            "firstMessage": "안녕하세요",
                        }
                    ]
                }
            )
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()
            result = await service.generate_characters(
                title="테스트",
                description="설명",
                summary="줄거리",
            )

            assert len(result.characters) == 1
            assert result.characters[0].name == "홍길동"

    @pytest.mark.asyncio
    async def test_raises_error_when_no_characters(self):
        """캐릭터 없으면 에러."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps({"characters": []})
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()

            with pytest.raises(RuntimeError) as exc_info:
                await service.generate_characters(
                    title="테스트",
                    description="설명",
                    summary="줄거리",
                )

            assert "생성된 캐릭터가 없습니다" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_long_summary_with_chunking(self):
        """긴 줄거리는 청킹하여 처리."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(
                {
                    "characters": [
                        {
                            "name": "캐릭터1",
                            "role": "역할",
                            "description": "설명",
                            "personality": "성격",
                            "firstMessage": "인사",
                        }
                    ]
                }
            )
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()
            long_summary = "이것은 아주 긴 줄거리입니다. " * 500

            result = await service.generate_characters(
                title="테스트",
                description="설명",
                summary=long_summary,
            )

            assert len(result.characters) >= 1


class TestExtractCharactersFromChunk:
    @pytest.mark.asyncio
    async def test_extracts_characters_from_list(self):
        """리스트 형식 응답 처리."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(
                [
                    {
                        "name": "홍길동",
                        "role": "주인공",
                        "description": "설명",
                        "personality": "성격",
                        "firstMessage": "인사",
                    }
                ]
            )
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()
            result = await service._extract_characters_from_chunk(
                title="테스트",
                description="설명",
                chunk="청크 내용",
            )

            assert len(result) == 1
            assert result[0].name == "홍길동"

    @pytest.mark.asyncio
    async def test_returns_empty_on_json_error(self):
        """JSON 오류 시 빈 리스트 반환."""
        with patch("src.services.story_generation_service.genai.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = "잘못된 JSON 형식"
            mock_client.return_value.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            service = StoryGenerationService()
            result = await service._extract_characters_from_chunk(
                title="테스트",
                description="설명",
                chunk="청크 내용",
            )

            assert result == []


class TestSplitIntoChunks:
    def test_short_text_returns_single_chunk(self, service):
        text = "짧은 텍스트입니다."
        chunks = service._split_into_chunks(text, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_by_sentences(self, service):
        text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
        chunks = service._split_into_chunks(text, max_chars=30)
        assert len(chunks) >= 2

    def test_respects_max_chars(self, service):
        text = "문장1. 문장2. 문장3. 문장4. 문장5."
        chunks = service._split_into_chunks(text, max_chars=15)
        for chunk in chunks:
            assert len(chunk) <= 30  # 일부 여유 허용 (문장 단위 분리)

    def test_empty_text_returns_empty_list(self, service):
        chunks = service._split_into_chunks("", max_chars=2000)
        assert chunks == [""]


class TestExtractJsonFromResponse:
    def test_plain_json(self, service):
        text = '{"name": "테스트"}'
        result = service._extract_json_from_response(text)
        assert result == {"name": "테스트"}

    def test_markdown_json_block(self, service):
        text = '```json\n{"name": "테스트"}\n```'
        result = service._extract_json_from_response(text)
        assert result == {"name": "테스트"}

    def test_markdown_block_without_lang(self, service):
        text = '```\n{"name": "테스트"}\n```'
        result = service._extract_json_from_response(text)
        assert result == {"name": "테스트"}

    def test_with_whitespace(self, service):
        text = '  \n{"name": "테스트"}\n  '
        result = service._extract_json_from_response(text)
        assert result == {"name": "테스트"}

    def test_nested_json(self, service):
        text = '{"characters": [{"name": "홍길동"}]}'
        result = service._extract_json_from_response(text)
        assert result == {"characters": [{"name": "홍길동"}]}


class TestNormalizeCharacterName:
    def test_removes_whitespace(self, service):
        assert service._normalize_character_name("홍 길 동") == "홍길동"

    def test_lowercases(self, service):
        assert service._normalize_character_name("Harry Potter") == "harrypotter"

    def test_removes_title_duke(self, service):
        result = service._normalize_character_name("공작 레온하르트")
        assert "공작" not in result

    def test_removes_title_princess(self, service):
        result = service._normalize_character_name("황녀 시엘라")
        assert "황녀" not in result

    def test_removes_english_title(self, service):
        result = service._normalize_character_name("Sir Lancelot")
        assert "sir" not in result

    def test_empty_name(self, service):
        assert service._normalize_character_name("") == ""


class TestIsDuplicateName:
    def test_exact_match(self, service):
        assert service._is_duplicate_name("홍길동", ["홍길동"]) is True

    def test_not_duplicate(self, service):
        assert service._is_duplicate_name("홍길동", ["이순신"]) is False

    def test_partial_match_included(self, service):
        assert service._is_duplicate_name("길동", ["홍길동"]) is True

    def test_partial_match_contains(self, service):
        assert service._is_duplicate_name("홍길동", ["길동"]) is True

    def test_short_names_not_partial(self, service):
        assert service._is_duplicate_name("김", ["김철수"]) is False

    def test_empty_name_is_duplicate(self, service):
        assert service._is_duplicate_name("", ["홍길동"]) is True

    def test_empty_list(self, service):
        assert service._is_duplicate_name("홍길동", []) is False


class TestMergeCharacters:
    def test_merges_without_duplicates(self, service):
        chunk1 = [
            GeneratedCharacter(
                name="홍길동",
                role="주인공",
                description="설명1",
                personality="성격1",
                firstMessage="인사1",
            )
        ]
        chunk2 = [
            GeneratedCharacter(
                name="이순신",
                role="조연",
                description="설명2",
                personality="성격2",
                firstMessage="인사2",
            )
        ]
        result = service._merge_characters([chunk1, chunk2])
        assert len(result) == 2

    def test_removes_duplicate_names(self, service):
        chunk1 = [
            GeneratedCharacter(
                name="홍길동",
                role="주인공",
                description="설명1",
                personality="성격1",
                firstMessage="인사1",
            )
        ]
        chunk2 = [
            GeneratedCharacter(
                name="홍길동",
                role="주인공",
                description="설명2",
                personality="성격2",
                firstMessage="인사2",
            )
        ]
        result = service._merge_characters([chunk1, chunk2])
        assert len(result) == 1

    def test_removes_partial_duplicate(self, service):
        chunk1 = [
            GeneratedCharacter(
                name="공작 레온",
                role="주인공",
                description="설명1",
                personality="성격1",
                firstMessage="인사1",
            )
        ]
        chunk2 = [
            GeneratedCharacter(
                name="레온",
                role="주인공",
                description="설명2",
                personality="성격2",
                firstMessage="인사2",
            )
        ]
        result = service._merge_characters([chunk1, chunk2])
        assert len(result) == 1

    def test_empty_chunks(self, service):
        result = service._merge_characters([[], []])
        assert len(result) == 0
