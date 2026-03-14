from src.utils.text import word_boundary_match


class TestWordBoundaryMatch:
    def test_korean_exact_match(self):
        assert word_boundary_match("공작", "공작") is True

    def test_korean_word_in_sentence(self):
        assert word_boundary_match("공작", "공작 부인이 왔다") is True

    def test_korean_no_match_substring(self):
        assert word_boundary_match("공작", "공작소에서 일한다") is False

    def test_korean_no_match_prefix(self):
        assert word_boundary_match("왕", "왕따를 당했다") is False

    def test_korean_match_with_space(self):
        assert word_boundary_match("왕", "왕 이 나타났다") is True

    def test_korean_match_at_end(self):
        assert word_boundary_match("왕", "그는 위대한 왕") is True

    def test_korean_match_with_punctuation(self):
        assert word_boundary_match("공주", "공주, 너는 아름답다") is True
        assert word_boundary_match("공주", "공주!") is True

    def test_korean_match_with_particle(self):
        assert word_boundary_match("홍길동", "홍길동이 왔다") is True
        assert word_boundary_match("홍길동", "홍길동을 만났다") is True
        assert word_boundary_match("홍길동", "홍길동의 집") is True
        assert word_boundary_match("공주", "공주가 있었다") is True

    def test_english_exact_match(self):
        assert word_boundary_match("king", "king") is True

    def test_english_word_in_sentence(self):
        assert word_boundary_match("king", "the king is here") is True

    def test_english_no_match_substring(self):
        assert word_boundary_match("king", "kingston") is False

    def test_english_no_match_suffix(self):
        assert word_boundary_match("king", "viking") is False

    def test_english_case_insensitive(self):
        assert word_boundary_match("King", "the KING is here") is True
        assert word_boundary_match("KING", "the king is here") is True

    def test_empty_keyword(self):
        assert word_boundary_match("", "some text") is True

    def test_empty_text(self):
        assert word_boundary_match("공주", "") is False
