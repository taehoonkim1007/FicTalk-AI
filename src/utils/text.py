import re

# 한국어 조사 패턴 (키워드 뒤에 올 수 있는 조사들)
KOREAN_PARTICLE_PATTERN = r"(?:이|가|을|를|은|는|의|에|에서|으로|로|와|과|도|만|까지|부터|에게|한테|께|입니다|입니까|이다)?"


def word_boundary_match(keyword: str, text: str) -> bool:
    """단어 경계를 고려한 키워드 매칭.

    한글: 키워드 앞에 한글 없음 + 키워드 뒤에 조사 허용 + 조사 뒤에 한글 없음
    영어: 단어 경계(\\b) 사용

    Args:
        keyword: 찾을 키워드
        text: 검색 대상 텍스트

    Returns:
        매칭 여부

    Examples:
        >>> word_boundary_match("공작", "공작 부인")
        True
        >>> word_boundary_match("공작", "공작소")
        False
        >>> word_boundary_match("홍길동", "홍길동이 왔다")
        True
        >>> word_boundary_match("king", "the king is here")
        True
        >>> word_boundary_match("king", "kingston")
        False
    """
    keyword_lower = keyword.lower()
    text_lower = text.lower()

    # 한글 키워드인 경우
    if re.search(r"[가-힣]", keyword):
        # 키워드 앞에 한글 없음 + 키워드 + 조사(선택) + 뒤에 한글 없음
        pattern = rf"(?<![가-힣]){re.escape(keyword_lower)}{KOREAN_PARTICLE_PATTERN}(?![가-힣])"
        return bool(re.search(pattern, text_lower))

    # 영어 키워드인 경우: 단어 경계 사용
    pattern = rf"\b{re.escape(keyword_lower)}\b"
    return bool(re.search(pattern, text_lower))
