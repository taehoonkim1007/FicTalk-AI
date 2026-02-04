from typing import Literal, TypedDict


class ChatState(TypedDict):
    """채팅 워크플로우 상태."""

    # 입력 정보
    character_name: str
    character_role: str
    character_personality: str
    story_id: str
    story_title: str
    story_summary: str
    user_message: str
    messages: list[dict]  # 이전 대화 내역

    # RAG 검색 결과
    rag_results: list[tuple[str, float]]  # (content, similarity)
    max_similarity: float

    # 모드 결정
    mode: Literal["rag", "creative"]

    # 최종 출력
    response: str
    used_rag: bool
