import asyncio
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.constants.settings import SESSION_TTL_MINUTES
from src.graphs.nodes import (
    generate_creative_response,
    generate_rag_response,
    retrieve_and_evaluate,
)
from src.graphs.state import ChatState
from src.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# 요청별 DB 세션을 저장하는 컨텍스트 변수 (동시성 안전)
_current_db_context: ContextVar[AsyncSession | None] = ContextVar("current_db", default=None)


class CacheEntry(TypedDict):
    """세션 캐시 엔트리."""

    messages: list[dict]
    expires_at: datetime


class ChatResult(TypedDict):
    """채팅 응답 결과."""

    response: str
    used_rag: bool
    mode: Literal["rag", "creative"]
    max_similarity: float
    session_id: str


class ChatSessionCache:
    """대화 세션 메모리 캐시.

    세션 ID를 키로 대화 내역을 저장하여 네트워크 비용을 절감합니다.
    TTL 기반으로 오래된 세션은 자동 만료됩니다.
    asyncio.Lock으로 동시성 보호됩니다.
    """

    def __init__(self, ttl_minutes: int = SESSION_TTL_MINUTES) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._ttl_minutes = ttl_minutes
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[dict] | None:
        """캐시된 메시지 반환."""
        async with self._lock:
            if session_id not in self._cache:
                return None

            entry = self._cache[session_id]
            if datetime.now() > entry["expires_at"]:
                del self._cache[session_id]
                logger.info(f"세션 만료: {session_id}")
                return None

            return entry["messages"]

    async def set(self, session_id: str, messages: list[dict]) -> None:
        """메시지 캐싱."""
        async with self._lock:
            self._cache[session_id] = {
                "messages": messages,
                "expires_at": datetime.now() + timedelta(minutes=self._ttl_minutes),
            }

    async def update(self, session_id: str, messages: list[dict]) -> None:
        """메시지 업데이트 및 TTL 갱신."""
        async with self._lock:
            if session_id in self._cache:
                self._cache[session_id]["messages"] = messages
                self._cache[session_id]["expires_at"] = datetime.now() + timedelta(
                    minutes=self._ttl_minutes
                )

    async def create_session(self, messages: list[dict] | None = None) -> str:
        """새 세션 생성 및 ID 반환."""
        session_id = str(uuid.uuid4())
        await self.set(session_id, messages or [])
        logger.info(f"새 세션 생성: {session_id}")
        return session_id

    async def clear_expired(self) -> int:
        """만료된 캐시 정리. 정리된 세션 수 반환."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [key for key, entry in self._cache.items() if now > entry["expires_at"]]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.info(f"만료된 세션 정리: {len(expired_keys)}개")
            return len(expired_keys)


class ChatService:
    """LangGraph 기반 채팅 서비스.

    RAG 검색 결과의 유사도에 따라 RAG 모드 또는 Creative 모드로 분기합니다.
    세션 캐싱을 통해 대화 내역을 서버에 저장하여 네트워크 비용을 절감합니다.

    워크플로우:
    ```
    사용자 메시지
        ↓
    [retrieve_and_evaluate] RAG 검색 + Self-RAG 평가
        ↓
    [조건부 분기]
        ├─ 적합성 평가 통과 (또는 유사도 ≥ 0.63) → [rag_response]
        └─ 적합성 평가 실패 (또는 유사도 < 0.63) → [creative_response]
        ↓
    응답 반환
    ```
    """

    def __init__(self) -> None:
        self._rag_service = RAGService()
        self._session_cache = ChatSessionCache()
        self._workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성."""
        workflow = StateGraph(ChatState)

        # 노드 추가
        workflow.add_node("retrieve_and_evaluate", self._retrieve_node)
        workflow.add_node("rag_response", self._rag_response_node)
        workflow.add_node("creative_response", self._creative_response_node)

        # 시작점 설정
        workflow.set_entry_point("retrieve_and_evaluate")

        # 조건부 엣지: 모드에 따라 분기
        workflow.add_conditional_edges(
            "retrieve_and_evaluate",
            self._route_by_mode,
            {
                "rag": "rag_response",
                "creative": "creative_response",
            },
        )

        # 종료 엣지
        workflow.add_edge("rag_response", END)
        workflow.add_edge("creative_response", END)

        return workflow

    def _route_by_mode(self, state: ChatState) -> str:
        """모드에 따라 라우팅."""
        return state["mode"]

    async def _retrieve_node(self, state: ChatState) -> dict:
        """RAG 검색 노드 (db 세션은 contextvars에서 가져옴)."""
        db = _current_db_context.get()
        if db is None:
            raise RuntimeError("DB session not set in context")
        return await retrieve_and_evaluate(
            state=state,
            rag_service=self._rag_service,
            db=db,
        )

    async def _rag_response_node(self, state: ChatState) -> dict:
        """RAG 모드 응답 노드."""
        return await generate_rag_response(state)

    async def _creative_response_node(self, state: ChatState) -> dict:
        """Creative 모드 응답 노드."""
        return await generate_creative_response(state)

    async def run(
        self,
        character_name: str,
        character_role: str,
        character_personality: str,
        story_id: str,
        story_title: str,
        story_summary: str,
        messages: list[dict] | None,
        user_message: str,
        db: AsyncSession,
        session_id: str | None = None,
    ) -> ChatResult:
        """채팅 응답 생성.

        Args:
            character_name: 캐릭터 이름
            character_role: 캐릭터 역할
            character_personality: 캐릭터 성격
            story_id: 스토리 ID (RAG 검색용)
            story_title: 스토리 제목
            story_summary: 스토리 줄거리
            messages: 이전 대화 내역 (session_id가 있으면 생략 가능)
            user_message: 사용자 메시지
            db: 데이터베이스 세션
            session_id: 세션 ID (캐시된 대화 사용 시)

        Returns:
            응답 결과 (response, used_rag, mode, max_similarity, session_id)
        """
        # DB 세션을 컨텍스트 변수에 저장 (요청별 스코프, 동시성 안전)
        token = _current_db_context.set(db)

        try:
            # 세션 캐시에서 메시지 가져오기 또는 새 세션 생성
            if session_id:
                cached_messages = await self._session_cache.get(session_id)
                if cached_messages is not None:
                    effective_messages = cached_messages
                    logger.info(f"캐시 히트: {session_id} (메시지 {len(cached_messages)}개)")
                else:
                    # 세션 만료 - 새 세션 생성
                    effective_messages = messages or []
                    session_id = await self._session_cache.create_session(effective_messages)
                    logger.info(f"세션 만료로 새 세션 생성: {session_id}")
            else:
                # 새 세션 생성
                effective_messages = messages or []
                session_id = await self._session_cache.create_session(effective_messages)

            # 초기 상태 구성
            initial_state: ChatState = {
                "character_name": character_name,
                "character_role": character_role,
                "character_personality": character_personality,
                "story_id": story_id,
                "story_title": story_title,
                "story_summary": story_summary,
                "messages": effective_messages,
                "user_message": user_message,
                "rag_results": [],
                "max_similarity": 0.0,
                "mode": "creative",
                "response": "",
                "used_rag": False,
            }

            # 워크플로우 컴파일 및 실행
            compiled = self._workflow.compile()
            result = await compiled.ainvoke(initial_state)

            # 응답 후 메시지 히스토리 업데이트
            updated_messages = [
                *effective_messages,
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result["response"]},
            ]
            await self._session_cache.update(session_id, updated_messages)

            logger.info(
                f"채팅 응답 생성 완료: {character_name} | "
                f"모드: {result['mode']} | "
                f"유사도: {result['max_similarity']:.3f} | "
                f"RAG 사용: {result['used_rag']} | "
                f"세션: {session_id}"
            )

            return {
                "response": result["response"],
                "used_rag": result["used_rag"],
                "mode": result["mode"],
                "max_similarity": result["max_similarity"],
                "session_id": session_id,
            }
        finally:
            # 컨텍스트 변수 복원
            _current_db_context.reset(token)
