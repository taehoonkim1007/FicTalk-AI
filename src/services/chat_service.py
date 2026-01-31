import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import GenerationConfig

from src.config import settings
from src.models.schemas import ChatMessage, ChatResponseResponse
from src.utils.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# 재시도 설정
MAX_RETRIES = 3
INITIAL_DELAY = 2  # 초기 대기 시간 (초)
MAX_DELAY = 30  # 최대 대기 시간 (초)

# 세션 캐시 설정
SESSION_TTL_MINUTES = 30  # 세션 유효 시간


class ChatSessionCache:
    """캐릭터별 채팅 세션 캐시."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, character_key: str) -> tuple[Any, list[dict]] | None:
        """캐시된 세션과 히스토리 반환."""
        if character_key not in self._cache:
            return None

        entry = self._cache[character_key]
        if datetime.now() > entry["expires_at"]:
            del self._cache[character_key]
            return None

        return entry["chat"], entry["history"]

    def set(
        self,
        character_key: str,
        chat: Any,
        history: list[dict],
    ) -> None:
        """세션과 히스토리 캐싱."""
        self._cache[character_key] = {
            "chat": chat,
            "history": history,
            "expires_at": datetime.now() + timedelta(minutes=SESSION_TTL_MINUTES),
        }

    def update_history(self, character_key: str, history: list[dict]) -> None:
        """히스토리만 업데이트."""
        if character_key in self._cache:
            self._cache[character_key]["history"] = history
            self._cache[character_key]["expires_at"] = datetime.now() + timedelta(
                minutes=SESSION_TTL_MINUTES
            )

    def invalidate(self, character_key: str) -> None:
        """캐시 무효화."""
        if character_key in self._cache:
            del self._cache[character_key]

    def clear_expired(self) -> None:
        """만료된 캐시 정리."""
        now = datetime.now()
        expired_keys = [key for key, entry in self._cache.items() if now > entry["expires_at"]]
        for key in expired_keys:
            del self._cache[key]


class ChatService:
    """Google Gemini를 사용한 캐릭터 채팅 서비스."""

    MODEL_NAME = "gemini-2.0-flash-lite"

    def __init__(self) -> None:
        genai.configure(api_key=settings.google_api_key)
        self.generation_config = GenerationConfig(
            temperature=0.8,
            max_output_tokens=512,
        )
        self._session_cache = ChatSessionCache()
        self._model_cache: dict[str, genai.GenerativeModel] = {}

    def _get_character_key(
        self, character_name: str, story_title: str, user_id: str = "default"
    ) -> str:
        """캐릭터별 고유 키 생성."""
        return f"{user_id}:{story_title}:{character_name}"

    def _get_or_create_model(
        self, character_key: str, system_instruction: str
    ) -> genai.GenerativeModel:
        """모델 캐시에서 가져오거나 새로 생성."""
        if character_key not in self._model_cache:
            self._model_cache[character_key] = genai.GenerativeModel(
                model_name=self.MODEL_NAME,
                system_instruction=system_instruction,
                generation_config=self.generation_config,
            )
            logger.info(f"새 모델 생성: {character_key}")
        return self._model_cache[character_key]

    async def generate_response(
        self,
        character_name: str,
        character_role: str,
        character_personality: str,
        story_title: str,
        story_summary: str,
        messages: list[ChatMessage],
        user_message: str,
    ) -> ChatResponseResponse:
        """캐릭터로서 응답 생성 (세션 캐싱 적용)."""
        character_key = self._get_character_key(character_name, story_title)

        # 시스템 프롬프트 구성
        system_prompt = CHAT_SYSTEM_PROMPT.format(
            character_name=character_name,
            character_role=character_role,
            character_personality=character_personality or "특별한 성격 설정 없음",
            story_title=story_title,
            story_summary=story_summary[:1000] if story_summary else "줄거리 정보 없음",
        )

        # 캐시된 모델 가져오기 또는 생성
        model = self._get_or_create_model(character_key, system_prompt)

        # 캐시된 세션 확인
        cached = self._session_cache.get(character_key)

        if cached:
            chat, cached_history = cached
            # 새로운 메시지가 있는지 확인
            new_messages = (
                messages[len(cached_history) :] if len(messages) > len(cached_history) else []
            )
            logger.info(
                f"캐시된 세션 사용: {character_key} "
                f"(캐시: {len(cached_history)}개, 새 메시지: {len(new_messages)}개)"
            )
        else:
            # 새 세션 생성 - 히스토리 없이 시작
            chat_history = []
            # 최근 메시지만 히스토리로 변환 (최대 5개로 축소)
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            for msg in recent_messages:
                role = "user" if msg.role == "user" else "model"
                chat_history.append({"role": role, "parts": [msg.content]})

            chat = model.start_chat(history=chat_history)
            self._session_cache.set(character_key, chat, messages)
            logger.info(f"새 세션 생성: {character_key} (히스토리: {len(chat_history)}개)")

        # 응답 생성 (재시도 로직 포함)
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await chat.send_message_async(user_message)
                response_text = response.text.strip()

                # 응답 정리 (캐릭터 이름 접두사 제거)
                if response_text.startswith(f"{character_name}:"):
                    response_text = response_text[len(f"{character_name}:") :].strip()

                # 히스토리 업데이트
                updated_messages = [
                    *messages,
                    ChatMessage(role="user", content=user_message),
                    ChatMessage(role="assistant", content=response_text),
                ]
                self._session_cache.update_history(character_key, updated_messages)

                logger.info(f"캐릭터 '{character_name}'({story_title}) 응답 생성 완료")
                return ChatResponseResponse(response=response_text)

            except ResourceExhausted as e:
                last_error = e
                delay = min(INITIAL_DELAY * (2**attempt), MAX_DELAY)
                logger.warning(
                    f"API 할당량 초과 (시도 {attempt + 1}/{MAX_RETRIES}). {delay}초 후 재시도..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                # 세션 오류 시 캐시 무효화 후 재시도
                if "history" in str(e).lower() or "session" in str(e).lower():
                    logger.warning(f"세션 오류, 캐시 무효화: {e}")
                    self._session_cache.invalidate(character_key)
                    if character_key in self._model_cache:
                        del self._model_cache[character_key]

                logger.error(f"채팅 응답 생성 실패: {e}")
                raise RuntimeError(f"응답 생성에 실패했습니다: {e}") from e

        # 모든 재시도 실패
        logger.error(f"최대 재시도 횟수 초과: {last_error}")
        raise RuntimeError(
            "API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        ) from last_error

    def clear_session(self, character_name: str, story_title: str) -> None:
        """특정 캐릭터의 세션 캐시 삭제."""
        character_key = self._get_character_key(character_name, story_title)
        self._session_cache.invalidate(character_key)
        if character_key in self._model_cache:
            del self._model_cache[character_key]
        logger.info(f"세션 캐시 삭제: {character_key}")
