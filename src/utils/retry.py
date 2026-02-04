"""재시도 유틸리티.

지수 백오프를 적용한 API 호출 재시도 로직을 제공합니다.
주로 Gemini API의 429 (Rate Limit) 에러 처리에 사용됩니다.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.common.constants.settings import INITIAL_DELAY, MAX_DELAY, MAX_RETRIES

logger = logging.getLogger(__name__)

# 재시도 대상 에러 패턴
RETRYABLE_ERROR_PATTERNS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "rate limit",
    "quota exceeded",
)


def _is_retryable_error(error: Exception) -> bool:
    """재시도 가능한 에러인지 확인."""
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


async def _retry_logic(
    func: Callable[..., Awaitable],
    args: tuple,
    kwargs: dict,
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    operation_name: str,
    error_message: str,
):
    """공통 재시도 로직."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            if _is_retryable_error(e) and attempt < max_retries - 1:
                delay = min(initial_delay * (2**attempt), max_delay)
                logger.warning(
                    f"{operation_name} 재시도 {attempt + 1}/{max_retries}, "
                    f"{delay}초 대기: {type(e).__name__}"
                )
                await asyncio.sleep(delay)
                last_error = e
            else:
                if _is_retryable_error(e):
                    logger.error(f"{operation_name} 최대 재시도 횟수 초과: {e}")
                    raise RuntimeError(error_message) from e
                else:
                    raise

    raise RuntimeError(error_message) from last_error


async def retry_api_call_dict(
    func: Callable[..., Awaitable[dict]],
    *args,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_DELAY,
    max_delay: float = MAX_DELAY,
    operation_name: str = "API 호출",
    error_message: str = "요청 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
    **kwargs,
) -> dict:
    """dict를 반환하는 API 호출 재시도.

    Args:
        func: 실행할 비동기 함수 (dict 반환)
        *args: 함수에 전달할 위치 인자
        max_retries: 최대 재시도 횟수
        initial_delay: 초기 대기 시간(초)
        max_delay: 최대 대기 시간(초)
        operation_name: 로깅용 작업 이름
        error_message: 실패 시 사용자에게 표시할 에러 메시지
        **kwargs: 함수에 전달할 키워드 인자

    Returns:
        함수 실행 결과 (dict)

    Example:
        ```python
        async def call_api():
            return await client.generate_content(...)

        result = await retry_api_call_dict(
            call_api,
            operation_name="채팅 응답",
            error_message="채팅 응답 생성에 실패했습니다.",
        )
        ```
    """
    return await _retry_logic(
        func, args, kwargs, max_retries, initial_delay, max_delay, operation_name, error_message
    )


async def retry_api_call(
    func: Callable[..., Awaitable],
    *args,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_DELAY,
    max_delay: float = MAX_DELAY,
    operation_name: str = "API 호출",
    error_message: str = "요청 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
    **kwargs,
):
    """범용 API 호출 재시도.

    Args:
        func: 실행할 비동기 함수
        *args: 함수에 전달할 위치 인자
        max_retries: 최대 재시도 횟수
        initial_delay: 초기 대기 시간(초)
        max_delay: 최대 대기 시간(초)
        operation_name: 로깅용 작업 이름
        error_message: 실패 시 사용자에게 표시할 에러 메시지
        **kwargs: 함수에 전달할 키워드 인자

    Returns:
        함수 실행 결과
    """
    return await _retry_logic(
        func, args, kwargs, max_retries, initial_delay, max_delay, operation_name, error_message
    )


async def retry_api_call_tuple(
    func: Callable[..., Awaitable[tuple[str, str]]],
    *args,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_DELAY,
    max_delay: float = MAX_DELAY,
    operation_name: str = "API 호출",
    error_message: str = "요청 처리에 실패했습니다. 잠시 후 다시 시도해주세요.",
    **kwargs,
) -> tuple[str, str]:
    """tuple[str, str]을 반환하는 API 호출 재시도.

    Args:
        func: 실행할 비동기 함수 (tuple[str, str] 반환)
        *args: 함수에 전달할 위치 인자
        max_retries: 최대 재시도 횟수
        initial_delay: 초기 대기 시간(초)
        max_delay: 최대 대기 시간(초)
        operation_name: 로깅용 작업 이름
        error_message: 실패 시 사용자에게 표시할 에러 메시지
        **kwargs: 함수에 전달할 키워드 인자

    Returns:
        함수 실행 결과 (tuple[str, str])

    Example:
        ```python
        async def generate_image():
            return (image_base64, prompt_used)

        result = await retry_api_call_tuple(
            generate_image,
            operation_name="이미지 생성",
            error_message="이미지 생성에 실패했습니다.",
        )
        ```
    """
    return await _retry_logic(
        func, args, kwargs, max_retries, initial_delay, max_delay, operation_name, error_message
    )
