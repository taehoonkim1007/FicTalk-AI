from unittest.mock import AsyncMock, patch

import pytest

from src.utils.retry import (
    _is_retryable_error,
    _retry_logic,
    retry_api_call,
    retry_api_call_dict,
    retry_api_call_tuple,
)


class TestIsRetryableError:
    def test_429_error(self):
        error = Exception("Error 429: Too Many Requests")
        assert _is_retryable_error(error) is True

    def test_resource_exhausted(self):
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        assert _is_retryable_error(error) is True

    def test_rate_limit(self):
        error = Exception("Rate limit exceeded for this API")
        assert _is_retryable_error(error) is True

    def test_quota_exceeded(self):
        error = Exception("quota exceeded for the model")
        assert _is_retryable_error(error) is True

    def test_case_insensitive(self):
        error = Exception("RATE LIMIT exceeded")
        assert _is_retryable_error(error) is True

    def test_non_retryable_error(self):
        error = Exception("Invalid API key")
        assert _is_retryable_error(error) is False

    def test_400_error(self):
        error = Exception("Error 400: Bad Request")
        assert _is_retryable_error(error) is False

    def test_500_error_not_retryable(self):
        error = Exception("Error 500: Internal Server Error")
        assert _is_retryable_error(error) is False

    def test_empty_error(self):
        error = Exception("")
        assert _is_retryable_error(error) is False


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """첫 번째 시도에서 성공하면 결과 반환."""
        mock_func = AsyncMock(return_value={"result": "success"})

        result = await _retry_logic(
            func=mock_func,
            args=("arg1",),
            kwargs={"key": "value"},
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
            operation_name="테스트",
            error_message="테스트 실패",
        )

        assert result == {"result": "success"}
        mock_func.assert_called_once_with("arg1", key="value")

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error_then_success(self):
        """재시도 가능한 에러 후 성공."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                {"result": "success"},
            ]
        )

        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await _retry_logic(
                func=mock_func,
                args=(),
                kwargs={},
                max_retries=3,
                initial_delay=0.01,
                max_delay=0.1,
                operation_name="테스트",
                error_message="테스트 실패",
            )

        assert result == {"result": "success"}
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_runtime_error(self):
        """최대 재시도 횟수 초과 시 RuntimeError 발생."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
            ]
        )

        with (
            patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await _retry_logic(
                func=mock_func,
                args=(),
                kwargs={},
                max_retries=3,
                initial_delay=0.01,
                max_delay=0.1,
                operation_name="테스트",
                error_message="커스텀 에러 메시지",
            )

        assert str(exc_info.value) == "커스텀 에러 메시지"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """재시도 불가능한 에러는 즉시 발생."""
        mock_func = AsyncMock(side_effect=ValueError("Invalid input"))

        with pytest.raises(ValueError) as exc_info:
            await _retry_logic(
                func=mock_func,
                args=(),
                kwargs={},
                max_retries=3,
                initial_delay=0.01,
                max_delay=0.1,
                operation_name="테스트",
                error_message="테스트 실패",
            )

        assert "Invalid input" in str(exc_info.value)
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self):
        """지수 백오프 대기 시간 계산 확인."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                {"result": "success"},
            ]
        )
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("src.utils.retry.asyncio.sleep", side_effect=mock_sleep):
            await _retry_logic(
                func=mock_func,
                args=(),
                kwargs={},
                max_retries=3,
                initial_delay=1.0,
                max_delay=10.0,
                operation_name="테스트",
                error_message="테스트 실패",
            )

        assert sleep_calls[0] == 1.0  # initial_delay * 2^0
        assert sleep_calls[1] == 2.0  # initial_delay * 2^1

    @pytest.mark.asyncio
    async def test_delay_capped_at_max_delay(self):
        """대기 시간이 max_delay를 초과하지 않음."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                {"result": "success"},
            ]
        )
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("src.utils.retry.asyncio.sleep", side_effect=mock_sleep):
            await _retry_logic(
                func=mock_func,
                args=(),
                kwargs={},
                max_retries=5,
                initial_delay=1.0,
                max_delay=2.5,
                operation_name="테스트",
                error_message="테스트 실패",
            )

        assert sleep_calls[0] == 1.0  # 1.0 * 2^0 = 1.0
        assert sleep_calls[1] == 2.0  # 1.0 * 2^1 = 2.0
        assert sleep_calls[2] == 2.5  # min(1.0 * 2^2, 2.5) = 2.5


class TestRetryApiCallDict:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        """dict 반환 확인."""
        mock_func = AsyncMock(return_value={"key": "value"})

        result = await retry_api_call_dict(
            mock_func,
            "arg1",
            max_retries=2,
            initial_delay=0.01,
            max_delay=0.1,
        )

        assert result == {"key": "value"}
        mock_func.assert_called_once_with("arg1")

    @pytest.mark.asyncio
    async def test_with_kwargs(self):
        """키워드 인자 전달 확인."""
        mock_func = AsyncMock(return_value={"result": "ok"})

        result = await retry_api_call_dict(
            mock_func,
            custom_param="test_value",
            max_retries=2,
        )

        assert result == {"result": "ok"}
        mock_func.assert_called_once_with(custom_param="test_value")


class TestRetryApiCall:
    @pytest.mark.asyncio
    async def test_returns_any_type(self):
        """다양한 반환 타입 처리."""
        mock_func = AsyncMock(return_value="string_result")

        result = await retry_api_call(
            mock_func,
            max_retries=2,
            initial_delay=0.01,
        )

        assert result == "string_result"

    @pytest.mark.asyncio
    async def test_with_positional_and_keyword_args(self):
        """위치 인자와 키워드 인자 함께 전달."""
        mock_func = AsyncMock(return_value=42)

        result = await retry_api_call(
            mock_func,
            "pos1",
            "pos2",
            kwarg1="kw1",
            max_retries=2,
        )

        assert result == 42
        mock_func.assert_called_once_with("pos1", "pos2", kwarg1="kw1")


class TestRetryApiCallTuple:
    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """tuple[str, str] 반환 확인."""
        mock_func = AsyncMock(return_value=("image_data", "prompt_used"))

        result = await retry_api_call_tuple(
            mock_func,
            max_retries=2,
            initial_delay=0.01,
        )

        assert result == ("image_data", "prompt_used")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """재시도 후 성공 시 tuple 반환."""
        mock_func = AsyncMock(
            side_effect=[
                Exception("RESOURCE_EXHAUSTED"),
                ("base64_image", "generated_prompt"),
            ]
        )

        with patch("src.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_api_call_tuple(
                mock_func,
                max_retries=3,
                initial_delay=0.01,
                operation_name="이미지 생성",
            )

        assert result == ("base64_image", "generated_prompt")
        assert mock_func.call_count == 2
