import asyncio
import logging
import uuid

from google import genai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings

logger = logging.getLogger(__name__)

# 동시 임베딩 생성 요청 수 제한 (Rate Limit 방지)
EMBEDDING_CONCURRENCY_LIMIT = 5


class EmbeddingService:
    """Gemini gemini-embedding-001을 사용한 임베딩 서비스."""

    MODEL_NAME = "gemini-embedding-001"
    EMBEDDING_DIMENSION = 768  # MRL로 768차원 출력
    CHUNK_SIZE = 500  # 청크당 최대 글자 수
    CHUNK_OVERLAP = 50  # 청크 간 중복 글자 수

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.google_api_key)
        self._api_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY_LIMIT)

    def chunk_text(self, text: str) -> list[str]:
        """텍스트를 오버랩이 있는 청크로 분할.

        Args:
            text: 분할할 텍스트

        Returns:
            청크 리스트
        """
        if not text or len(text) <= self.CHUNK_SIZE:
            return [text] if text else []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE

            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마침표, 물음표, 느낌표 찾기
                for delimiter in ["。", ".", "!", "?", "!", "?", "\n"]:
                    last_delimiter = text[start:end].rfind(delimiter)
                    if last_delimiter != -1 and last_delimiter > self.CHUNK_SIZE // 2:
                        end = start + last_delimiter + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 다음 청크 시작점 (오버랩 적용)
            start = end - self.CHUNK_OVERLAP if end < len(text) else end

        logger.info(f"텍스트 청킹 완료: {len(text)}자 → {len(chunks)}개 청크")
        return chunks

    async def generate_embedding(self, text_content: str) -> list[float]:
        """Gemini를 사용하여 텍스트 임베딩 생성.

        Args:
            text_content: 임베딩할 텍스트

        Returns:
            768차원 임베딩 벡터
        """
        try:
            result = await self._client.aio.models.embed_content(
                model=self.MODEL_NAME,
                contents=text_content,
                config={
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": self.EMBEDDING_DIMENSION,
                },
            )
            if not result.embeddings:
                raise RuntimeError("임베딩 결과가 비어있습니다")
            embedding = result.embeddings[0].values
            if embedding is None:
                raise RuntimeError("임베딩 값이 비어있습니다")
            logger.debug(f"임베딩 생성 완료: {len(text_content)}자 → {len(embedding)}차원")
            return list(embedding)
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise RuntimeError(f"임베딩 생성에 실패했습니다: {e}") from e

    async def generate_query_embedding(self, query: str) -> list[float]:
        """쿼리용 임베딩 생성 (검색 최적화).

        Args:
            query: 검색 쿼리

        Returns:
            768차원 임베딩 벡터
        """
        try:
            result = await self._client.aio.models.embed_content(
                model=self.MODEL_NAME,
                contents=query,
                config={
                    "task_type": "RETRIEVAL_QUERY",
                    "output_dimensionality": self.EMBEDDING_DIMENSION,
                },
            )
            if not result.embeddings:
                raise RuntimeError("쿼리 임베딩 결과가 비어있습니다")
            embedding = result.embeddings[0].values
            if embedding is None:
                raise RuntimeError("쿼리 임베딩 값이 비어있습니다")
            return list(embedding)
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 실패: {e}")
            raise RuntimeError(f"쿼리 임베딩 생성에 실패했습니다: {e}") from e

    async def process_story_summary(
        self,
        db: AsyncSession,
        story_id: str,
        summary: str,
    ) -> int:
        """스토리 요약을 청킹하고 임베딩 생성 후 DB 저장.

        임베딩 생성은 병렬로 처리되며, 세마포어로 동시 요청 수가 제한됩니다.

        Args:
            db: 데이터베이스 세션
            story_id: 스토리 ID
            summary: 스토리 요약 텍스트

        Returns:
            생성된 청크 수
        """
        # 기존 청크 삭제
        await db.execute(
            text('DELETE FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )

        # 텍스트 청킹
        chunks = self.chunk_text(summary)

        if not chunks:
            logger.warning(f"스토리 {story_id}: 청킹할 내용 없음")
            return 0

        async def generate_with_limit(chunk_content: str) -> list[float]:
            async with self._api_semaphore:
                return await self.generate_embedding(chunk_content)

        # 병렬로 임베딩 생성
        logger.info(f"스토리 {story_id}: {len(chunks)}개 청크 임베딩 병렬 생성 시작")
        embeddings = await asyncio.gather(*[generate_with_limit(chunk) for chunk in chunks])

        # 순차적으로 DB 저장 (순서 보장)
        for index, (chunk_content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = str(uuid.uuid4())

            # pgvector 형식으로 저장
            embedding_str = f"[{','.join(map(str, embedding))}]"

            await db.execute(
                text("""
                    INSERT INTO "StoryContent" (id, content, "chunkIndex", embedding, "createdAt", "storyId")
                    VALUES (:id, :content, :chunk_index, CAST(:embedding AS vector), NOW(), :story_id)
                """),
                {
                    "id": chunk_id,
                    "content": chunk_content,
                    "chunk_index": index,
                    "embedding": embedding_str,
                    "story_id": story_id,
                },
            )

        await db.commit()
        logger.info(f"스토리 {story_id}: {len(chunks)}개 청크 저장 완료")
        return len(chunks)

    async def get_story_chunk_count(self, db: AsyncSession, story_id: str) -> int:
        """스토리의 청크 수 조회."""
        result = await db.execute(
            text('SELECT COUNT(*) FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )
        return result.scalar() or 0
