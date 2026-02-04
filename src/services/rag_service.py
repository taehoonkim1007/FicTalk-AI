"""RAG 서비스 - 벡터 검색을 통한 관련 컨텍스트 조회."""

import logging
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.constants.settings import (
    KEYWORD_BOOST,
    KOREAN_STOPWORDS,
    RERANK_FINAL_TOP_K,
    RERANK_INITIAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAGService:
    """pgvector를 사용한 RAG 검색 서비스."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()

    def _extract_keywords(self, query: str) -> list[str]:
        """쿼리에서 키워드 추출 (불용어 제거).

        Args:
            query: 사용자 질문

        Returns:
            키워드 리스트
        """
        # 한글, 영문, 숫자만 추출
        words = re.findall(r"[가-힣a-zA-Z0-9]+", query)
        # 불용어 제거 및 2글자 이상만 유지
        keywords = [w for w in words if w not in KOREAN_STOPWORDS and len(w) >= 2]
        return keywords

    def _calculate_keyword_boost(self, content: str, keywords: list[str]) -> float:
        """청크 내용과 키워드 매칭 점수 계산.

        Args:
            content: 청크 내용
            keywords: 키워드 리스트

        Returns:
            부스트 점수 (0.0 ~ KEYWORD_BOOST)
        """
        if not keywords:
            return 0.0

        content_lower = content.lower()
        matched = sum(1 for kw in keywords if kw.lower() in content_lower)
        # 매칭된 키워드 비율에 따라 부스트
        return (matched / len(keywords)) * KEYWORD_BOOST

    async def search_relevant_chunks(
        self,
        db: AsyncSession,
        story_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """사용자 쿼리와 관련된 스토리 청크 검색.

        Args:
            db: 데이터베이스 세션
            story_id: 스토리 ID
            query: 사용자 질문/메시지
            top_k: 반환할 청크 수

        Returns:
            관련 청크 내용 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # pgvector cosine similarity 검색
            # 1 - cosine_distance = cosine_similarity (높을수록 유사)
            result = await db.execute(
                text("""
                    SELECT content, 1 - (embedding <=> :query_embedding::vector) as similarity
                    FROM "StoryContent"
                    WHERE "storyId" = :story_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding::vector
                    LIMIT :top_k
                """),
                {
                    "query_embedding": embedding_str,
                    "story_id": story_id,
                    "top_k": top_k,
                },
            )

            rows = result.fetchall()

            if not rows:
                logger.info(f"스토리 {story_id}: 관련 청크 없음")
                return []

            # 유사도 임계값 이상인 청크만 필터링
            filtered_results = [(row[0], row[1]) for row in rows if row[1] >= SIMILARITY_THRESHOLD]

            if not filtered_results:
                logger.info(
                    f"스토리 {story_id}: 유사도 임계값({SIMILARITY_THRESHOLD}) 이상인 청크 없음 "
                    f"(최고 유사도: {rows[0][1]:.3f})"
                )
                return []

            chunks = [r[0] for r in filtered_results]
            similarities = [r[1] for r in filtered_results]

            logger.info(
                f"스토리 {story_id}: {len(chunks)}개 청크 검색 완료 "
                f"(유사도: {[f'{s:.3f}' for s in similarities]})"
            )

            return chunks

        except Exception as e:
            logger.error(f"RAG 검색 실패: {e}")
            # 검색 실패 시 빈 리스트 반환 (채팅은 계속 진행)
            return []

    async def search_relevant_chunks_with_scores(
        self,
        db: AsyncSession,
        story_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """사용자 쿼리와 관련된 스토리 청크 검색 (유사도 점수 포함).

        Args:
            db: 데이터베이스 세션
            story_id: 스토리 ID
            query: 사용자 질문/메시지
            top_k: 반환할 청크 수

        Returns:
            (청크 내용, 유사도) 튜플 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # pgvector cosine similarity 검색
            # 임베딩 벡터는 직접 삽입 (SQL injection 위험 없음 - 숫자 배열)
            sql = f"""
                SELECT content, 1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM "StoryContent"
                WHERE "storyId" = :story_id
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """
            result = await db.execute(
                text(sql),
                {"story_id": story_id, "top_k": top_k},
            )

            rows = result.fetchall()

            if not rows:
                logger.info(f"스토리 {story_id}: 관련 청크 없음")
                return []

            # (content, similarity) 튜플 리스트 반환
            results = [(row[0], float(row[1])) for row in rows]

            logger.info(
                f"스토리 {story_id}: {len(results)}개 청크 검색 (점수 포함) "
                f"(유사도: {[f'{s:.3f}' for _, s in results]})"
            )

            return results

        except Exception as e:
            logger.error(f"RAG 검색 실패 (점수 포함): {e}")
            return []

    async def search_with_reranking(
        self,
        db: AsyncSession,
        story_id: str,
        query: str,
    ) -> list[tuple[str, float]]:
        """Reranking을 적용한 청크 검색.

        1단계: 넓게 검색 (RERANK_INITIAL_TOP_K개)
        2단계: 키워드 부스팅으로 재정렬
        3단계: 상위 RERANK_FINAL_TOP_K개 반환

        Args:
            db: 데이터베이스 세션
            story_id: 스토리 ID
            query: 사용자 질문

        Returns:
            (청크 내용, 최종 점수) 튜플 리스트
        """
        try:
            # 1단계: 넓게 검색
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            sql = f"""
                SELECT content, 1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM "StoryContent"
                WHERE "storyId" = :story_id
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """
            result = await db.execute(
                text(sql),
                {"story_id": story_id, "top_k": RERANK_INITIAL_TOP_K},
            )
            rows = result.fetchall()

            if not rows:
                logger.info(f"스토리 {story_id}: 관련 청크 없음")
                return []

            # 2단계: 키워드 추출 및 부스팅
            keywords = self._extract_keywords(query)
            logger.debug(f"추출된 키워드: {keywords}")

            reranked = []
            for content, similarity in rows:
                boost = self._calculate_keyword_boost(content, keywords)
                final_score = similarity + boost
                reranked.append((content, final_score, similarity, boost))

            # 3단계: 최종 점수로 재정렬
            reranked.sort(key=lambda x: x[1], reverse=True)

            # 상위 RERANK_FINAL_TOP_K개 선택
            top_results = reranked[:RERANK_FINAL_TOP_K]

            # 로깅
            logger.info(
                f"스토리 {story_id}: Reranking 완료 "
                f"(키워드: {keywords}, "
                f"점수: {[f'{r[1]:.3f}(벡터:{r[2]:.3f}+부스트:{r[3]:.3f})' for r in top_results]})"
            )

            return [(content, final_score) for content, final_score, _, _ in top_results]

        except Exception as e:
            logger.error(f"Reranking 검색 실패: {e}")
            return []

    async def get_context_for_chat(
        self,
        db: AsyncSession,
        story_id: str,
        user_message: str,
        max_context_length: int = 1500,
    ) -> str:
        """채팅용 컨텍스트 생성.

        Args:
            db: 데이터베이스 세션
            story_id: 스토리 ID
            user_message: 사용자 메시지
            max_context_length: 최대 컨텍스트 길이

        Returns:
            포맷된 컨텍스트 문자열
        """
        chunks = await self.search_relevant_chunks(db, story_id, user_message)

        if not chunks:
            return ""

        # 청크들을 합쳐서 컨텍스트 생성
        context_parts: list[str] = []
        total_length = 0

        for i, chunk in enumerate(chunks, 1):
            if total_length + len(chunk) > max_context_length:
                break
            context_parts.append(f"[관련 내용 {i}]\n{chunk}")
            total_length += len(chunk)

        return "\n\n".join(context_parts)
