import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_embedding_service
from src.models.database import get_db
from src.models.schemas import (
    BatchProcessResponse,
    ChunkCountResponse,
    ProcessStoryRequest,
    ProcessStoryResponse,
    StoryProcessResult,
)
from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/story", response_model=ProcessStoryResponse)
async def process_story_embedding(
    request: ProcessStoryRequest,
    db: AsyncSession = Depends(get_db),
    service: EmbeddingService = Depends(get_embedding_service),
) -> ProcessStoryResponse:
    """스토리 요약을 청킹하고 임베딩 생성.

    - **story_id**: 스토리 UUID
    - **summary**: 스토리 줄거리 텍스트

    스토리 요약을 500자 단위로 청킹하고, 각 청크에 대해
    Gemini gemini-embedding-001 모델로 768차원 임베딩을 생성합니다.
    """
    try:
        chunk_count = await service.process_story_summary(
            db=db,
            story_id=request.story_id,
            summary=request.summary,
        )

        return ProcessStoryResponse(
            story_id=request.story_id,
            chunk_count=chunk_count,
            message=f"{chunk_count}개의 청크가 생성되었습니다.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/story/{story_id}/chunks", response_model=ChunkCountResponse)
async def get_story_chunks(
    story_id: str,
    db: AsyncSession = Depends(get_db),
    service: EmbeddingService = Depends(get_embedding_service),
) -> ChunkCountResponse:
    """스토리의 임베딩 청크 수 조회."""
    chunk_count = await service.get_story_chunk_count(db, story_id)

    return ChunkCountResponse(
        story_id=story_id,
        chunk_count=chunk_count,
    )


@router.post("/batch", response_model=BatchProcessResponse)
async def process_all_stories(
    db: AsyncSession = Depends(get_db),
    service: EmbeddingService = Depends(get_embedding_service),
) -> BatchProcessResponse:
    """임베딩 없는 모든 스토리 일괄 처리.

    기존 DB에 저장된 스토리 중 임베딩이 없는 스토리를 찾아
    자동으로 임베딩을 생성합니다.
    """
    result = await db.execute(
        text("""
            SELECT s.id, s.title, s.summary
            FROM "Story" s
            WHERE s.id NOT IN (
                SELECT DISTINCT "storyId" FROM "StoryContent"
            )
            ORDER BY s."createdAt" DESC
        """)
    )
    stories = result.fetchall()

    if not stories:
        return BatchProcessResponse(
            processed=0,
            skipped=0,
            failed=0,
            results=[],
        )

    logger.info(f"일괄 처리 시작: {len(stories)}개 스토리")

    results: list[StoryProcessResult] = []
    processed = 0
    skipped = 0
    failed = 0

    for story_id, title, summary in stories:
        try:
            if not summary or len(summary.strip()) == 0:
                results.append(
                    StoryProcessResult(
                        story_id=story_id,
                        title=title or "제목 없음",
                        chunk_count=0,
                        status="skipped (no summary)",
                    )
                )
                skipped += 1
                continue

            chunk_count = await service.process_story_summary(
                db=db,
                story_id=story_id,
                summary=summary,
            )

            results.append(
                StoryProcessResult(
                    story_id=story_id,
                    title=title or "제목 없음",
                    chunk_count=chunk_count,
                    status="success",
                )
            )
            processed += 1
            logger.info(f"처리 완료: {title} ({chunk_count}개 청크)")

        except Exception as e:
            results.append(
                StoryProcessResult(
                    story_id=story_id,
                    title=title or "제목 없음",
                    chunk_count=0,
                    status=f"failed: {str(e)[:100]}",
                )
            )
            failed += 1
            logger.error(f"처리 실패: {title} - {e}")

    logger.info(f"일괄 처리 완료: 성공 {processed}, 건너뜀 {skipped}, 실패 {failed}")

    return BatchProcessResponse(
        processed=processed,
        skipped=skipped,
        failed=failed,
        results=results,
    )
