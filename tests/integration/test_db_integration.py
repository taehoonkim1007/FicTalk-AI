import asyncio
import uuid

import pytest
from sqlalchemy import text

from tests.integration.conftest import skip_if_no_db


@skip_if_no_db
class TestDatabaseTransactions:
    """데이터베이스 트랜잭션 테스트."""

    @pytest.mark.asyncio
    async def test_commit_persists_data(self, db_session, clean_db, story_id):
        """커밋 후 데이터가 영속되는지 확인."""
        chunk_id = str(uuid.uuid4())

        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES (:id, :content, :chunk_index, :story_id)
            """),
            {
                "id": chunk_id,
                "content": "테스트 청크",
                "chunk_index": 0,
                "story_id": story_id,
            },
        )
        await db_session.commit()

        # 새 쿼리로 확인
        result = await db_session.execute(
            text('SELECT content FROM "StoryContent" WHERE id = :id'),
            {"id": chunk_id},
        )
        row = result.fetchone()

        assert row is not None
        assert row[0] == "테스트 청크"

    @pytest.mark.asyncio
    async def test_rollback_reverts_data(self, db_session, clean_db, story_id):
        """롤백 시 데이터가 되돌려지는지 확인."""
        chunk_id = str(uuid.uuid4())

        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES (:id, :content, :chunk_index, :story_id)
            """),
            {
                "id": chunk_id,
                "content": "롤백될 청크",
                "chunk_index": 0,
                "story_id": story_id,
            },
        )
        # 롤백
        await db_session.rollback()

        # 데이터가 없어야 함
        result = await db_session.execute(
            text('SELECT COUNT(*) FROM "StoryContent" WHERE id = :id'),
            {"id": chunk_id},
        )
        count = result.scalar()

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_and_insert_atomicity(self, db_session, clean_db, story_id):
        """삭제 후 삽입의 원자성 확인."""
        # 기존 데이터 삽입
        old_chunk_id = str(uuid.uuid4())
        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES (:id, '기존 청크', 0, :story_id)
            """),
            {"id": old_chunk_id, "story_id": story_id},
        )
        await db_session.commit()

        # 삭제 후 새 데이터 삽입 (하나의 트랜잭션)
        await db_session.execute(
            text('DELETE FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )

        new_chunk_id = str(uuid.uuid4())
        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                VALUES (:id, '새 청크', 0, :story_id)
            """),
            {"id": new_chunk_id, "story_id": story_id},
        )
        await db_session.commit()

        # 기존 청크는 없고 새 청크만 있어야 함
        result = await db_session.execute(
            text('SELECT id, content FROM "StoryContent" WHERE "storyId" = :story_id'),
            {"story_id": story_id},
        )
        rows = result.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == new_chunk_id
        assert rows[0][1] == "새 청크"


@skip_if_no_db
class TestPgvectorOperations:
    """pgvector 관련 작업 테스트."""

    @pytest.mark.asyncio
    async def test_store_vector_embedding(self, db_session, clean_db, story_id):
        """768차원 벡터 저장."""
        chunk_id = str(uuid.uuid4())
        # 768차원 테스트 벡터 생성
        test_embedding = [0.1] * 768
        embedding_str = f"[{','.join(map(str, test_embedding))}]"

        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", embedding, "storyId")
                VALUES (:id, :content, :chunk_index, CAST(:embedding AS vector), :story_id)
            """),
            {
                "id": chunk_id,
                "content": "벡터 테스트 청크",
                "chunk_index": 0,
                "embedding": embedding_str,
                "story_id": story_id,
            },
        )
        await db_session.commit()

        # 벡터 조회
        result = await db_session.execute(
            text('SELECT embedding FROM "StoryContent" WHERE id = :id'),
            {"id": chunk_id},
        )
        row = result.fetchone()

        assert row is not None
        assert row[0] is not None

    @pytest.mark.asyncio
    async def test_cosine_similarity_search(self, db_session, clean_db, story_id):
        """코사인 유사도 검색."""
        # 두 개의 벡터 삽입 (하나는 쿼리와 유사, 하나는 다름)
        similar_embedding = [0.1] * 768
        different_embedding = [-0.1] * 768

        chunk1_id = str(uuid.uuid4())
        chunk2_id = str(uuid.uuid4())

        await db_session.execute(
            text("""
                INSERT INTO "StoryContent" (id, content, "chunkIndex", embedding, "storyId")
                VALUES
                    (:id1, '유사한 청크', 0, CAST(:emb1 AS vector), :story_id),
                    (:id2, '다른 청크', 1, CAST(:emb2 AS vector), :story_id)
            """),
            {
                "id1": chunk1_id,
                "id2": chunk2_id,
                "emb1": f"[{','.join(map(str, similar_embedding))}]",
                "emb2": f"[{','.join(map(str, different_embedding))}]",
                "story_id": story_id,
            },
        )
        await db_session.commit()

        # 쿼리 벡터로 유사도 검색
        query_embedding = [0.1] * 768
        query_str = f"[{','.join(map(str, query_embedding))}]"

        result = await db_session.execute(
            text(f"""
                SELECT content, 1 - (embedding <=> '{query_str}'::vector) as similarity
                FROM "StoryContent"
                WHERE "storyId" = :story_id AND embedding IS NOT NULL
                ORDER BY embedding <=> '{query_str}'::vector
                LIMIT 2
            """),
            {"story_id": story_id},
        )
        rows = result.fetchall()

        assert len(rows) == 2
        # 첫 번째 결과가 더 유사해야 함
        assert rows[0][1] > rows[1][1]
        assert rows[0][0] == "유사한 청크"


@skip_if_no_db
class TestConcurrentAccess:
    """동시 접근 테스트."""

    @pytest.mark.asyncio
    async def test_concurrent_inserts(self, test_engine, setup_database, story_id):
        """동시 삽입 처리."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        async_session_factory = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async def insert_chunk(chunk_num: int):
            async with async_session_factory() as session:
                chunk_id = str(uuid.uuid4())
                await session.execute(
                    text("""
                        INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                        VALUES (:id, :content, :chunk_index, :story_id)
                    """),
                    {
                        "id": chunk_id,
                        "content": f"동시 삽입 청크 {chunk_num}",
                        "chunk_index": chunk_num,
                        "story_id": story_id,
                    },
                )
                await session.commit()
                return chunk_id

        # 10개 동시 삽입
        tasks = [insert_chunk(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # 모든 삽입 성공 확인
        assert len(results) == 10

        # DB에서 확인
        async with async_session_factory() as session:
            result = await session.execute(
                text('SELECT COUNT(*) FROM "StoryContent" WHERE "storyId" = :story_id'),
                {"story_id": story_id},
            )
            count = result.scalar()

            # 정리
            await session.execute(
                text('DELETE FROM "StoryContent" WHERE "storyId" = :story_id'),
                {"story_id": story_id},
            )
            await session.commit()

        assert count == 10

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, test_engine, setup_database, story_id):
        """동시 읽기/쓰기 처리."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        async_session_factory = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # 초기 데이터 삽입
        async with async_session_factory() as session:
            chunk_id = str(uuid.uuid4())
            await session.execute(
                text("""
                    INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                    VALUES (:id, '초기 청크', 0, :story_id)
                """),
                {"id": chunk_id, "story_id": story_id},
            )
            await session.commit()

        async def read_chunk():
            async with async_session_factory() as session:
                result = await session.execute(
                    text('SELECT COUNT(*) FROM "StoryContent" WHERE "storyId" = :story_id'),
                    {"story_id": story_id},
                )
                return result.scalar()

        async def write_chunk(num: int):
            async with async_session_factory() as session:
                new_id = str(uuid.uuid4())
                await session.execute(
                    text("""
                        INSERT INTO "StoryContent" (id, content, "chunkIndex", "storyId")
                        VALUES (:id, :content, :chunk_index, :story_id)
                    """),
                    {
                        "id": new_id,
                        "content": f"추가 청크 {num}",
                        "chunk_index": num,
                        "story_id": story_id,
                    },
                )
                await session.commit()

        # 동시 읽기/쓰기
        tasks = [
            read_chunk(),
            write_chunk(1),
            read_chunk(),
            write_chunk(2),
            read_chunk(),
        ]
        await asyncio.gather(*tasks)

        # 최종 확인
        async with async_session_factory() as session:
            result = await session.execute(
                text('SELECT COUNT(*) FROM "StoryContent" WHERE "storyId" = :story_id'),
                {"story_id": story_id},
            )
            final_count = result.scalar()

            # 정리
            await session.execute(
                text('DELETE FROM "StoryContent" WHERE "storyId" = :story_id'),
                {"story_id": story_id},
            )
            await session.commit()

        # 초기 1개 + 추가 2개 = 3개
        assert final_count == 3
