import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, embedding, health, image_generation, story_generation, tts
from src.config import settings
from src.models.database import init_db
from src.services.chat_service import ChatService
from src.services.embedding_service import EmbeddingService
from src.services.image_generation_service import ImageGenerationService
from src.services.story_generation_service import StoryGenerationService
from src.services.voice_service import VoiceService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    await init_db()

    # 서비스 싱글톤 초기화
    app.state.chat_service = ChatService()
    app.state.embedding_service = EmbeddingService()
    app.state.story_generation_service = StoryGenerationService()
    app.state.image_generation_service = ImageGenerationService()
    app.state.voice_service = VoiceService()
    logger.info("서비스 초기화 완료")

    yield

    # Shutdown
    logger.info("서비스 종료")


app = FastAPI(
    title="FicTalk AI",
    description="AI service for character conversation with RAG",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    story_generation.router, prefix="/api/story-generation", tags=["Story Generation"]
)
app.include_router(
    image_generation.router, prefix="/api/image-generation", tags=["Image Generation"]
)
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(embedding.router, prefix="/api/embeddings", tags=["Embeddings"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
