from fastapi import Request

from src.services.chat_service import ChatService
from src.services.embedding_service import EmbeddingService
from src.services.image_generation_service import ImageGenerationService
from src.services.story_generation_service import StoryGenerationService
from src.services.voice_service import VoiceService


def get_chat_service(request: Request) -> ChatService:
    """ChatService 싱글톤 인스턴스 반환."""
    return request.app.state.chat_service


def get_embedding_service(request: Request) -> EmbeddingService:
    """EmbeddingService 싱글톤 인스턴스 반환."""
    return request.app.state.embedding_service


def get_story_generation_service(request: Request) -> StoryGenerationService:
    """StoryGenerationService 싱글톤 인스턴스 반환."""
    return request.app.state.story_generation_service


def get_image_generation_service(request: Request) -> ImageGenerationService:
    """ImageGenerationService 싱글톤 인스턴스 반환."""
    return request.app.state.image_generation_service


def get_voice_service(request: Request) -> VoiceService:
    """VoiceService 싱글톤 인스턴스 반환."""
    return request.app.state.voice_service
