from pydantic import BaseModel, Field


# ============ Chat Schemas ============
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    character_id: str = Field(..., description="Character ID to chat with")
    user_message: str = Field(..., description="User's message")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation history"
    )


class ChatResponse(BaseModel):
    character_response: str = Field(..., description="Character's response")
    sources: list[str] = Field(default_factory=list, description="Source references from RAG")


# ============ Scenario Schemas ============
class ScenarioRequest(BaseModel):
    character_id: str = Field(..., description="Character ID")
    scenario_prompt: str = Field(..., description="What-if scenario prompt")
    context: str | None = Field(None, description="Additional context")


class ScenarioResponse(BaseModel):
    alternative_story: str = Field(..., description="Generated alternative story")
    reasoning: str = Field(..., description="Reasoning behind the story")


# ============ TTS Schemas ============
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    character_id: str = Field(..., description="Character ID for voice selection")


# ============ Index Schemas (for NestJS integration) ============
class IndexTextRequest(BaseModel):
    book_id: str = Field(..., description="Book/Story ID")
    character_id: str = Field(..., description="Character ID")
    text_content: str = Field(..., description="Text content to index")
    metadata: dict | None = Field(None, description="Additional metadata")


class IndexTextResponse(BaseModel):
    success: bool
    chunks_indexed: int = Field(..., description="Number of chunks indexed")


# ============ Story Generation Schemas ============
class SummaryGenerationRequest(BaseModel):
    title: str = Field(..., description="스토리 제목", max_length=200)
    description: str = Field(..., description="스토리 한줄 요약", max_length=500)


class SummaryGenerationResponse(BaseModel):
    summary: str = Field(..., description="생성된 줄거리")


class CharacterGenerationRequest(BaseModel):
    title: str = Field(..., description="스토리 제목", max_length=200)
    description: str = Field(..., description="스토리 한줄 요약", max_length=500)
    summary: str = Field(..., description="스토리 줄거리", max_length=4000)


class GeneratedCharacter(BaseModel):
    name: str = Field(..., description="캐릭터 이름")
    role: str = Field(..., description="역할 (주인공 또는 조연)")
    description: str = Field(..., description="줄거리에 명시된 팩트 (나이, 직업, 관계, 외모 등)")
    personality: str = Field(..., description="인물의 성격과 심리 서술형 묘사")
    firstMessage: str = Field(..., description="첫 인사 메시지")


class CharacterGenerationResponse(BaseModel):
    characters: list[GeneratedCharacter] = Field(..., description="생성된 캐릭터 목록")


# ============ Image Generation Schemas ============
class ProfileImageRequest(BaseModel):
    name: str = Field(..., description="캐릭터 이름", max_length=100)
    role: str = Field(..., description="역할 (주인공 또는 조연)", max_length=50)
    description: str = Field(..., description="캐릭터 설명", max_length=1000)
    personality: str = Field(..., description="캐릭터 성격", max_length=500)


class ProfileImageResponse(BaseModel):
    image_base64: str = Field(..., description="생성된 이미지 (base64 인코딩)")
    prompt_used: str = Field(..., description="이미지 생성에 사용된 프롬프트")
