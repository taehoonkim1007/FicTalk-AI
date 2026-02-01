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
class CharacterImageRequest(BaseModel):
    """프로필 이미지 및 캐릭터 배경 이미지 생성 요청."""

    description: str = Field(..., description="캐릭터 설명", max_length=1000)
    personality: str = Field(..., description="캐릭터 성격", max_length=500)


class StoryImageRequest(BaseModel):
    """커버 이미지 및 스토리 배경 이미지 생성 요청."""

    title: str = Field(..., description="스토리 제목", max_length=200)
    description: str = Field(..., description="스토리 설명", max_length=500)
    summary: str = Field(..., description="스토리 요약", max_length=4000)


class ImageGenerationResponse(BaseModel):
    """이미지 생성 공통 응답."""

    image_base64: str = Field(..., description="생성된 이미지 (base64 인코딩)")
    prompt_used: str = Field(..., description="이미지 생성에 사용된 프롬프트")


# ============ Chat Response Generation Schemas ============
class ChatResponseRequest(BaseModel):
    character_name: str = Field(..., description="캐릭터 이름")
    character_role: str = Field(..., description="캐릭터 역할")
    character_personality: str = Field(..., description="캐릭터 성격")
    story_title: str = Field(..., description="스토리 제목")
    story_summary: str = Field(..., description="스토리 줄거리", max_length=4000)
    messages: list[ChatMessage] = Field(default_factory=list, description="이전 대화 내역")
    user_message: str = Field(..., description="사용자 메시지", max_length=2000)


class ChatResponseResponse(BaseModel):
    response: str = Field(..., description="캐릭터의 응답")


# ============ Voice Recommendation Schemas ============
class VoiceSettings(BaseModel):
    """ElevenLabs TTS 음성 설정."""

    stability: float = Field(0.5, ge=0.0, le=1.0, description="음성 안정성 (낮을수록 감정 풍부)")
    similarity_boost: float = Field(0.75, ge=0.0, le=1.0, description="원본 음성 유사도")
    style: float = Field(0.0, ge=0.0, le=1.0, description="스타일 과장 정도")
    speed: float = Field(1.0, ge=0.7, le=1.2, description="말하기 속도")


class VoiceAttributes(BaseModel):
    """AI가 분석한 캐릭터 음성 특성."""

    gender: str = Field(..., description="성별: male, female, neutral")
    age: str = Field(..., description="나이대: child, young, middle, old")
    accent: str = Field(..., description="억양: Korean, American, British 등")
    tone: list[str] = Field(..., description="음색 특성: calm, warm, deep, intense 등")
    keywords: list[str] = Field(..., description="검색 키워드: narrator, gentle 등")


class VoiceRecommendRequest(BaseModel):
    """Voice 추천 요청."""

    description: str = Field(..., description="캐릭터 설명", max_length=1000)
    personality: str = Field(..., description="캐릭터 성격", max_length=500)


class VoiceRecommendResponse(BaseModel):
    """Voice 추천 응답."""

    voice_id: str = Field(..., description="ElevenLabs Voice ID")
    voice_name: str = Field(..., description="Voice 이름")
    attributes: VoiceAttributes = Field(..., description="분석된 음성 특성")
    voice_settings: VoiceSettings = Field(..., description="AI가 추천한 음성 설정")


class TTSSampleRequest(BaseModel):
    """TTS 샘플 음성 생성 요청."""

    voice_id: str = Field(..., description="ElevenLabs Voice ID")
    text: str = Field(..., description="샘플 텍스트", max_length=500)
    voice_settings: VoiceSettings | None = Field(None, description="음성 설정 (없으면 기본값 사용)")
