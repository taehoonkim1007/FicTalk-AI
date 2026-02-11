from pydantic import BaseModel, Field


# ============ Chat Schemas ============
class ChatMessage(BaseModel):
    """대화 메시지."""

    role: str = Field(..., description="메시지 역할: 'user' 또는 'assistant'")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    """채팅 응답 생성 요청."""

    character_name: str = Field(..., description="캐릭터 이름")
    character_role: str = Field(..., description="캐릭터 역할")
    character_personality: str = Field(..., description="캐릭터 성격")
    story_id: str = Field(..., description="스토리 ID (RAG 검색용)")
    story_title: str = Field(..., description="스토리 제목")
    story_summary: str = Field(..., description="스토리 줄거리", max_length=4000)
    messages: list[ChatMessage] | None = Field(
        None, description="이전 대화 내역 (session_id가 있으면 생략 가능)"
    )
    user_message: str = Field(..., description="사용자 메시지", max_length=1000)
    session_id: str | None = Field(None, description="세션 ID (캐시된 대화 사용 시)")


class ChatResponse(BaseModel):
    """채팅 응답 생성 결과."""

    response: str = Field(..., description="캐릭터의 응답")
    mode: str = Field("creative", description="응답 모드: 'rag' 또는 'creative'")
    used_rag: bool = Field(False, description="RAG 컨텍스트 사용 여부")
    max_similarity: float = Field(0.0, description="RAG 검색 최대 유사도 점수")
    session_id: str = Field(..., description="세션 ID (다음 요청에 사용)")


# ============ Story Generation Schemas ============
class SummaryGenerationRequest(BaseModel):
    """줄거리 생성 요청."""

    title: str = Field(..., description="스토리 제목", max_length=100)
    description: str = Field(..., description="스토리 한줄 요약", max_length=400)


class SummaryGenerationResponse(BaseModel):
    """줄거리 생성 응답."""

    summary: str = Field(..., description="생성된 줄거리")


class CharacterGenerationRequest(BaseModel):
    """캐릭터 생성 요청."""

    title: str = Field(..., description="스토리 제목", max_length=100)
    description: str = Field(..., description="스토리 한줄 요약", max_length=400)
    summary: str = Field(..., description="스토리 줄거리", max_length=4000)


class GeneratedCharacter(BaseModel):
    """생성된 캐릭터 정보."""

    name: str = Field(..., description="캐릭터 이름")
    role: str = Field(..., description="역할 (주인공 또는 조연)")
    description: str = Field(..., description="줄거리에 명시된 팩트 (나이, 직업, 관계, 외모 등)")
    personality: str = Field(..., description="인물의 성격과 심리 서술형 묘사")
    firstMessage: str = Field(..., description="첫 인사 메시지")


class CharacterGenerationResponse(BaseModel):
    """캐릭터 생성 응답."""

    characters: list[GeneratedCharacter] = Field(..., description="생성된 캐릭터 목록")


# ============ Image Generation Schemas ============
class CharacterImageRequest(BaseModel):
    """프로필/캐릭터 배경 이미지 생성 요청."""

    description: str = Field(..., description="캐릭터 설명", max_length=400)
    personality: str = Field(..., description="캐릭터 성격", max_length=400)


class StoryImageRequest(BaseModel):
    """커버/스토리 배경 이미지 생성 요청."""

    title: str = Field(..., description="스토리 제목", max_length=100)
    description: str = Field(..., description="스토리 설명", max_length=400)
    summary: str = Field(..., description="스토리 요약", max_length=4000)


class ImageGenerationResponse(BaseModel):
    """이미지 생성 응답."""

    image_base64: str = Field(..., description="생성된 이미지 (base64 인코딩)")
    prompt_used: str = Field(..., description="이미지 생성에 사용된 프롬프트")


# ============ TTS Schemas ============
class VoiceSettings(BaseModel):
    """ElevenLabs TTS 음성 설정."""

    stability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="음성 안정성 (낮을수록 감정 풍부)"
    )
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, description="원본 음성 유사도")
    style: float = Field(default=0.0, ge=0.0, le=1.0, description="스타일 과장 정도")
    speed: float = Field(default=1.0, ge=0.7, le=1.2, description="말하기 속도")


class VoiceAttributes(BaseModel):
    """AI가 분석한 캐릭터 음성 특성."""

    gender: str = Field(..., description="성별: male, female, neutral")
    age: str = Field(..., description="나이대: child, young, middle, old")
    accent: str = Field(..., description="억양: Korean, American, British 등")
    tone: list[str] = Field(..., description="음색 특성: calm, warm, deep, intense 등")
    keywords: list[str] = Field(..., description="검색 키워드: narrator, gentle 등")


class VoiceRecommendRequest(BaseModel):
    """Voice 추천 요청."""

    description: str = Field(..., description="캐릭터 설명", max_length=400)
    personality: str = Field(..., description="캐릭터 성격", max_length=400)


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
    voice_settings: VoiceSettings | None = Field(None, description="음성 설정 (없으면 기본값)")


# ============ Embedding Schemas ============
class ProcessStoryRequest(BaseModel):
    """스토리 임베딩 요청."""

    story_id: str = Field(..., description="스토리 UUID")
    summary: str = Field(..., description="스토리 줄거리 텍스트")


class ProcessStoryResponse(BaseModel):
    """스토리 임베딩 응답."""

    story_id: str = Field(..., description="스토리 UUID")
    chunk_count: int = Field(..., description="생성된 청크 수")
    message: str = Field(..., description="처리 결과 메시지")


class ChunkCountResponse(BaseModel):
    """청크 수 조회 응답."""

    story_id: str = Field(..., description="스토리 UUID")
    chunk_count: int = Field(..., description="청크 수")


class StoryProcessResult(BaseModel):
    """개별 스토리 처리 결과."""

    story_id: str = Field(..., description="스토리 UUID")
    title: str = Field(..., description="스토리 제목")
    chunk_count: int = Field(..., description="생성된 청크 수")
    status: str = Field(..., description="처리 상태")


class BatchProcessResponse(BaseModel):
    """일괄 처리 응답."""

    processed: int = Field(..., description="처리 성공 수")
    skipped: int = Field(..., description="건너뛴 수")
    failed: int = Field(..., description="실패 수")
    results: list[StoryProcessResult] = Field(..., description="개별 처리 결과 목록")
