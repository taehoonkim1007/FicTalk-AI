import base64

from fastapi import APIRouter, HTTPException

from src.models.schemas import TTSSampleRequest, VoiceRecommendRequest, VoiceRecommendResponse
from src.services.voice_service import VoiceService

router = APIRouter()


@router.post("/voice-id", response_model=VoiceRecommendResponse)
async def get_voice_id(request: VoiceRecommendRequest) -> VoiceRecommendResponse:
    """캐릭터 Voice ID 조회 API.

    캐릭터의 설명과 성격을 AI로 분석하여 ElevenLabs Voice Library에서
    가장 적합한 voice ID를 반환합니다.

    - **description**: 캐릭터 설명 (최대 1000자)
    - **personality**: 캐릭터 성격 (최대 500자)

    Returns:
        - **voice_id**: ElevenLabs Voice ID
        - **voice_name**: Voice 이름
        - **attributes**: 분석된 음성 특성 (gender, age, accent, tone, keywords)
    """
    service = VoiceService()
    try:
        return await service.recommend_voice(request.description, request.personality)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sample")
async def generate_sample_audio(request: TTSSampleRequest) -> dict:
    """TTS 샘플 음성 생성 API.

    지정된 voice ID로 텍스트를 음성으로 변환하여 반환합니다.

    - **voice_id**: ElevenLabs Voice ID
    - **text**: 샘플 텍스트 (최대 500자)
    - **voice_settings**: 음성 설정 (optional)

    Returns:
        - **audio_base64**: 생성된 음성 (base64 인코딩, MP3 포맷)
    """
    service = VoiceService()
    try:
        audio_bytes = await service.generate_sample_audio(
            request.voice_id, request.text, request.voice_settings
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_base64": audio_base64}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
