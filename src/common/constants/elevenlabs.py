# =============================================================================
# ElevenLabs API 설정
# =============================================================================
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# 기본 Voice ID (매칭 실패 시 폴백)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - 기본 여성 음성
DEFAULT_MALE_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam - 기본 남성 음성

# TTS 모델
TTS_MODEL_ID = "eleven_multilingual_v2"

# =============================================================================
# API 타임아웃 설정 (초)
# =============================================================================
ELEVENLABS_SEARCH_TIMEOUT = 30.0
ELEVENLABS_TTS_TIMEOUT = 60.0

# =============================================================================
# Voice 매칭 점수 가중치
# =============================================================================
SCORE_GENDER_MATCH = 50  # 성별 일치 보너스
SCORE_GENDER_MISMATCH = -100  # 성별 불일치 페널티
SCORE_AGE_MATCH = 15  # 나이대 매칭
SCORE_ACCENT_MATCH = 10  # 억양 매칭
SCORE_USE_CASE_MATCH = 5  # use_case 매칭
SCORE_KEYWORD_MATCH = 3  # 키워드 매칭
SCORE_TONE_MATCH = 2  # tone 매칭

# =============================================================================
# VoiceSettings 기본값 및 범위
# =============================================================================
VOICE_STABILITY_DEFAULT = 0.5
VOICE_STABILITY_MIN = 0.0
VOICE_STABILITY_MAX = 1.0

VOICE_SIMILARITY_BOOST_DEFAULT = 0.75
VOICE_SIMILARITY_BOOST_MIN = 0.0
VOICE_SIMILARITY_BOOST_MAX = 1.0

VOICE_STYLE_DEFAULT = 0.0
VOICE_STYLE_MIN = 0.0
VOICE_STYLE_MAX = 1.0

VOICE_SPEED_DEFAULT = 1.0
VOICE_SPEED_MIN = 0.7
VOICE_SPEED_MAX = 1.2
