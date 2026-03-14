from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google Gemini
    google_api_key: str

    # ElevenLabs
    elevenlabs_api_key: str

    # Database
    database_url: str

    # NestJS Backend
    nestjs_backend_url: str

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


settings = Settings()  # type: ignore[call-arg]
