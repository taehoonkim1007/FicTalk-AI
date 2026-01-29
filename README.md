# FicTalk AI

AI service for FicTalk - Character conversation with RAG.

## Setup

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn src.main:app --reload
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Chat with character
- `POST /api/scenario/what-if` - Generate alternative story
- `POST /api/tts/generate` - Generate speech
- `POST /api/index/text` - Index text for RAG
