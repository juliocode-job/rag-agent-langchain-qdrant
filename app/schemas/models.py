from pydantic import BaseModel
from typing import Any


class ChatRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What does the document say about AI?"}]
        }
    }


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []


class IngestResponse(BaseModel):
    message: str
    chunks_created: int
