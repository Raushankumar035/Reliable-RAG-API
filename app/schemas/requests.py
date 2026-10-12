from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the ingested documents.")
    top_k: Optional[int] = Field(
        None, ge=1, le=50, description="How many chunks to retrieve (defaults to server TOP_K)."
    )

    @field_validator("question")
    @classmethod
    def _clean(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("question must contain at least 3 characters")
        return value
