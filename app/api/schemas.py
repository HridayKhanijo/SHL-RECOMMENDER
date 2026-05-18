from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v):
        if v and v[-1].role != "user":
            raise ValueError("Last message must be from the user.")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def at_most_ten(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must have at most 10 items")
        return v