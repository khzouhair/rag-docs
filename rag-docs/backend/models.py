from pydantic import BaseModel
from typing import Optional, List

class HistorySource(BaseModel):
    source: str
    text: str

class HistoryEntry(BaseModel):
    role: str
    text: str
    sources: Optional[List[HistorySource]] = None

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    allow_dangerous_deserialization: Optional[bool] = False
    history: Optional[List[HistoryEntry]] = None

class SourceOut(BaseModel):
    source: str
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
    prompt: Optional[str] = None
