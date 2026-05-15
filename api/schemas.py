"""Pydantic request/response models for the API."""
from pydantic import BaseModel


class ReportResponse(BaseModel):
    report: str
    model: str = "medgemma-1.5-4b-it"


class RetrievedItem(BaseModel):
    study_id: int | str
    report: str
    score: float


class QAResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedItem] = []
    retriever: str | None = None
