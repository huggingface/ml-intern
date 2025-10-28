"""Shared data models for the HF agent project"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Discussion(BaseModel):
    """Model for a discussion thread"""

    title: str
    url: str
    topic_id: int
    category: int
    created_at: datetime


class QuestionAndSolution(BaseModel):
    """Model for a QA pair from a discussion"""

    discussion_title: str
    discussion_url: str
    discussion_topic_id: int
    discussion_category: int
    discussion_created_at: datetime
    thread: list[dict]
    question: str
    solution: str


class Correctness(str, Enum):
    yes = "yes"
    no = "no"


class JudgementResult(BaseModel):
    """Structured output for LLM judge evaluation"""

    extracted_final_answer: str = Field(
        description="The final exact/snippet answer extracted from the response"
    )
    reasoning: str = Field(
        description="Explanation of why the answer is correct or incorrect"
    )
    correct: Correctness = Field(description="'yes' if correct, 'no' if incorrect")
    confidence: int = Field(
        description="Confidence score between 0 and 100", ge=0, le=100
    )


class EvaluationResult(BaseModel):
    """Model for evaluation results including metadata"""

    success: bool
    judgement: JudgementResult | None = None
    error: str | None = None
