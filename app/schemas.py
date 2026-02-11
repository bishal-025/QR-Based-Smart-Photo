from typing import List

from pydantic import BaseModel


class Score(BaseModel):
    label: str
    score: float


class ImagePrediction(BaseModel):
    filename: str
    label: str
    confidence: float
    all_scores: List[Score]


class BatchPredictionResponse(BaseModel):
    predictions: List[ImagePrediction]


