from fastapi import APIRouter, Request

from app.prediction import NewsPredictor
from app.schemas import (
    ModelInfo,
    NewsArticle,
    PredictionResult,
)


router = APIRouter()


def get_predictor(request: Request) -> NewsPredictor:
    return request.app.state.predictor


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/model", response_model=ModelInfo)
def model_info(request: Request):
    predictor = get_predictor(request)
    return ModelInfo(
        model=predictor.model_name,
        validation_macro_f1=predictor.validation_macro_f1,
    )


@router.post("/predict", response_model=PredictionResult)
def predict(article: NewsArticle, request: Request):
    return get_predictor(request).predict(article)