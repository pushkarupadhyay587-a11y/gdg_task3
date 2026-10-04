from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    title: str = Field(min_length=1, description="News article headline")
    description: str = Field(min_length=1, description="News article text")


class PredictionResult(BaseModel):
    class_index: int
    category: str


class ModelInfo(BaseModel):
    model: str
    validation_macro_f1: float | None = None