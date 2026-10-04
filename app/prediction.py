import joblib
import pandas as pd

from app.config import (
    BEST_MODEL_PATH,
    LABELS,
    LEGACY_MODEL_PATH,
    VECTORIZER_PATH,
)
from src.preprocess import preprocessor


class NewsPredictor:
    def __init__(self, model, vectorizer, model_name, validation_macro_f1=None):
        self.model = model
        self.vectorizer = vectorizer
        self.model_name = model_name
        self.validation_macro_f1 = validation_macro_f1

    @classmethod
    def load(cls):
        if BEST_MODEL_PATH.exists():
            bundle = joblib.load(BEST_MODEL_PATH)
            return cls(
                model=bundle["model"],
                vectorizer=bundle["vectorizer"],
                model_name=bundle["model_name"],
                validation_macro_f1=bundle.get("validation_macro_f1"),
            )

        if LEGACY_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            return cls(
                model=joblib.load(LEGACY_MODEL_PATH),
                vectorizer=joblib.load(VECTORIZER_PATH),
                model_name="LinearSVC",
            )

        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m src.train` first."
        )

    def predict(self, article):
        frame = pd.DataFrame(
            [{"Title": article.title, "Description": article.description}]
        )
        preprocessor(frame)
        features = self.vectorizer.transform(frame["clean"])
        class_index = int(self.model.predict(features)[0])
        return {"class_index": class_index, "category": LABELS[class_index]}