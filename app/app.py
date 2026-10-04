from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import API_TITLE, API_VERSION
from app.prediction import NewsPredictor
from app.routes import router


@asynccontextmanager
async def lifespan(application: FastAPI):
	application.state.predictor = NewsPredictor.load()
	yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
app.include_router(router)
