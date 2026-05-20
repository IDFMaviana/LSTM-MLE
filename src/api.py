from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(title="LSTM Stock Predictor", version="1.0")


class PredictRequest(BaseModel):
    symbol: str
    history: List[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    try:
        result = predict(payload.symbol.upper(), payload.history)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
