import json
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.model import StockLSTM

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE_DIR / "models"


def _ensure_artifacts(symbol: str):
    metadata_path = ARTIFACTS / f"{symbol}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados ausentes para {symbol}")
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _artifact_path(metadata: dict, new_key: str, old_key: str) -> Path:
    filename = metadata.get(new_key)
    if not filename:
        fallback = metadata.get(old_key, "")
        filename = Path(fallback).name if fallback else ""
    if not filename:
        raise ValueError(f"Metadado {new_key} ausente")
    return ARTIFACTS / filename


@lru_cache(maxsize=4)
def _load_model(symbol: str):
    metadata = _ensure_artifacts(symbol)
    scaler_path = _artifact_path(metadata, "scaler_filename", "scaler_path")
    model_path = _artifact_path(metadata, "model_filename", "model_path")
    scaler = joblib.load(scaler_path)
    net = StockLSTM(
        input_size=1,
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
    )
    state_dict = torch.load(model_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return metadata, scaler, net


def predict(symbol: str, history: List[float]) -> dict:
    metadata, scaler, net = _load_model(symbol)
    lookback = int(metadata["lookback"])
    if len(history) < lookback:
        raise ValueError(f"History precisa ter pelo menos {lookback} valores")
    window = np.array(history[-lookback:], dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(window)
    tensor = torch.from_numpy(scaled).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = net(tensor)
    pred_array = pred_scaled.squeeze(0).cpu().numpy().reshape(-1, 1)
    pred_inverse = scaler.inverse_transform(pred_array).squeeze()
    return {"symbol": symbol, "prediction": float(pred_inverse), "lookback": lookback}
