from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceDataset:
    features: np.ndarray
    targets: np.ndarray


def download_price_data(
    symbol: str,
    start: Optional[str] = None,
    end_date: Optional[str] = None,
    years: Optional[int] = None,
) -> pd.DataFrame:
    end = pd.Timestamp(end_date).date() if end_date else date.today()
    if start is not None:
        start_date = pd.Timestamp(start).date()
    elif years is not None:
        start_date = end - timedelta(days=365 * years)
    else:
        raise ValueError("Informe start ou years para baixar os dados")
    if start_date >= end:
        raise ValueError("start precisa ser anterior a end")
    df = yf.download(
        symbol,
        start=start_date.isoformat(),
        end=end.isoformat(),
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} entre {start} e {end_date}")
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def create_sequences(values: np.ndarray, lookback: int) -> SequenceDataset:
    if lookback < 2:
        raise ValueError("lookback precisa ser pelo menos 2")
    X, y = [], []
    for idx in range(lookback, len(values)):
        X.append(values[idx - lookback : idx])
        y.append(values[idx])
    return SequenceDataset(
        features=np.array(X, dtype=np.float32), targets=np.array(y, dtype=np.float32)
    )


def prepare_data(
    df: pd.DataFrame, lookback: int, split_ratio: float = 0.8
) -> Tuple[SequenceDataset, SequenceDataset, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    dataset = create_sequences(scaled, lookback)
    cutoff = int(len(dataset.features) * split_ratio)
    train = SequenceDataset(
        features=dataset.features[:cutoff], targets=dataset.targets[:cutoff]
    )
    val = SequenceDataset(
        features=dataset.features[cutoff:], targets=dataset.targets[cutoff:]
    )
    return train, val, scaler
