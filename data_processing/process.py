from .feature_engineering import SetFeature

import numpy as np
import pandas as pd

EXCLUDE_COLUMNS = {"code", "date", "open", "high", "low", "close", "adjclose", "volume"}

def calculate_label(series: pd.Series, target_percent: float) -> bool:
    if target_percent >= 0:
        return series["high"] >= series["open"] * (1 + target_percent)
    else:
        return series["low"] <= series["open"] * (1 - target_percent)

def process_data(df: pd.DataFrame, config: dict = None) -> tuple[np.ndarray, np.ndarray | None]:

    target_percent = config["target_percent"]
    window_length = config["window_length"]

    df = df.reset_index(drop=True)
    df = SetFeature(df)
    df = df.dropna()
    
    feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]

    if config['predict']:
        data = df[feature_columns].to_numpy()[-config['window_length']:]
        return np.expand_dims(data, axis=0), None
    
    x, y = [], []
    for i in range(len(df) - window_length):
        data = df.iloc[i:i + window_length][feature_columns].to_numpy()
        label = calculate_label(df.iloc[i + window_length], target_percent)

        x.append(data)
        y.append(label)

    return np.array(x), np.array(y, dtype=bool)