import numpy as np
import pandas as pd

def GenerateRSI(df: pd.DataFrame, period=14) -> pd.DataFrame:
    return df.assign(
        prev_close_price=df['close'].shift(),
        is_rised=lambda x: x['close'] > x['prev_close_price'],
        u=lambda x: np.where(x['is_rised'], x['close'] - x['prev_close_price'], 0).astype(float),
        d=lambda x: np.where(~x['is_rised'], x['prev_close_price'] - x['close'], 0).astype(float),
        ema_u=lambda x: x['u'].ewm(span=period, adjust=False).mean(),
        ema_d=lambda x: x['d'].ewm(span=period, adjust=False).mean(),
        rsi=lambda x: round((x['ema_u'] / (x['ema_u'] + x['ema_d'])) * 100, 2)
    ).drop(columns=['prev_close_price', 'is_rised', 'u', 'd', 'ema_u', 'ema_d'])

def GenerateBollingBand(df: pd.DataFrame, period=20) -> pd.DataFrame:
    return df.assign(
        bb_ma=lambda x: x['close'].rolling(window=period).mean(),
        std=lambda x: x['close'].rolling(window=period).std(),
        bb_ub=lambda x: x['bb_ma'] + 2*x['std'],
        bb_lb= lambda x: x['bb_ma'] - 2*x['std']
    ).drop(columns=['std'])

def GenerateMACD(df, short_period=12, long_period=26, macd_period=9) -> pd.DataFrame:
    return df.assign(
        ema_short=lambda x: x['close'].ewm(span=short_period, adjust=False).mean(),
        ema_long=lambda x: x['close'].ewm(span=long_period, adjust=False).mean(),
        dif=lambda x: x['ema_short'] - x['ema_long'],
        dem=lambda x: x['dif'].ewm(span=macd_period, adjust=False).mean(),
        macd=lambda x: x['dif'] - x['dem']
    ).drop(columns=['ema_short', 'ema_long', 'dif', 'dem'])

def GenerateMovingAverage(df, price_type, period) -> pd.DataFrame:
    df[f'MA{period}_{price_type}'] = df[price_type].rolling(window=period).mean()
    return df

def GenerateZScore(df, price_type, period) -> pd.DataFrame:
    df[f'MA{period}'] = df[price_type].rolling(window=period).mean()
    df[f'std_MA{period}'] = df[price_type].rolling(window=period).mean()
    df[f'ZScore_MA{period}_{price_type}'] = (df[price_type] - df[f'MA{period}'])/df[f'std_MA{period}']
    return df.drop(columns=[f'MA{period}', f'std_MA{period}'])