import numpy as np
import pandas as pd

def GenerateRSI(df: pd.DataFrame, period=14, price_type='close') -> pd.DataFrame:
    df['prev_price'] = df[price_type].shift()
    df['is_rised'] = df[price_type] > df['prev_price']
    df['u'] = np.where(df['is_rised'], df[price_type] - df['prev_price'], 0).astype(float)
    df['d'] = np.where(~df['is_rised'], df['prev_price'] - df[price_type], 0).astype(float)
    df['ema_u'] = df['u'].ewm(span=period, adjust=False).mean()
    df['ema_d'] = df['d'].ewm(span=period, adjust=False).mean()
    df[f'rsi_{price_type}'] = round((df['ema_u'] / (df['ema_u'] + df['ema_d'])) * 100, 2)
    return df.drop(columns=['prev_price', 'is_rised', 'u', 'd', 'ema_u', 'ema_d'])

def GenerateBollingBand(df: pd.DataFrame, period=20, price_type='close') -> pd.DataFrame:
    df[f'bb_ma_{price_type}'] = df[price_type].rolling(window=period).mean()
    df['std'] = df[price_type].rolling(window=period).std()
    df[f'bb_ub_{price_type}'] = df[f'bb_ma_{price_type}'] + 2*df['std']
    df[f'bb_lb_{price_type}'] = df[f'bb_ma_{price_type}'] - 2*df['std']
    return df.drop(columns=['std'])

def GenerateMACD(df, short_period=12, long_period=26, macd_period=9, price_type='close') -> pd.DataFrame:
    df['ema_short'] = df[price_type].ewm(span=short_period, adjust=False).mean()
    df['ema_long'] = df[price_type].ewm(span=long_period, adjust=False).mean()
    df['dif'] = df['ema_short'] - df['ema_long']
    df['dem'] = df['dif'].ewm(span=macd_period, adjust=False).mean()
    df[f'macd_{price_type}'] = df['dif'] - df['dem']
    return df.drop(columns=['ema_short', 'ema_long', 'dif', 'dem'])

def GenerateMovingAverage(df, period, price_type='close') -> pd.DataFrame:
    df[f'MA{period}_{price_type}'] = df[price_type].rolling(window=period).mean()
    return df.drop(columns=['ema_short', 'ema_long', 'dif', 'dem'])

def GenerateZScore(df, period, price_type='close') -> pd.DataFrame:
    df['MA'] = df[price_type].rolling(window=period).mean()
    df['std_MA'] = df[price_type].rolling(window=period).std()
    df[f'Z-Score_MA{period}_{price_type}'] = (df[price_type] - df['MA'])/df['std_MA']
    return df.drop(columns=['MA', 'std_MA'])

def GenerateADX(df, period=14) -> pd.DataFrame:
    df['prev_high'] = df['high'].shift()
    df['prev_low'] = df['low'].shift()
    df['prev_close'] = df['close'].shift()
    df['UpMove'] = df['high'] - df['prev_high']
    df['DownMove'] = df['prev_low'] - df['low']
    df['TR'] = np.maximum(df['high'], df['prev_close']) - np.minimum(df['low'], df['prev_close'])
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['+DM'] = np.where((df['UpMove']>df['DownMove']) & (df['UpMove']>0), 
                               df['UpMove'], 0).astype(float)
    df['-DM'] = np.where((df['DownMove']>df['UpMove']) & (df['DownMove']>0), 
                            df['DownMove'], 0).astype(float)
    df['+DI'] = 100*(df['+DM'].rolling(window=period).mean()/df['ATR'])
    df['-DI'] = 100*(df['-DM'].rolling(window=period).mean()/df['ATR'])
    df['DX'] = 100*(abs(df['+DI'] - df['-DI'])/abs(df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    return df.drop(columns=['prev_high', 'prev_low', 'prev_close', 'UpMove', 'DownMove',
                            '+DM', '-DM', 'TR', 'ATR', '+DI', '-DI', 'DX'])
