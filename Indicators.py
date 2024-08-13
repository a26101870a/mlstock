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

def GenerateCCI(df, period=20):
    CONSTANT = (1/0.015)
    df['typical_price'] = (df['high'] + df['low'] + df['close'])/3
    df['CCI'] = CONSTANT*(df['typical_price'] - df['typical_price'].rolling(window=period).mean())/\
                (abs((df['typical_price'] - df['typical_price'].rolling(window=period).mean()))).rolling(window=period).mean()
    return df.drop(columns=['typical_price'])

def GenerateKST(df):
    df['ROC1'] = 100*(df['close']/df['close'].shift(10)-1)
    df['ROC2'] = 100*(df['close']/df['close'].shift(15)-1)
    df['ROC3'] = 100*(df['close']/df['close'].shift(20)-1)
    df['ROC4'] = 100*(df['close']/df['close'].shift(30)-1)

    df['KST'] = 1*df['ROC1'].rolling(window=10).mean() + \
                2*df['ROC1'].rolling(window=10).mean() + \
                3*df['ROC1'].rolling(window=10).mean() + \
                4*df['ROC1'].rolling(window=15).mean()

    return df.drop(columns=['ROC1', 'ROC2', 'ROC3', 'ROC4'])

def GenerateIchimoku(df):
    df['Tenkan_Sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['Chinkou_Span'] = df['close'].shift(-26)
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['Kumo_top'] = df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
    df['Kumo_bottom'] = df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)
    return df

def GenerateMassIndex(df):
    df['DailyRange'] = df['high'] - df['low']
    df['EMA_9_Range'] = df['DailyRange'].ewm(span=9, adjust=False).mean()
    df['EMA_Ratio'] = df['EMA_9_Range'] / df['EMA_9_Range'].ewm(span=9, adjust=False).mean()
    df['MassIndex'] = df['EMA_Ratio'].rolling(window=25).sum()
    return df.drop(columns=['DailyRange', 'EMA_9_Range','EMA_Ratio'])

def GenerateSARInverse(df, initial_af=0.02, step_af=0.02, max_af=0.20):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    sar = close.copy()
    trend = np.ones(len(df), dtype=int) # Start with an uptrend (1 for up, -1 for down)
    ep = high.copy()
    af = np.full(len(df), initial_af)
    inverse = np.zeros(len(df), dtype=int)
    
    for i in range(1, len(df)):
        if trend[i-1] == 1:  # Uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if low[i] < sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = initial_af
                inverse[i] = -1
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(max_af, af[i-1] + step_af)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        elif trend[i-1] == -1:  # Downtrend
            sar[i] = sar[i-1] - af[i-1] * (sar[i-1] - ep[i-1])
            if high[i] > sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = initial_af
                inverse[i] = 1
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(max_af, af[i-1] + step_af)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    df['trend'] = trend.astype('int8')
    df['Inverse'] = inverse.astype('int8')

    return df

def GenerateTRIX(df, period=30, price_type='close'):
    ema1 = df[price_type].ewm(span=period, adjust=False).mean()
    
    # Calculate the second EMA (EMA of the first EMA)
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    
    # Calculate the third EMA (EMA of the second EMA)
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # Calculate the Rate of Change of the Triple EMA
    df['TRIX'] = ema3.pct_change() * 100

    return df

def GenerateVortex(df, period=21):
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - df['prev_close']),
        np.abs(df['low'] - df['prev_close'])
    )

    df['VM+'] = np.abs(df['high'] - df['prev_low'])
    df['VM-'] = np.abs(df['low'] - df['prev_high'])

    df['sum_TR'] = df['TR'].rolling(window=period).sum()
    df['sum_VM+'] = df['VM+'].rolling(window=period).sum()
    df['sum_VM-'] = df['VM-'].rolling(window=period).sum()

    df['VI+'] = df['sum_VM+'] / df['sum_TR']
    df['VI-'] = df['sum_VM-'] / df['sum_TR']

    return df.drop(columns=['prev_high', 'prev_low', 'prev_close', 'TR', 'VM+', 'VM-', 'sum_VM+', 'sum_VM-', 'sum_TR'])

def GenerateMoneyFlowIndex(df, period=14):
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['Raw Money Flow'] = df['TP'] * df['volume']
    df['prev_TP'] = df['TP'].shift()

    df['Positive Money Flow'] = np.where(df['TP'] > df['prev_TP'], df['Raw Money Flow'], 0)
    df['Negative Money Flow'] = np.where(df['TP'] < df['prev_TP'], df['Raw Money Flow'], 0)

    df['Positive MF Sum'] = df['Positive Money Flow'].rolling(window=period).sum()
    df['Negative MF Sum'] = df['Negative Money Flow'].rolling(window=period).sum()
    df['MFR'] = df['Positive MF Sum'] / df['Negative MF Sum']

    df['MFI'] = (100 - (100 / (1 + df['MFR']))).astype('float32')

    return df.drop(columns=['TP', 'Raw Money Flow', 'prev_TP', 'Positive Money Flow', 'Negative Money Flow', 
                     'Positive MF Sum', 'Negative MF Sum', 'MFR'])

def GenerateKDJ(df, period=9):
    low_list=df['low'].rolling(window=period).min()
    # low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = df['high'].rolling(window=9).max()
    # high_list.fillna(value=df['high'].expanding().max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['KDJ_K'] = rsv.ewm(com=2).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    
    return df

def GenerateTSI(df, long_period=25, short_period=13, price_type='close'):
    df['PC'] = df[price_type] - df[price_type].shift()

    df['EMA1_PC'] = df['PC'].ewm(span=short_period, adjust=False).mean()
    df['EMA1_AbsPC'] = df['PC'].abs().ewm(span=short_period, adjust=False).mean()

    df['EMA2_PC'] = df['EMA1_PC'].ewm(span=long_period, adjust=False).mean()
    df['EMA2_AbsPC'] = df['EMA1_AbsPC'].ewm(span=long_period, adjust=False).mean()

    df['TSI'] = 100 * (df['EMA2_PC'] / df['EMA2_AbsPC'])

    return df.drop(columns=['PC', 'EMA1_PC', 'EMA1_AbsPC', 'EMA2_PC', 'EMA2_AbsPC'])

def GenerateUltimateOscillator(df, short_period=7, medium_period=14, long_period=28):
    df['prev_close'] = df['close'].shift()
    df['BP'] = df['close'] - np.minimum(df['low'], df['prev_close'])

    df['TR'] = np.maximum(
        np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['prev_close'])),
        np.abs(df['low'] - df['prev_close'])
    )

    df['AVG1'] = df['BP'].rolling(window=short_period).sum() / df['TR'].rolling(window=short_period).sum()
    df['AVG2'] = df['BP'].rolling(window=medium_period).sum() / df['TR'].rolling(window=medium_period).sum()
    df['AVG3'] = df['BP'].rolling(window=long_period).sum() / df['TR'].rolling(window=long_period).sum()

    df['UO'] = 100 * (
        (4 * df['AVG1'] + 2 * df['AVG2'] + df['AVG3']) /
        (4 + 2 + 1)
    )
    
    return df.drop(columns=['prev_close', 'BP', 'TR', 'AVG1', 'AVG2', 'AVG3'])

def GenerateWilliamsR(df, period=10):
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()

    df['Williams%R'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
    
    return df

def GenerateADL(df):
    df['MFM'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['MFM'] = df['MFM'].fillna(0)
    df['MFV'] = df['MFM'] * df['volume']
    df['ADL'] = df['MFV'].cumsum()

    return df.drop(columns=['MFM', 'MFV'])

def GenerateEMV(df):
    df['Midpoint'] = (df['high'] + df['low']) / 2
    df['Prev Midpoint'] = df['Midpoint'].shift(1)
    df['Midpoint Move'] = df['Midpoint'] - df['Prev Midpoint']
    df['Box Ratio'] = df['volume'] / (df['high'] - df['low'])
    df['EMV'] = df['Midpoint Move'] / df['Box Ratio']

    return df.drop(columns=['Midpoint', 'Prev Midpoint', 'Midpoint Move', 'Box Ratio'])

def GenerateForceIndex(df):
    df['prev_close'] = df['close'].shift(1)
    df['ForceIndex'] = (df['close'] - df['prev_close']) * df['volume']

    return df.drop(columns=['prev_close'])

def GenerateNVI(df, initial_value=1000):
    df['NVI'] = initial_value
    for i in range(1, len(df)):
        if df['volume'].iloc[i] < df['volume'].iloc[i - 1]:
            df['NVI'].iloc[i] = df['NVI'].iloc[i - 1] + (df['close'].iloc[i] - df['close'].iloc[i - 1])
        else:
            df['NVI'].iloc[i] = df['NVI'].iloc[i - 1]
    
    return df

def GenerateVPT(df):
    df['prev_close'] = df['close'].shift()
    df['price_change'] = (df['close'] - df['prev_close'])/df['prev_close']
    df['VPT'] = (df['volume'] * df['price_change']).cumsum()

    return df.drop(columns=['prev_close', 'Prev price_change'])

def GenerateATR(df, period=14):
    df['prev_close'] = df['close'].shift()
    
    df['TR'] = np.maximum(
        np.maximum(df['high'] - df['low'],
                   np.abs(df['high'] - df['prev_close'])),
        np.abs(df['low'] - df['prev_close'])
    )
    
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    return df.drop(columns=['TR', 'Prev ATR'])

def GenerateDonchianChannel(df, period=20):
    df['DonchianUpper'] = df['high'].rolling(window=period).max()
    df['DonchianLower'] = df['low'].rolling(window=period).min()
    df['DonchianMiddle'] = (df['DonchianUpper'] + df['DonchianLower']) / 2

    return df

def GenerateKeltnerChannel(df, period=20, multiplier=2):
    df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
    df['MiddleLine'] = df['TypicalPrice'].ewm(span=period, adjust=False).mean()
    df['TR'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(np.abs(df['high'] - df['close'].shift()), 
                                     np.abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=period).mean()

    df['UpperBand'] = df['MiddleLine'] + (multiplier * df['ATR'])
    df['LowerBand'] = df['MiddleLine'] - (multiplier * df['ATR'])

    return df.drop(columns=['TypicalPrice', 'TR', 'ATR'])