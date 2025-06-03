from . import Indicators
import pandas as pd

def Zscore_Normalize(df: pd.DataFrame, list_period: list[int] = [5, 10, 20], list_exclude_columns: list[str] =None):
    if list_exclude_columns is None:
        list_exclude_columns = []

    float_cols = df.select_dtypes(include='float').columns
    target_cols = [col for col in float_cols if col not in list_exclude_columns]

    for column in target_cols:
        for period in list_period:
            df = Indicators.GenerateZScore(df, period, column)

    return df

def SetFeature(df):
    df = Indicators.CalculatePercentChange(df)
    df = Indicators.RSIFlag(df)
    df = Zscore_Normalize(df)
    return df