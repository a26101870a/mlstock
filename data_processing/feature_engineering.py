from . import Indicators

def Feature(df):
    df = Indicators.CalculatePercentChange(df)
    df = Indicators.RSIFlag(df)
    return df