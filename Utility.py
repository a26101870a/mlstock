import SQLSentence
import mysql
import datetime
import pandas as pd
import torch

# Constants
TABLE_STOCK_CODE = 'stock.stock_code'
TABLE_STOCK_PRICE = 'stock.stock_price'

THRESHOLD_VOLUME = 1000

# Functions
def GetStockCodeInformation(connection):
    df = SQLSentence.QuerySQL(TABLE_STOCK_CODE, connection)
    return df['code'].tolist(), df.set_index('code')['name'].to_dict()

def GetAllData(connection):
    df = SQLSentence.QuerySQL(TABLE_STOCK_PRICE, connection)
    df = df.drop('id', axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['volume'] != 0]
    
    return df

def GetMySQLLoggingInformation():
    res={}
    with open('mysql_logging_information.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(':')
            res[key.strip()] = value.strip()
    return res

def connect_to_database():
    information = GetMySQLLoggingInformation()
    return mysql.connector.connect(
        host=information['host'],
        port=information['port'],
        user=information['user'],
        password=information['password'],
        database="stock"
    )
    
def Init():
    with connect_to_database() as connection:
    
        if connection.is_connected():
            print(f"Connected to MySQL database {(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")

        stock_code_list, code_name_dict = GetStockCodeInformation(connection)
        df = GetAllData(connection)

        print(f"Close database {(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")

        return stock_code_list, code_name_dict, df
    
def PreprocessData(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop('id', axis=1)
    data = data[data['volume'] >= THRESHOLD_VOLUME]
    return data

def ProcessData(data: pd.DataFrame) -> torch.Tensor:
    data = data.drop(['code', 'date'], axis=1)
    data = data.apply(pd.to_numeric, downcast='float').to_numpy()
    return data