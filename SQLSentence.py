from typing import Union, List, Tuple, Optional
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
import pandas as pd
from datetime import datetime, timedelta

def InsertSQL(datatable:str, columns:List[str], data:List[Tuple], connection:Union[PooledMySQLConnection, MySQLConnectionAbstract]):
       
    sql_insert = f"""
    INSERT INTO {datatable} ({", ".join(columns)})
    VALUES ({", ".join(["%s"] * len(columns))})
    """

    try:
        with connection.cursor() as cursor:
            cursor.executemany(sql_insert, data)
            connection.commit()
            print(f"Insert Data To {datatable} Successfully")
    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()

def QuerySQL(datatable:str, connection:Union[PooledMySQLConnection, MySQLConnectionAbstract], columns:Optional[List[str]]=None) -> pd.DataFrame:

    columns = "*" if columns is None else ", ".join(columns)

    sql_query = f"SELECT {columns} FROM {datatable}"
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            records = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return pd.DataFrame(records, columns=column_names)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# 需要原因:更新資料庫需要取得最近的一個日期
def GetLatestDate(connection:Union[PooledMySQLConnection, MySQLConnectionAbstract]) -> datetime.date:
    sql_query = "SELECT date FROM stock.stock_price order by date desc limit 1;"
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            records = cursor.fetchall()
            return records[0][0]
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def GetPriceByTypeId(type_id: int, connection:Union[PooledMySQLConnection, MySQLConnectionAbstract]) -> pd.DataFrame:

    sql_query = f"""
        SELECT 
            *
        FROM
            stock.stock_price
        WHERE
            code IN (SELECT 
                    code
                FROM
                    stock.stock_code_type
                WHERE
                    stock_type_id = {type_id});
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            records = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return pd.DataFrame(records, columns=column_names)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    
def GetCodeByTypeId(type_id: int, connection:Union[PooledMySQLConnection, MySQLConnectionAbstract]) -> pd.DataFrame:

    sql_query = f"""
        SELECT 
            code
        FROM
            stock.stock_code_type
        WHERE
            stock_type_id = {type_id};
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            records = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return pd.DataFrame(records, columns=column_names)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()