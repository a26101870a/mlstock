import SQLSentence
import mysql
import datetime
import pandas as pd

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
    df = df[df['volume'] >= THRESHOLD_VOLUME]
    df[df.columns[df.columns.get_loc("open"):df.columns.get_loc("volume")]] = df[df.columns[2:-1]].astype(float)
    
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

def write_train_detail_to_txt(type_id, model_name, test_loss, file_path='model_records.txt'):
    with open(file_path, 'a') as f:
        f.write(f"{type_id},{model_name},{test_loss}\n")

def load_best_model_record_txt(type_id, model_name, file_path='model_records.txt'):
    best_loss = float('inf')
    with open(file_path, 'r') as f:
        for line in f:
            tid, name, loss = line.strip().split(',')
            if int(tid) == type_id and name == model_name:
                if float(loss) < best_loss:
                    best_loss = float(loss)
    return best_loss

def update_model_record_txt(type_id, model_name, new_test_loss, file_path='model_records.txt'):
    updated = False
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            tid, name, _ = line.strip().split(',')
            if int(tid) == type_id and name == model_name:
                line = f"{type_id},{model_name},{new_test_loss}\n"
                updated = True
            lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    if not updated:
        write_train_detail_to_txt(type_id, model_name, new_test_loss, file_path)