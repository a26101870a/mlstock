import SQLSentence
import mysql
import datetime
import pandas as pd
import csv

# Constants
TABLE_STOCK_CODE = 'stock.stock_code'
TABLE_STOCK_TYPE = 'stock.stock_type'
TABLE_STOCK_PRICE = 'stock.stock_price'
TABLE_STOCK_CODE_TYPE = 'stock.stock_code_type'
PATH_CHECKPOINT = 'checkpoint'

THRESHOLD_VOLUME = 1000

# Functions
def GetStockCodeInformation(connection):
    df = SQLSentence.QuerySQL(TABLE_STOCK_CODE, connection)
    return df['code'].tolist(), df.set_index('code')['name'].to_dict()

def GetAllData(connection):
    df = SQLSentence.QuerySQL(TABLE_STOCK_PRICE, connection)
    df = df.drop('id', axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['volume'] >= THRESHOLD_VOLUME)]
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

def update_model_record_txt(record, file_path='model_records.csv'):

    record['target_pct'] = f"{int(record['target_pct'] * 100)}%"
    updated = False
    updated_rows = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row['type_id']) == record['type_id'] and 
                row['target_pct'] == record['target_pct'] and 
                row['model_name'] == record['model_name']):

                row.update({
                    'test_loss': record['test_loss'],
                    'test_acc': f"{record['test_acc']}%",
                    'test_fp': f"{record['test_fp']}%",
                    'test_fn': f"{record['test_fn']}%",
                    'train_loss': record['train_loss'],
                    'train_acc': f"{record['train_acc']}%",
                    'train_fp': f"{record['train_fp']}%",
                    'train_fn': f"{record['train_fn']}%"
                })
                updated = True
            updated_rows.append(row)

    # 若無符合條件的行，將新記錄加入
    if not updated:
        updated_rows.append({
            'type_id': record['type_id'],
            'target_pct': record['target_pct'],
            'model_name': record['model_name'],
            'test_loss': record['test_loss'],
            'test_acc': f"{record['test_acc']}%",
            'test_fp': f"{record['test_fp']}%",
            'test_fn': f"{record['test_fn']}%",
            'train_loss': record['train_loss'],
            'train_acc': f"{record['train_acc']}%",
            'train_fp': f"{record['train_fp']}%",
            'train_fn': f"{record['train_fn']}%"
        })

    # 將更新後的資料寫回檔案
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['type_id', 'target_pct', 'model_name', 'test_loss', 'test_acc', 
                      'test_fp', 'test_fn', 'train_loss', 'train_acc', 'train_fp', 'train_fn']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

def load_best_model_record_txt(type_id, target_pct, model_name, file_path='model_records.csv'):
    best_loss = float('inf')
    target_pct = str(int(target_pct*100))+"%"

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row['type_id']) == type_id and 
                row['target_pct'] == target_pct and 
                row['model_name'] == model_name):
                    if float(row['test_loss']) < best_loss:
                        best_loss = float(row['test_loss'][:-1])
            return best_loss

def fetch_data_from_db():
    connect = connect_to_database()
    main_data = GetAllData(connect)
    
    list_stock_type = (SQLSentence.QuerySQL(TABLE_STOCK_CODE_TYPE, connect, ['distinct stock_type_id']))['stock_type_id'].tolist()
    df_code_name = SQLSentence.QuerySQL(TABLE_STOCK_CODE, connect)
    
    dict_code_name = dict(zip(df_code_name.code, df_code_name.name))
    dict_type_name = ((SQLSentence.QuerySQL(TABLE_STOCK_TYPE, connect)).drop(['id'], axis=1)).to_dict()['name']

    return main_data, list_stock_type, dict_code_name, dict_type_name

def Init():
    with connect_to_database() as connection:
    
        if connection.is_connected():
            print(f"Connected to MySQL database {(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")

        stock_code_list, code_name_dict = GetStockCodeInformation(connection)
        df = GetAllData(connection)

        print(f"Close database {(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")

        return stock_code_list, code_name_dict, df
