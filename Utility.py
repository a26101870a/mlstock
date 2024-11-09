import SQLSentence
import mysql
import datetime
import pandas as pd

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

def write_train_detail_to_txt(type_id, target_pct, model_name, test_loss, test_acc, file_path='model_records.txt'):
    with open(file_path, 'a') as f:
        f.write(f"{type_id},{target_pct},{model_name},{test_loss},{test_acc}%\n")

def load_best_model_record_txt(type_id, target_pct, model_name, file_path='model_records.txt'):
    best_loss = float('inf')
    target_pct = str(int(target_pct*100))+"%"
    with open(file_path, 'r') as f:
        for line in f:
            tid, target, name, loss, _ = line.strip().split(',')
            if int(tid) == type_id and target_pct == target and name == model_name:
                if float(loss) < best_loss:
                    best_loss = float(loss)
    return best_loss

def update_model_record_txt(type_id, target_pct, model_name, new_test_loss, new_test_acc, file_path='model_records.txt'):
    updated = False
    lines = []

    target_pct = str(int(target_pct*100))+"%"

    with open(file_path, 'r') as f:
        for line in f:
            tid, target, name, _, _ = line.strip().split(',')
            if int(tid) == type_id and target_pct == target and name == model_name:
                line = f"{type_id},{target_pct},{model_name},{new_test_loss},{new_test_acc}%\n"
                updated = True
            lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    if not updated:
        write_train_detail_to_txt(type_id, target_pct, model_name, new_test_loss, file_path)

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
