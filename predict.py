import os
import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import Set
import Utility
import SQLSentence
import Model

print('Package OK')

# Constants
TABLE_CODE = 'stock.stock_code'
TABLE_TYPE = 'stock.stock_type'
TABLE_CODE_TYPE = 'stock.stock_code_type'

PATH_CHECKPOINT = 'checkpoint'

print('Constant OK')

# Build Basic Data
connect = Utility.connect_to_database()
main_data = Utility.GetAllData(connect)
list_stock_type = (SQLSentence.QuerySQL(TABLE_CODE_TYPE, connect, ['distinct stock_type_id']))['stock_type_id'].tolist()
df_code_name = SQLSentence.QuerySQL(TABLE_CODE, connect)
dict_code_name = dict(zip(df_code_name.code, df_code_name.name))
dict_type_name = ((SQLSentence.QuerySQL(TABLE_TYPE, connect)).drop(['id'], axis=1)).to_dict()['name']

del df_code_name

print('Build Basic Data OK')

# Build Environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(PATH_CHECKPOINT):
    os.makedirs(PATH_CHECKPOINT)

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

print('Build Environment OK')

# Dataset
class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.dim = len(self.data[0])-1

    def __getitem__(self, index):
        data = torch.tensor(self.data[index][:-1], dtype=torch.float32)
        label = torch.tensor(self.data[index][-1], dtype=torch.float32)
        return data, label
    
    def __len__(self):
        return len(self.data)

print('Dataset OK')

# Some Functions
def process_stock_data(stock_data, target, window_length, predict = False):
    stock_data = stock_data.drop(['code', 'date'], axis=1).reset_index(drop=True)
    stock_data = Set.Feature(stock_data)
    stock_data = stock_data.dropna()

    index_col_open = stock_data.columns.get_loc("open")
    index_col_high = stock_data.columns.get_loc("high")
    index_col_volume = stock_data.columns.get_loc("volume")

    stock_data = stock_data.to_numpy()

    data_merged = []

    if predict:
        return stock_data[-window_length:, index_col_volume + 1:].flatten()

    for i in range(len(stock_data) - window_length):
        window_data = stock_data[i:i + window_length, index_col_volume + 1:].flatten()
        target_value = (stock_data[i + window_length, index_col_high] - stock_data[i + window_length, index_col_open]) / stock_data[i + window_length, index_col_open]

        if target >= 0:
            label = int(target_value >= target)
        else:
            label = int(target_value <= target)
        
        window_data_with_label = np.append(window_data, label)

        data_merged.append(window_data_with_label)

    return data_merged

def build_dataset(main_data, code_list, target, window_length):
    all_train_data = []
    all_test_data = []

    for code in code_list:
        stock_data = main_data[main_data['code'] == code].copy()
        train_size = int(0.9 * len(stock_data))

        process_data = process_stock_data(stock_data, target, window_length)
               
        train_data = process_data[:train_size]
        test_data = process_data[train_size:]

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
    
    return all_train_data, all_test_data

def build_dataloader(main_data, code_list, target, batch_size, window_length):
    train_data, test_data = build_dataset(main_data, code_list, target, window_length)
    dataset_train = StockDataset(train_data)
    dataset_test = StockDataset(test_data)
    datalaoder_train = DataLoader(dataset_train, batch_size, shuffle=True)
    datalaoder_test = DataLoader(dataset_test, batch_size, shuffle=False)

    return datalaoder_train, datalaoder_test

def calculate_correct_count(predicted, label):
    type1_correct = 0
    type2_correct = 0
    type1_count = 0
    type2_count = 0

    for i in range(label.size(0)):
        # Type1
        if label[i] == 1:
            type1_count += 1
            if predicted[i] == 1:
                type1_correct += 1

        # Type2
        elif label[i] == 0:
            type2_count += 1
            if predicted[i] == 0:
                type2_correct += 1

    return type1_correct, type2_correct, type1_count, type2_count

def predict_result(main_data, code, model, device, target, window_length, min_period=100):
    stock_data = main_data[main_data['code'] == code].tail(min_period).copy()
    process_data = process_stock_data(stock_data, target, window_length, predict=True)
    output = model(torch.tensor(process_data, dtype=torch.float32).reshape(1, process_data.shape[0]).to(device))
    return round(output[0].cpu().detach().item(), 4)

print('Function OK')

# Predict
target_pct = 0.02
window_length = 20
threshold=0.5

postfix='best'
list_selected_type_id = [0, 4, 7, 8, 13, 17, 18, 24, 25, 26, 27, 30, 31, 32, 35, 37]
current_date = (SQLSentence.GetLatestDate(connect)).strftime("%Y-%m-%d")
file_name = f'prediction/predict_{current_date}.txt'

if not os.path.exists(file_name):
    with open(file_name, 'w') as f:
        pass

print('Sarting Predict OK')
with open(file_name, 'a', encoding='utf-8') as f:
    for type_id in tqdm.tqdm(list_selected_type_id):

        list_unique_code_from_data = (SQLSentence.GetCodeByTypeId(type_id, connect))['code'].tolist()

        model = Model.Model_CNN_LSTM(720).to(device)
        model.eval()
        list_code_investable = []

        try:
            model.load_state_dict(torch.load(f"{PATH_CHECKPOINT}/type_{type_id}/type_{type_id}_{model.__class__.__name__}_Target_{int(target_pct*100)}_{postfix}", weights_only=True))
        except:
            print(f'{dict_type_name[type_id]} No preserved model: {model.__class__.__name__}')

        for code in list_unique_code_from_data:
            output = predict_result(main_data, code, model, device, target_pct, window_length)
            if output > threshold:
                list_code_investable.append((code, output))

        if len(list_code_investable) != 0:
            f.write("------------------------------------------------------------------------")
            f.write(f"\n產業: {dict_type_name[type_id]}\n")
            for code, probability in list_code_investable:
                f.write(f'{code} {dict_code_name[code]:<10} Prob: {probability}\n')