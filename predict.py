import datetime
print(datetime.datetime.now())

import os
import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
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
main_data, list_stock_type, dict_code_name, dict_type_name = Utility.fetch_data_from_db()

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

# Some Functions

# Dataset
class StockDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.shape = data.shape

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)

        return data, label
    
    def __len__(self):
        return len(self.data)
    
# Invoke Model
def invoke_model(mode, model, dataloader, criterion, device, optimizer=None):
    
    model.train() if mode == 'train' else model.eval()

    total_loss = 0
    correct, accuracy_count = 0, 0
    investable_count, not_investable_count = 0, 0
    actual_investable_count, actual_not_investable_count = 0, 0
    type1_correct, type2_correct = 0, 0
    type1_count, type2_count = 0, 0

    for input, label in tqdm.tqdm(dataloader):
        input, label = input.to(device), label.to(device)

        if mode == 'train' and optimizer is not None:
            optimizer.zero_grad()
        
        outputs = model(input)
        loss = criterion(outputs.squeeze(1), label)
        
        total_loss+=loss.detach().cpu().item()
        
        predicted = (outputs.squeeze(1) > 0).float()

        if mode == 'train' and optimizer is not None:
            loss.backward()
            optimizer.step()

        correct += (predicted == label).sum().item()
        accuracy_count += label.size(0)

        investable_count += (predicted == 1).sum().item()
        not_investable_count += (predicted == 0).sum().item()

        actual_investable_count += (label == 1).sum().item()
        actual_not_investable_count += (label == 0).sum().item()

        t1_correct, t2_correct, t1_cnt, t2_cnt = calculate_correct_count(predicted, label)
        type1_correct += t1_correct
        type2_correct += t2_correct
        type1_count += t1_cnt
        type2_count += t2_cnt

    print(f'\n預測可投資的次數: {investable_count} / 實際上預測可投資的次數: {actual_investable_count}')
    print(f'不可投資的次數: {not_investable_count} / 實際上不可投資的次數: {actual_not_investable_count}')

    avg_loss = round(total_loss/len(dataloader), 6)
    accuracy =  round((correct/accuracy_count)*100 , 2)
    type1_correct_ratio = round((type1_correct/type1_count)*100, 2)
    type2_correct_ratio = round((type2_correct/type2_count)*100, 2)

    print(f'\n{mode} Loss: {avg_loss}')
    print(f"Accuracy: {accuracy}%")
    print(f"\nType1 Correct Ratio: {type1_correct_ratio}%")
    print(f"Type2 Correct Ratio: {type2_correct_ratio}%\n")

    return avg_loss, type1_correct_ratio, type2_correct_ratio

def process_stock_data(stock_data, target, window_length, predict = False):
    stock_data = stock_data.drop(['code', 'date'], axis=1).reset_index(drop=True)
    stock_data = Set.Feature(stock_data)
    stock_data = stock_data.dropna()

    index_col_open = stock_data.columns.get_loc("open")
    index_col_high = stock_data.columns.get_loc("high")
    index_col_low = stock_data.columns.get_loc("low")
    index_col_volume = stock_data.columns.get_loc("volume")

    stock_data = stock_data.to_numpy()

    data_merged = []
    label_merged = []

    if predict:
        return stock_data[-window_length:, index_col_volume + 1:]

    for i in range(len(stock_data) - window_length):
        data = stock_data[i:i + window_length, index_col_volume + 1:]

        if target >= 0:
            target_value = (stock_data[i + window_length, index_col_high] - stock_data[i + window_length, index_col_open]) / \
                stock_data[i + window_length, index_col_open]
            label = int(target_value >= target)
        else:
            target_value = (stock_data[i + window_length, index_col_low] - stock_data[i + window_length, index_col_open]) / \
                stock_data[i + window_length, index_col_open]
            label = int(target_value <= target)

        data_merged.append(data)
        label_merged.append(label)

    return np.array(data_merged), np.array(label_merged)

def build_dataset(main_data, code_list, target, window_length):
    all_train_data = []
    all_test_data = []

    all_train_label = []
    all_test_label = []

    for code in code_list:
        stock_data = main_data[main_data['code'] == code].copy()

        data, label = process_stock_data(stock_data, target, window_length)
        train_size = int(0.9 * len(data))
               
        train_data = data[:train_size]
        test_data = data[train_size:]

        train_label = label[:train_size]
        test_label = label[train_size:]

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        all_train_label.extend(train_label)
        all_test_label.extend(test_label)
    
    return np.array(all_train_data), np.array(all_test_data), np.array(all_train_label), np.array(all_test_label)

def build_dataloader(main_data, code_list, target, batch_size, window_length):
    train_data, test_data, train_label, test_label = build_dataset(main_data, code_list, target, window_length)
    dataset_train = StockDataset(train_data, train_label)
    dataset_test = StockDataset(test_data, test_label)
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
    data = process_stock_data(stock_data, target, window_length, predict=True)
    output = model(torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device))
    return round(output[0].cpu().detach().item(), 4)

def GetDatasetShape(main_data, window_length, code=1101, min_period=100):
    stock_data = main_data[main_data['code'] == code].tail(min_period).copy()
    stock_data = stock_data.drop(['code', 'date'], axis=1).reset_index(drop=True)
    stock_data = Set.Feature(stock_data)
    stock_data = stock_data.dropna()

    index_col_volume = stock_data.columns.get_loc("volume")

    stock_data = stock_data.to_numpy()

    return np.expand_dims(stock_data[-window_length:, index_col_volume + 1:], axis=0).shape

def adjust_date(date):
    # 判斷日期是星期幾 (星期一是0，星期日是6)
    weekday = date.weekday()

    # 如果不是星期五、六、日，將日期加一天
    if weekday not in [4, 5, 6]:
        return date + datetime.timedelta(days=1)
    # 星期五、六、日，將日期調整至下周一
    else:
        return date + datetime.timedelta(days=(7 - weekday))
    
print('Function OK')

print('Sarting Predict')

# Predict
list_target_pct = [0.02, 0.03, 0.05, -0.02, -0.03, -0.05]
window_length = 20
threshold=0.5

# [0, 7, 8, 13, 17, 18, 24, 25, 26, 27, 30, 31, 32, 35, 37]
# list_selected_type_id = Set.GetInfo('select_stock_list')
list_selected_type_id = [0, 7, 8, 13, 17, 18, 24, 25, 26, 30, 31, 32, 35, 37]
current_date = adjust_date((SQLSentence.GetLatestDate(connect))).strftime("%Y-%m-%d")

model_name = 'CNN_LSTM'
ModelClass = getattr(Model, model_name)
postfix='best'

directory_path = f'prediction/{current_date}'
os.mkdir(directory_path) 

for target_pct in list_target_pct:

    file_name = f'{model_name}_{postfix}_{int(target_pct*100)}%.txt'
    file_path = os.path.join(directory_path, file_name) 

    with open(file_path, 'w') as f:
        pass

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"------------------------------Model: {model_name}_{postfix}")

        for type_id in tqdm.tqdm(list_selected_type_id):

            list_unique_code_from_data = (SQLSentence.GetCodeByTypeId(type_id, connect))['code'].tolist()

            model = ModelClass(GetDatasetShape(main_data, window_length)).to(device)
            model.eval()
            list_code_investable = []

            try:
                model.load_state_dict(torch.load(f"{PATH_CHECKPOINT}/type_{type_id}/type_{type_id}_{model.__class__.__name__}_Target_{int(target_pct*100)}_{postfix}", weights_only=True))
            except:
                print(f'{dict_type_name[type_id]} No preserved model: {model.__class__.__name__}')
                continue

            for code in list_unique_code_from_data:
                output = predict_result(main_data, code, model, device, target_pct, window_length)
                if output > threshold:
                    list_code_investable.append((code, output))

            if len(list_code_investable) != 0:
                f.write("------------------------------")
                f.write(f"\n產業: {dict_type_name[type_id]}\n")
                for code, probability in list_code_investable:
                    f.write(f'{code} {dict_code_name[code]:<10} Prob: {probability}\n')
        f.write("\n\n")

print('Complete Prediction')

print('Calculate Tier')

dict_code_pred = {}

for target_pct in list_target_pct:

    file_name = f'{model_name}_{postfix}_{int(target_pct*100)}%.txt'
    file_path = os.path.join(directory_path, file_name) 

    with open(file_path, 'r', encoding='utf-8') as file:

        dict_code_pred[target_pct] = []

        for line in file:
            if 'Prob' in line:
                dict_code_pred[target_pct].append(int(line[:4]))

print('Build Bull Tier List')

file_name = f'bull_tier.txt'
file_path = os.path.join(directory_path, file_name)

with open(file_path, 'w', encoding='utf-8') as f:

    max_value = max(list_target_pct)
    second_max_value = max([value for value in list_target_pct if value != max_value])
    third_max_value = max([value for value in list_target_pct if (value != max_value) and (value != second_max_value) ])

    tier1 = list(set(dict_code_pred[max_value]) & set(dict_code_pred[second_max_value]) & set(dict_code_pred[third_max_value]))
    tier2 = list(set(dict_code_pred[second_max_value]) & set(dict_code_pred[third_max_value]))

    f.write(f'Tier 1: {tier1}\n')
    f.write(f'Tier 2: {tier2}\n\n')

    for value in [max_value, second_max_value, third_max_value]:
        f.write(f'{int(value*100)}% Code List: {list(dict_code_pred[value])}\n')
        f.write(f'{int(value*100)}% Length: {len(list(dict_code_pred[value]))}\n')

print('Build Bear Tier List')

file_name = f'bear_tier.txt'
file_path = os.path.join(directory_path, file_name)

with open(file_path, 'w', encoding='utf-8') as f:

    min_value = min(list_target_pct)
    second_min_value = min([value for value in list_target_pct if value != min_value])
    third_min_value = min([value for value in list_target_pct if (value != min_value) and (value != second_min_value) ])

    tier1 = list(set(dict_code_pred[min_value]) & set(dict_code_pred[second_min_value]) & set(dict_code_pred[third_min_value]))
    tier2 = list(set(dict_code_pred[second_min_value]) & set(dict_code_pred[third_min_value]))

    f.write(f'Tier 1: {tier1}\n')
    f.write(f'Tier 2: {tier2}\n\n')

    for value in [min_value, second_min_value, third_min_value]:
        f.write(f'{int(value*100)}% Code List: {list(dict_code_pred[value])}\n')
        f.write(f'{int(value*100)}% Length: {len(list(dict_code_pred[value]))}\n')

print(datetime.datetime.now())