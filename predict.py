import datetime
print(datetime.datetime.now())

import os
import csv
import tqdm
import torch
import numpy as np
from random import sample

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
PATH_MITAKE_EXPORT_DATA = 'D:/MitakeGU/USER/OUT/export_data.csv'

print('Constant OK')

# Build Basic Data
connect = Utility.connect_to_database()
main_data, _, dict_code_name, dict_type_name = Utility.fetch_data_from_db()

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

def conver_to_int_percent(percent: float):
    return f"{int(percent*100)}%"

def proccess_prediction_old(model_name, postfix, device, window_length, list_target_pct, threshold, directory_path, main_data, dict_type_name, connect):

    list_selected_type_id = Set.GetInfo('select_stock_list')
    model_class = getattr(Model, model_name)

    for target_pct in list_target_pct:

        file_name = f'{model_name}_{postfix}_{int(target_pct*100)}%.txt'
        file_path = os.path.join(directory_path, file_name) 

        with open(file_path, 'w') as f:
            pass

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"------------------------------Model: {model_name}_{postfix}")
            print(f'Processing {model_name} {target_pct}')
            for type_id in tqdm.tqdm(list_selected_type_id):

                list_unique_code_from_data = (SQLSentence.GetCodeByTypeId(type_id, connect))['code'].tolist()

                model = model_class(GetDatasetShape(main_data, window_length)).to(device)
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

def get_probability_from_prediction(model_name, postfix, list_target_pct, directory_path):
    dict_code_pred = {}

    for target_pct in list_target_pct:

        file_name = f'{model_name}_{postfix}_{int(target_pct*100)}%.txt'
        file_path = os.path.join(directory_path, file_name) 

        with open(file_path, 'r', encoding='utf-8') as file:

            dict_code_pred[target_pct] = []

            for line in file:
                if 'Prob' in line:
                    dict_code_pred[target_pct].append(int(line[:4]))

    return dict_code_pred

def build_bull_tier_list(dict_code_pred,list_target_pct, directory_path):

    file_name = f'bull_tier.txt'
    file_path = os.path.join(directory_path, file_name)

    bull_max, bull_second, bull_third = [], [], []

    with open(file_path, 'w', encoding='utf-8') as f:

        max_value = max(list_target_pct)
        second_max_value = max([value for value in list_target_pct if value != max_value])
        third_max_value = max([value for value in list_target_pct if (value != max_value) and (value != second_max_value) ])

        bull_max, bull_second, bull_third = max_value, second_max_value, third_max_value

        tier1 = list(set(dict_code_pred[max_value]) & set(dict_code_pred[second_max_value]) & set(dict_code_pred[third_max_value]))
        tier2 = list(set(dict_code_pred[second_max_value]) & set(dict_code_pred[third_max_value]))

        f.write(f'Tier 1: {tier1}\n')
        f.write(f'Tier 2: {tier2}\n\n')

        for value in [max_value, second_max_value, third_max_value]:
            f.write(f'{int(value*100)}% Code List: {list(dict_code_pred[value])}\n')
            f.write(f'{int(value*100)}% Length: {len(list(dict_code_pred[value]))}\n')

        f.write(f'\nRandom Select: {sample(dict_code_pred[bull_third],int(len(dict_code_pred[bull_third])*0.5))}\n')

    return bull_max, bull_second, bull_third

def build_bear_tier_list(dict_code_pred, list_target_pct, directory_path):

    file_name = f'bear_tier.txt'
    file_path = os.path.join(directory_path, file_name)

    bear_max, bear_second, bear_third = [], [], []

    with open(file_path, 'w', encoding='utf-8') as f:

        min_value = min(list_target_pct)
        second_min_value = min([value for value in list_target_pct if value != min_value])
        third_min_value = min([value for value in list_target_pct if (value != min_value) and (value != second_min_value) ])

        bear_max, bear_second, bear_third = min_value, second_min_value, third_min_value

        tier1 = list(set(dict_code_pred[min_value]) & set(dict_code_pred[second_min_value]) & set(dict_code_pred[third_min_value]))
        tier2 = list(set(dict_code_pred[second_min_value]) & set(dict_code_pred[third_min_value]))

        f.write(f'Tier 1: {tier1}\n')
        f.write(f'Tier 2: {tier2}\n\n')

        for value in [min_value, second_min_value, third_min_value]:
            f.write(f'{int(value*100)}% Code List: {list(dict_code_pred[value])}\n')
            f.write(f'{int(value*100)}% Length: {len(list(dict_code_pred[value]))}\n')

        f.write(f'\nRandom Select: {sample(dict_code_pred[bear_third],int(len(dict_code_pred[bear_third])*0.5))}\n')

    return bear_max, bear_second, bear_third

def build_hedging(bull1, bull2, bull3, bear1, bear2, bear3, dict_code_pred, directory_path):

    # file_name = f'hedging.txt'
    # file_path = os.path.join(directory_path, file_name)

    lbear = dict_code_pred[bear1] + dict_code_pred[bear2] + dict_code_pred[bear3]
    lbull = dict_code_pred[bull1] + dict_code_pred[bull2] + dict_code_pred[bull3]

    lbear = set(lbear)
    lbull = set(lbull)

    lhedging = list(lbear.intersection(lbull))
    non_intersecting_bull = list((lbull - lbear))
    non_intersecting_bear = list((lbear - lbull))

    # with open(file_path, 'w', encoding='utf-8') as f:
    #     f.write(str(lhedging))

    with open(os.path.join(directory_path, 'bull_tier.txt'), 'a', encoding='utf-8') as f:
        f.write(f'\nHedging:  {str(lhedging)}')
        f.write(f'\nNon Hedging:  {str(non_intersecting_bull)}')

    with open(os.path.join(directory_path, 'bear_tier.txt'),'a', encoding='utf-8') as f:
        f.write(f'\nHedging:  {str(lhedging)}')
        f.write(f'\nNon Hedging:  {str(non_intersecting_bear)}')

def proccess_prediction(list_model_name, postfix, device, window_length, list_target_pct, directory_path, main_data, dict_type_name, connect):

    list_selected_type_id = Set.GetInfo('select_stock_list')

    for model_name in list_model_name:
        model_class = getattr(Model, model_name)
        #讀取txt as dictionary
        for target_pct in list_target_pct:

            file_name = f'{model_name}_{postfix}_{int(target_pct*100)}%_a.txt'
            file_path = os.path.join(directory_path, file_name)

            with open(file_path, 'w') as f:
                pass

            with open(file_path, 'a', encoding='utf-8') as f:

                f.write(f"------------------------------Model: {model_name}_{postfix}")
                print(f'Processing {model_name} {int(target_pct*100)}%')
                for type_id in tqdm.tqdm(list_selected_type_id):

                    list_unique_code_from_data = (SQLSentence.GetCodeByTypeId(type_id, connect))['code'].tolist()

                    model = model_class(GetDatasetShape(main_data, window_length)).to(device)
                    model.eval()

                    list_all_output = []

                    try:
                        model.load_state_dict(torch.load(f"{PATH_CHECKPOINT}/type_{type_id}/type_{type_id}_{model.__class__.__name__}_Target_{int(target_pct*100)}_{postfix}", weights_only=True))
                    except:
                        print(f'{dict_type_name[type_id]} No preserved model: {model.__class__.__name__}')
                        continue

                    for code in list_unique_code_from_data:
                        output = predict_result(main_data, code, model, device, target_pct, window_length)
                        list_all_output.append((code, output))
                        
                    f.write("------------------------------")
                    f.write(f"\n產業: {dict_type_name[type_id]} {type_id}\n")
                    
                    for code, probability in list_all_output:
                        f.write(f'{code} {dict_code_name[code]:<10} Prob: {probability}\n')

                f.write("\n\n")

def convert_to_str_percent(percent: float):
    return f"{int(percent*100)}%"

def get_dict_accuracy():
    file_path = 'model_records.csv'

    # Model: { target_pct: type_id: test_accuracy}
    # model_name - target_pct - type_id
    dict_accuracy = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model_name'] and row['target_pct'] and row['type_id'] and row['test_fp']:
                dict_accuracy[(row['model_name'], row['target_pct'], int(row['type_id']))] = float(row['test_acc'][:-1]) 
    return dict_accuracy

def check_trade_type(list_target_percent):
    if all(percent > 0 for percent in list_target_percent):
        return 'bull'
    elif all(percent < 0 for percent in list_target_percent):
        return 'bear'
    else:
        raise ValueError("Inconsistent target percent values: list_target_percent must contain either all positive or all negative values.")
    
def generate_dict_asc_output_sorted_code(list_model_name, list_target_percent, postfix, directory_path):
    dict_accuracy = get_dict_accuracy()
    dict_data = {}

    for model_name in list_model_name:
        for target_perctent in list_target_percent:
            with open(f'{directory_path}/{model_name}_{postfix}_{convert_to_str_percent(target_perctent)}_a.txt', 'r', encoding='utf-8') as file:
                dict_code_output = {}
                type_id = 0

                for line in file:
                    if '產業' in line:
                        type_id = int((line.split(' '))[-1])
                        weight = dict_accuracy[(model_name, convert_to_str_percent(target_perctent), type_id)]

                    if 'Prob' in line:
                        code = int(line[:4])
                        output = float((line.split(' '))[-1])
                        current_output = round(weight * output, 8)

                        # If the company code is reapted in multiple type id, then it should take the highest output.
                        if code not in dict_code_output:
                            dict_code_output[code] = current_output
                        else:
                            if dict_code_output[code] < current_output:
                                dict_code_output[code] = current_output
                                
                dict_data[(model_name, convert_to_str_percent(target_perctent))] = dict(sorted(dict_code_output.items(), key=lambda item: item[1], reverse=True))

    return dict_data

def ordered_output_code(list_model_name, list_target_percent, dict_data):
    dict_desc_sorted_code = {}
    for i, model_name in enumerate(list_model_name):
        for target_percent in list_target_percent:
            if i != 0:
                dict_desc_sorted_code = {value: (dict_desc_sorted_code[value]+index) for index, value in enumerate(dict_data[(model_name, convert_to_str_percent(target_percent))])}
            else:
                dict_desc_sorted_code = {value: index for index, value in enumerate(dict_data[(model_name, convert_to_str_percent(target_percent))])}

    list_desc_sorted_code = [key for key, _ in sorted(dict_desc_sorted_code.items(), key=lambda item: item[1])]

    return dict_desc_sorted_code, list_desc_sorted_code

def build_tier_list(list_model_name, list_target_percent, postfix, directory_path, target_number=10):

    dict_output_sorted_code = generate_dict_asc_output_sorted_code(list_model_name, list_target_percent, postfix, directory_path)
    dict_result, list_result = ordered_output_code(list_model_name, list_target_percent, dict_output_sorted_code)

    file_name = f'{check_trade_type(list_target_percent)}.txt'
    file_path = os.path.join(directory_path, file_name)

    with open(file_path, 'w') as f:
        pass

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(list_result[:target_number]))
        f.write('\n')
        f.write(str(list_result))
        f.write('\n')
        f.write(str(dict(sorted(dict_result.items(), key=lambda item: item[1], reverse=True))))

    return list_result[:target_number]

def export_data_to_MITAKE(bull_code_list, bear_code_list):
    with open(PATH_MITAKE_EXPORT_DATA, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        rows[1] = rows[1][:2] + bull_code_list  # 修改 bull row
        rows[2] = rows[2][:2] + bear_code_list  # 修改 bear row
        rows[3] = rows[3][:2] + list(set(bull_code_list) & set(bear_code_list))  # 修改 hedging row

        # 寫回檔案（覆蓋模式）
        with open(PATH_MITAKE_EXPORT_DATA, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

print('Function OK')

print('Proccess Paramaters')

list_model_name = ['CNN_LSTM', 'TCN']
list_target_pct = [0.02, 0.03, -0.02, -0.03]
list_bull_target = [0.02, 0.03]
list_bear_target = [-0.02, -0.03]

window_length = 20
postfix='best'

list_selected_type_id = Set.GetInfo('select_stock_list')
current_date = adjust_date((SQLSentence.GetLatestDate(connect))).strftime("%Y-%m-%d")

directory_path = f'prediction/{current_date}'

if not os.path.exists(directory_path):
    os.mkdir(directory_path) 

print('Sarting Predict')
proccess_prediction(list_model_name, postfix, device, window_length, list_target_pct, directory_path, main_data, dict_type_name, connect)
print('Complete Prediction')

print('Build Bull Tier List')
bull_code_list = build_tier_list(list_model_name, list_bull_target, postfix, directory_path)

print('Build Bear Tier List')
bear_code_list = build_tier_list(list_model_name, list_bear_target, postfix, directory_path)

export_data_to_MITAKE(bull_code_list, bear_code_list)

print(datetime.datetime.now())