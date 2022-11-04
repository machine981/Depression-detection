import pickle
import json
from utils.extract_fourclass_data import read_single_xml
import os
import torch
import numpy as np


print(np.power(0.999,1e-2))



# completeName = '/sdb/nlp21/Project/physical/STATENet_Time_Aware_Suicide_Assessment-master/data/samp_data.pkl'
# with open(completeName, 'rb') as pkl_file:
#     data = pickle.load(pkl_file)

# # print(data[['enc']])

# for i in range(100):
#     print(len(data[['enc']].iat[i,0]))

# data_list = json.load(open('/sdb/nlp21/Project/physical/depression-main/data/total_reddit_data.json', 'r', encoding='utf-8'))
# for i,data in enumerate(data_list):
#     if len(data['text']) <= 5:
#         print(i)
#         print(data['text'])
    
# print(len(data_list))

# data = torch.Tensor([4,0,1,2,1,4,2,2,2,2])
# # print(torch.unique(data, return_counts=True))

# print(data[data==5].size(0))
# ID_list = []
# text_list = []
# for data in data_list:
#     ID_list.append(data['ID'])
#     text_list.extend(data['text'])


# print(len(ID_list))
# print(len(set(ID_list)))

# print(len(text_list))
# print(len(set(text_list)))



def revise_data(path):
    old_pre = '/sdb/nlp21/Project/physical/depression-main/raw_data'
    new_pre = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2022/DATA'
    new_path = os.path.join(new_pre)
    old_path = os.path.join(old_pre, path)

    file_list = os.listdir(old_path)
    for file in file_list:
        if not file.endswith('.xml'):
            continue
        with open(os.path.join(new_path, file),'w') as file_revise:
            with open(os.path.join(old_path, file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line=line.replace('&','and')
                    file_revise.write(line)


# data_2019 = '2019/DATA'
# data_2020a = '2020/eRisk2020-T2/DATA'
# data_2020b = '2020/eRisk2020-T2-TRAINING-DATA/DATA'
# data_2021 = '2021/DATA'
# data_17 = '2022/training data/2017 cases/non-depression'
# data_18 = '2022/training data/2018 cases/non-depression'


# revise_data(data_2019)
# revise_data(data_2020a)
# revise_data(data_2020b)
# revise_data(data_2021)
# revise_data(data_17)
# revise_data(data_18)


# read_single_xml('/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2021/DATA/erisk2021-T3_Subject21feeeeeeeeei8.xml')
