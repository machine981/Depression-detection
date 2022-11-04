import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

from argparse import ArgumentParser
import json
import pandas as pd
from tqdm import tqdm

from pretrain_gcn import load_datasets, load_state, gcn
from pretrain_bert import BERT_model, BERT_dataset, train_collate_fn

import datetime

class bert_config():
# for bert config
    def __init__(self):
        self.device = 'cuda:1'
        self.data_path = '/sdb/nlp21/Project/physical/depression-main/data/total_reddit_data.json'
        self.model_path = '/sdb/nlp21/Project/physical/depression-main/checkpoints/pretrain_bert/best_model'

class gcn_config():
# for gcn config
    def __init__(self):
        self.hidden_size_1 = 1300
        self.hidden_size_2 = 768
        self.num_classes = 4
        self.test_ratio = 0.1
        self.path = '/sdb/nlp21/Project/physical/depression-main/data/encoded_GCN_reddit_data.csv'
        self.num_epochs = 3300
        self.lr = 0.011
        self.model_no = 2
        self.device = 'cuda:1'
        self.dropout = 0

def get_bert_repre(data_num=190, extra_name=''):

    args = bert_config()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path)
    model.to(args.device)
    data = BERT_dataset(args.data_path, tokenizer, extra_name)
    dataloader=DataLoader(data, batch_size=16, shuffle=False, collate_fn=train_collate_fn)
    bert_repre = []
    model.eval()
    for batch in tqdm(dataloader):
        inputs=batch.to(args.device)
        output = model(input_ids=inputs, return_dict=True)
        repre = output['pooler_output'].cpu().detach().numpy().tolist()
        bert_repre.extend(repre)
        
    print("Get bert representation")
    return bert_repre

def get_gcn_repre(data_num=190, extra_name=''):
    args = gcn_config()
    f, X, A_hat, _, _, _, _ = load_datasets(args, extra_name)
    net = gcn(X.shape[1], A_hat, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=True)
    net.eval()
    output = net.get_state(f)
    gcn_repre = output[:data_num].detach().numpy().tolist()
    print("Get gcn representation")

    return gcn_repre

def get_history_num(path):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    length_list = []
    total_length = 0
    for data in data_list:
        length = len(data['text'])
        total_length += length
        length_list.append(length)

    print(f"total length num: {total_length}")

    return length_list

def gen_datetime(dates):
    date_list = []
    for date in dates:
        fmt = '%Y-%m-%d %H:%M:%S'
        date_time = datetime.datetime.strptime(date.strip(), fmt)
        date_list.append(date_time)

    return date_list

def gen_pickle_data(gcn_rep, bert_rep, all_data):
    cur_index = 0
    part_num = 10
    datatype = 'with_bina'
    df = pd.DataFrame(columns=['label','curr_enc','enc','hist_dates'])
    for i,data in enumerate(all_data):
        # cur_index = 0
        length = len(data['text'])-part_num
        temp_df = pd.DataFrame(columns=['label','curr_enc','enc','hist_dates'])

        temp_df.assign(label=lambda x: x.label.astype(int))
        temp_df.assign(enc=lambda x: x.enc.astype(object))
        temp_df.assign(curr_enc=lambda x: x.curr_enc.astype(object))
        temp_df.assign(hist_dates=lambda x: x.hist_dates.astype(object))

        date_time = gen_datetime(data['date'])
        real_length = min(100, length)

        temp_df.at[0,'label'] = int(data['label'])
        temp_df.at[0,'curr_enc'] = gcn_rep[i]
        temp_df.at[0,'enc'] = bert_rep[cur_index:cur_index+real_length]
        temp_df.at[0,'hist_dates'] = date_time

        # temp_list = [data['label'], gcn_rep[i], bert_rep[cur_index:cur_index+length],data['date']]
        # serie = pd.Series(temp_list, index=['label','curr_enc','enc','hist_dates'])
        df = pd.concat([df,temp_df],ignore_index=True)

        cur_index += length
    df.to_pickle(f'/sdb/nlp21/Project/physical/depression-main/data/processed_{datatype}_data.pkl')

def gen_pickle_reddit_data(gcn_rep, bert_rep, all_data):
    cur_index = 0
    datatype = 'reddit'
    df = pd.DataFrame(columns=['label','curr_enc','enc','hist_dates'])
    for i,data in enumerate(all_data):
        # cur_index = 0
        temp_df = pd.DataFrame(columns=['label','curr_enc','enc','hist_dates'])

        temp_df.assign(label=lambda x: x.label.astype(int))
        temp_df.assign(enc=lambda x: x.enc.astype(object))
        temp_df.assign(curr_enc=lambda x: x.curr_enc.astype(object))
        temp_df.assign(hist_dates=lambda x: x.hist_dates.astype(object))

        fmt = '%Y-%m-%d %H:%M:%S'
        date = '2018-07-09 05:18:12'
        date_time = datetime.datetime.strptime(date.strip(), fmt)

        temp_df.at[0,'label'] = int(data['label'])
        temp_df.at[0,'curr_enc'] = gcn_rep[i]
        temp_df.at[0,'enc'] = [bert_rep[i]]
        temp_df.at[0,'hist_dates'] = [date_time]

        # temp_list = [data['label'], gcn_rep[i], bert_rep[cur_index:cur_index+length],data['date']]
        # serie = pd.Series(temp_list, index=['label','curr_enc','enc','hist_dates'])
        df = pd.concat([df,temp_df],ignore_index=True)

    df.to_pickle(f'/sdb/nlp21/Project/physical/depression-main/data/processed_{datatype}_data.pkl')

if __name__ == '__main__':
    # 调整生成数据时，需要修改 gcn bert 文件路径、gcn-data_num、main文件路径



    # path = f'/sdb/nlp21/Project/physical/depression-main/data/total_data.json'
    # all_data = json.load(open(path, 'r', encoding='utf-8'))
    # bert_repre = get_bert_repre()
    # gcn_repre = get_gcn_repre()
    # gen_pickle_data(gcn_repre,bert_repre,all_data)

    path = f'/sdb/nlp21/Project/physical/depression-main/data/total_reddit_data.json'
    all_data = json.load(open(path, 'r', encoding='utf-8'))
    bert_repre = get_bert_repre(extra_name='reddit')
    gcn_repre = get_gcn_repre(data_num=3549, extra_name='reddit')
    gen_pickle_reddit_data(gcn_repre,bert_repre,all_data)
    # gen_pickle_data(gcn_repre,bert_repre,all_data)
    
