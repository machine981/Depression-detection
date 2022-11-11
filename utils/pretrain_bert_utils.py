import json
import os

import torch
import torch.utils.data as data
import numpy as np

from tqdm import tqdm

def read_data(path, tokenizer, extra_name=''):
    # TODO
    part_num = 10
    data_file = os.path.join('/sdb/nlp21/Project/physical/depression-main/data', f'encoded_BERT_{extra_name}_data.json')
    if not os.path.exists(data_file):
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        sentence_list = []
        for data in tqdm(data_list):
            if extra_name == 'reddit':
                sent = data['text'].replace('\n','')
                encode_sent = tokenizer.encode(sent)
                sentence_list.append(encode_sent)
            else:
                for sent in data['text'][part_num:]:
                    # if len(sent)>5:
                    sent = sent.replace('\n','')
                    encode_sent = tokenizer.encode(sent)
                    sentence_list.append(encode_sent)
        
        json.dump(sentence_list, open(data_file, 'w'), ensure_ascii=False)
    else:
        sentence_list = json.load(open(data_file, 'r', encoding='utf-8'))

    return sentence_list

def read_whole_data(tokenizer, extra_name=''):
    # TODO
    part_num = 10
    file_path = ['/sdb/nlp21/Project/physical/depression-main/data/total_5c_data.json',\
                '/sdb/nlp21/Project/physical/depression-main/data/total_reddit_data.json']
    data_file = os.path.join('/sdb/nlp21/Project/physical/depression-main/data', f'encoded_BERT_whole_data.json')
    sentence_list = []
    if not os.path.exists(data_file):
        data_list = json.load(open(file_path[0], 'r', encoding='utf-8'))
        for data in tqdm(data_list):
            for sent in data['text'][part_num:]:
                # if len(sent)>5:
                sent = sent.replace('\n','')
                encode_sent = tokenizer.encode(sent)
                sentence_list.append(encode_sent)
        
        data_list = json.load(open(file_path[1], 'r', encoding='utf-8'))
        for data in tqdm(data_list):
            sent = data['text'].replace('\n','')
            encode_sent = tokenizer.encode(sent)
            sentence_list.append(encode_sent)
        
        json.dump(sentence_list, open(data_file, 'w'), ensure_ascii=False)
    else:
        sentence_list = json.load(open(data_file, 'r', encoding='utf-8'))

    return sentence_list

class BERT_dataset(data.Dataset):
    def __init__(self, path, tokenizer, extra_name=''):
        self.data_info = read_data(path, tokenizer, extra_name)
        # self.data_info = read_whole_data(tokenizer, extra_name)

    def __getitem__(self, index):
        return self.data_info[index]
    
    def __len__(self):
        return len(self.data_info)

def train_collate_fn(batch):
    pad_batch = padSeqs(batch, pad_id=0)
    batch_tensor=torch.from_numpy(pad_batch).long()
    return batch_tensor

def padSeqs(sequences, pad_id, maxlen=None, seq=False):
    lengths = [len(x) for x in sequences]
    maxlen=max(lengths)
    maxlen=min(512, maxlen)
    if seq:
        pad_batch=np.ones((len(sequences), maxlen))*(-100)
    else:
        pad_batch=np.ones((len(sequences), maxlen))*pad_id
    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        pad_batch[idx, :len(trunc)] = trunc
            
    return pad_batch