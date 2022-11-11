from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForMaskedLM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import os
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np

from utils.pretrain_bert_utils import BERT_dataset,train_collate_fn

import setproctitle
setproctitle.setproctitle('pretrain_BERT_rep')

class BERT_model(object):
    def __init__(self, config):
        self.cfg = config
        self.device=self.cfg.device
        if self.cfg.mode == 'eval':
            self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_path)
            self.model=BertForMaskedLM.from_pretrained(self.cfg.model_path)
            print("Loading finetuned bert paras!")
        elif self.cfg.mode == 'train':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.model=BertForMaskedLM.from_pretrained("bert-base-cased")
            print("Loading pretrained bert paras!")
        self.model.to(self.device)

    def train(self, extra_name=''):
        train_data=BERT_dataset(self.cfg.data_path, self.tokenizer, extra_name)
        dev_data=BERT_dataset(self.cfg.data_path, self.tokenizer, extra_name)
        train_dataloader=DataLoader(train_data, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=train_collate_fn)
        dev_dataloader=DataLoader(dev_data, batch_size=self.cfg.eval_batch_size, collate_fn=train_collate_fn)

        optimizer, scheduler = self.get_optimizers(len(train_dataloader), self.model)
        global_step = 0
        min_loss=10000
        last_loss=10000

        for epoch in range(self.cfg.epoch_num):
            tr_loss = 0.0
            step_loss=0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            print("Epoch:{}".format(epoch))  
            for batch_idx, batch in pbar:
                try:  # avoid OOM
                    self.model.train()
                    inputs=batch.to(self.device) #B, T
                    labels=inputs
                    outputs = self.model(input_ids=inputs, labels=labels, return_dict=True)
                    # loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                    loss = outputs['loss']
                    loss=loss/self.cfg.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    step_loss+=loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    if (batch_idx+1) % self.cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        print("WARNING: ran out of memory,times: {}".format(oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print(str(exception))
                        raise exception
                     
            print('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss))
            current_loss = tr_loss
            # save model checkpoint
            if last_loss > current_loss:
                self.save_model(name='reddit_best_model')
                last_loss = current_loss



    def get_optimizers(self, num_samples, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.learning_rate)
        num_training_steps = num_samples*self.cfg.epoch_num // (self.cfg.gradient_accumulation_steps*self.cfg.batch_size)
        num_warmup_steps = int(num_training_steps*self.cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps)
        return optimizer, scheduler
    
    def eval(self, data):
        self.model.eval()
        total_loss=0
        with torch.no_grad():
            for batch in data:
                inputs=batch.to(self.device) #B, T
                labels=inputs
                outputs = self.model(inputs)
                loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                total_loss+=loss.item()
        return total_loss/len(data)

    def save_model(self, name):
        save_path = os.path.join(self.cfg.exp_path, name)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"model has been saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1895)
    parser.add_argument('--pad_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    # TODO
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data_path', default='/sdb/nlp21/Project/physical/depression-main/data/total_reddit_data.json')
    parser.add_argument('--exp_path', default='/sdb/nlp21/Project/physical/depression-main/checkpoints/pretrain_bert')
    parser.add_argument('--model_path', default=None)


    cfg = parser.parse_args()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = BERT_model(cfg)
    model.train(extra_name='reddit')

