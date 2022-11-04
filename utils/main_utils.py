import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
# import transformers
# from sentence_transformers import SentenceTransformer

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def pad_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens]


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]
    timestamp = [item[3] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens, timestamp]


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp

class DepressDataset(Dataset):
    def __init__(self, label, tweet, temporal, timestamp, current=True, random=False):
        super().__init__()
        self.label = label
        self.tweet = tweet
        self.temporal = temporal
        self.current = current
        self.timestamp = timestamp
        self.random = random

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        labels = torch.tensor(int(self.label[item]))
        tweet_features = self.tweet[item]
        if self.current:
            result = self.temporal[item]
            if self.random:
                np.random.shuffle(result)
            temporal_tweet_features = torch.tensor(result)
            timestamp = torch.tensor(get_timestamp(self.timestamp[item]))
        else:
            if len(self.temporal[item]) == 1:
                temporal_tweet_features = torch.zeros((1, 768), dtype=torch.float32)
                timestamp = torch.zeros((1, 1), dtype=torch.float32)
            else:
                temporal_tweet_features = torch.tensor(self.temporal[item][1:])
                timestamp = torch.tensor(get_timestamp(self.timestamp[item][1:]))

        return [labels, tweet_features, temporal_tweet_features, timestamp]

