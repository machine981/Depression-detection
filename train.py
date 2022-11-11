import argparse
import copy
import json
import os
import pickle
from datetime import datetime

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from model import HistoricCurrent, Historic, Current
from utils.main_utils import pad_ts_collate, DepressDataset

import sys
sys.path.append('/sdb/nlp21/Project/physical/depression-main/')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)
    true = true.view(batch_size,1)
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    class_labels = torch.arange(no_of_classes).float().cuda()
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    y = nn.Softmax(dim=1)(-phi)
    return y

def loss_function(output, labels, expt_type, scale):
    targets = true_metric_loss(labels, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, scale=1):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax", "ordinary".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    weight_cls = [1e-2]*no_of_classes
    for i in range(no_of_classes):
        # tell whether i is in the batch of labels
        if labels[labels==i].size(0):
            weight_cls[i]=samples_per_cls.pop(0)
        else:
            continue

    effective_num = 1.0 - np.power(beta, weight_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    elif loss_type == 'ordinary':
        no_of_class = logits.size()[-1]
        cb_loss = loss_function(logits, labels, no_of_class, scale)
    return cb_loss

def train_loop(model, dataloader, optimizer, device, dataset_len, class_num, loss_type):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for bi, inputs in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
        labels, tweet_features, temporal_features, lens, timestamp = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)
        temporal_features = temporal_features.to(device)
        lens = lens.to('cpu')
        timestamp = timestamp.to(device)

        optimizer.zero_grad()
        output = model(tweet_features, temporal_features, lens, timestamp)
        _, preds = torch.max(output, 1)

        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist(), class_num, loss_type)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len, class_num, loss_type):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    for bi, inputs in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
        labels, tweet_features, temporal_features, lens, timestamp = inputs

        labels = labels.to(device)
        tweet_features = tweet_features.to(device)
        temporal_features = temporal_features.to(device)
        lens = lens.to(device)
        timestamp = timestamp.to(device)

        with torch.no_grad():
            output = model(tweet_features, temporal_features, lens, timestamp)

        _, preds = torch.max(output, 1)
        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist(), class_num, loss_type)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)


def loss_fn(output, targets, samples_per_cls, class_num, loss_type):
    beta = 0.9999
    gamma = 2.0
    no_of_classes = class_num
    # loss_type = loss_type

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)

def grade_f1_score(confusion_mat):
    G_TP=0
    G_FN=0
    G_FP=0
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            if i==j:
                G_TP+=confusion_mat[i][j]
            elif i<j:
                G_FN+= confusion_mat[i][j]
            elif i>j:
                G_FP+= confusion_mat[i][j]
    G_precision = G_TP/(G_TP+G_FP)
    G_recall = G_TP/(G_TP+G_FN)
    G_f1 = 2*G_precision*G_recall/(G_precision+G_recall)
    print(f"***Grade Eval: GP:{G_precision}, GR:{G_recall}, GF:{G_f1}")
    return {'GP':G_precision, 'GR':G_recall, 'GF':G_f1}

def main(config):
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size

    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = config.embedding_dim

    NUM_LAYERS = config.num_layer
    DROPOUT = config.dropout
    CURRENT = config.current
    LOSS_FUNC = config.loss
    RANDOM = config.random

    DATA_DIR = config.data_dir
    DATA_NAME = config.dataset
    CLASS_NUM = config.class_num
    
    SEED = config.seed

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if config.base_model == "historic":
        model = Historic(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CLASS_NUM)
    elif config.base_model == "current":
        model = Current(HIDDEN_DIM, DROPOUT, CLASS_NUM)
    elif config.base_model == "historic-current":
        model = HistoricCurrent(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, config.model, CLASS_NUM, device)
    else:
        assert False

    with open(os.path.join(DATA_DIR, f'data/processed_{DATA_NAME}_data.pkl'), "rb") as f:
        df_total = pickle.load(f)
    
    # with open('/sdb/nlp21/Project/physical/STATENet_Time_Aware_Suicide_Assessment-master/data/samp_data.pkl','rb') as f:
    #     df_total = pickle.load(f)

    
    total_dataset = DepressDataset(df_total.label.values, df_total.curr_enc.values, df_total.enc.values,
                                    df_total.hist_dates, CURRENT, RANDOM)

    train_size = int(len(total_dataset)*0.7)
    val_size = int(len(total_dataset)*0.1)
    test_size = len(total_dataset) - train_size - val_size
    train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(\
                    total_dataset, [train_size, val_size, test_size],generator=torch.Generator().manual_seed(SEED))
    print(f'train_size:{train_size},val_size:{val_size},test_size:{test_size}')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_ts_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_ts_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_ts_collate)


    LEARNING_RATE = config.learning_rate

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=EPOCHS
    )

    model_name = f'{int(datetime.timestamp(datetime.now()))}_{config.base_model}_{config.model}_{config.hidden_dim}_{config.num_layer}_{config.learning_rate}'

    best_metric = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(model)
    print(optimizer)
    print(scheduler)

    for epoch in range(EPOCHS):
        loss, accuracy = train_loop(model, train_dataloader, optimizer, device, len(train_dataset), CLASS_NUM, LOSS_FUNC)
        eval_loss, eval_accuracy, __, _ = eval_loop(model, val_dataloader, device, len(val_dataset), CLASS_NUM, LOSS_FUNC)

        metric = f1_score(_, __, average="macro")
        recall = recall_score(_, __, average="macro")
        confusion = confusion_matrix(_, __, labels=[0, 1, 2, 3])
        print(confusion)
        grade_rep = grade_f1_score(confusion)
        if scheduler is not None:
            scheduler.step()

        print(
            f'epoch {epoch + 1}:: train: loss: {loss:.4f}, accuracy: {accuracy:.4f} | valid: loss: {eval_loss:.4f}, accuracy: {eval_accuracy:.4f}, f1: {metric:.4f}, recall: {recall:.4f}')
        if metric > best_metric:
            best_metric = metric
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 25 == 24:
            if scheduler is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_f1': best_metric
                }, os.path.join(DATA_DIR, f'checkpoints/saved_model/{model_name}_{epoch}.tar'))

    print(best_metric.item())
    model.load_state_dict(best_model_wts)

    if not os.path.exists('saved_model'):
        os.mkdir("saved_model")

    torch.save(model.state_dict(), os.path.join(DATA_DIR, f'checkpoints/saved_model/best_model_{model_name}.pt'))

    _, _, y_pred, y_true = eval_loop(model, val_dataloader, device, len(val_dataset), CLASS_NUM, LOSS_FUNC)

    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True)
    # print(report)
    result = {'best_f1': best_metric.item(),
              'lr': LEARNING_RATE,
              'model': str(model),
              'optimizer': str(optimizer),
              'scheduler': str(scheduler),
              'base-model': config.base_model,
              'model-name': config.model,
              'epochs': EPOCHS,
              'embedding_dim': EMBEDDING_DIM,
              'hidden_dim': HIDDEN_DIM,
              'num_layers': NUM_LAYERS,
              'dropout': DROPOUT,
              'current': CURRENT,
              'loss':LOSS_FUNC,
              'val_report': report}

    # with open(os.path.join(DATA_DIR, f'checkpoints/saved_model/VAL_{model_name}.json'), 'w') as f:
    #     json.dump(result, f)

    if config.test:
        _, _, y_pred, y_true = eval_loop(model, test_dataloader, device, len(test_dataset), CLASS_NUM, LOSS_FUNC)
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        print(confusion)

        report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True)
        print(report)
        grade_rep = grade_f1_score(confusion)
        result['test_report'] = report
        result['grade_report'] = grade_rep

        with open(os.path.join(DATA_DIR, f'checkpoints/saved_model/TEST_{model_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)




if __name__ == '__main__':
    base_model_set = {"historic", "historic-current", "current"}
    model_set = {"tlstm", "bilstm", "bilstm-attention"}
    loss_set = {"focal","softmax","sigmoid","ordinary"}
    dataset_set = {"reddit","depression","5c"}

    parser = argparse.ArgumentParser(description="Temporal Suicidal Modelling")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-e", "--epochs", default=300, type=int)
    parser.add_argument("-hd", "--hidden_dim", default=512, type=int)
    parser.add_argument("-ed", "--embedding-dim", default=768, type=int)
    parser.add_argument("-n", "--num_layer", default=2, type=int)
    parser.add_argument("-cn", "--class_num", default=4, type=int)
    parser.add_argument("-d", "--dropout", default=0.5, type=float)
    parser.add_argument("--base_model", type=str, choices=base_model_set, default="current")
    parser.add_argument("--loss", type=str, choices=base_model_set, default="focal")
    parser.add_argument("--dataset", type=str, choices=base_model_set, default="reddit")
    parser.add_argument("--model", type=str, choices=model_set, default="tlstm")
    parser.add_argument("-t", "--test", action="store_true", default=True)
    parser.add_argument("--current", action="store_false")
    parser.add_argument("--data_dir", type=str, default="/sdb/nlp21/Project/physical/depression-main")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", default=1895, type=int)
    config = parser.parse_args()

    main(config)

