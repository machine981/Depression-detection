import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import networkx as nx
import numpy as np
from argparse import ArgumentParser
import logging
import json

from utils.pretrain_gcn_utils import load_pickle, save_as_pickle, generate_text_graph
# from evaluate_results import evaluate_model_results

import sys
sys.path.append('/sdb/nlp21/Project/physical/depression-main/')

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class gcn(nn.Module):
    def __init__(self, X_size, A_hat, args, bias=True): # X_size = num features
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, args.hidden_size_1))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1, args.hidden_size_2))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_1))
            self.bias.data.normal_(0,var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(args.hidden_size_2))
            self.bias2.data.normal_(0,var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = nn.Linear(args.hidden_size_2, args.num_classes)
        self.dropout = nn.Dropout(p=args.dropout)
        
    def forward(self, X): ### 2-layer GCN architecture
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = self.dropout(X)
        X = F.relu(torch.mm(self.A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = self.dropout(X)
        X = F.relu(torch.mm(self.A_hat, X))
        return self.fc1(X)

    def get_state(self, X):
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = F.relu(torch.mm(self.A_hat, X))

        return X

def load_datasets(args, extra_name):
    """Loads dataset and graph if exists, else create and process them from raw data
    Returns --->
    f: torch tensor input of GCN (Identity matrix)
    X: input of GCN (Identity matrix)
    A_hat: transformed adjacency matrix A
    selected: indexes of selected labelled nodes for training
    test_idxs: indexes of not-selected nodes for inference/testing
    labels_selected: labels of selected labelled nodes for training
    labels_not_selected: labels of not-selected labelled nodes for inference/testing
    """
    logger.info("Loading data...")
    df_data_path = f"/sdb/nlp21/Project/physical/depression-main/data/{extra_name}_df_data.pkl"
    graph_path = f"/sdb/nlp21/Project/physical/depression-main/data/{extra_name}_text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph(args.path)
    df_data = load_pickle(df_data_path)
    G = load_pickle(graph_path)
    
    logger.info("Building adjacency and degree matrices...")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    
    logger.info("Splitting labels for training and inferring...")
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["l"].unique():
        dum = df_data[df_data["l"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))
    # save_as_pickle("test_idxs.pkl", test_idxs)
    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)
    # save_as_pickle("selected.pkl", selected)
    
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(df_data["l"]) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(df_data["l"]) if idx not in selected]
    f = torch.from_numpy(f).float()
    # save_as_pickle("labels_selected.pkl", labels_selected)
    # save_as_pickle("labels_not_selected.pkl", labels_not_selected)
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs
    
def load_state(net, optimizer, scheduler, model_no=0, load_best=False):
    """ Loads saved model and optimizer states if exists """
    logger.info("Initializing model and optimizer states...")
    base_path = "/sdb/nlp21/Project/physical/depression-main/checkpoints/pretrain_gcn"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "/sdb/nlp21/Project/physical/depression-main/data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "/sdb/nlp21/Project/physical/depression-main/data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([e for e in labels_e] == labels)/len(labels)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hidden_size_1", type=int, default=1300, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=768, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--path", default='/sdb/nlp21/Project/physical/depression-main/data/encoded_GCN_reddit_data.csv', help="Path of Graph")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--model_no", type=int, default=2, help="Model ID")
    args = parser.parse_args()
    # save_as_pickle("args.pkl", args)
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args, extra_name='reddit')
    net = gcn(X.shape[1], A_hat, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=False)
    losses_per_epoch, evaluation_untrained = load_results(model_no=args.model_no)
    
    logger.info("Starting training process...")
    net.train()
    evaluation_trained = []
    for e in range(start_epoch, args.num_epochs):
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long())
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                trained_accuracy = evaluate(output[selected], labels_selected); untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
            print("Labels of trained nodes: \n", output[selected].max(1)[1])
            net.train()
            if trained_accuracy > best_pred:
                best_pred = trained_accuracy
                torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("/sdb/nlp21/Project/physical/depression-main/checkpoints/pretrain_gcn" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))
        if (e % 10) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, evaluation_untrained)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("/sdb/nlp21/Project/physical/depression-main/checkpoints/pretrain_gcn",\
                    "test_checkpoint_%d.pth.tar" % args.model_no))
        scheduler.step()
        
    logger.info("Finished training!")
    evaluation_trained = np.array(evaluation_trained); evaluation_untrained = np.array(evaluation_untrained)