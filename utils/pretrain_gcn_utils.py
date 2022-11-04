import csv
import os
import pickle
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
import logging

import sys
sys.path.append('/sdb/nlp21/Project/physical/depression-main/')

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def gen_csv_data(data_path ,mode, part_num=10, extra_name=''):
    # mode: whole, current
    # TODO
    data_file = os.path.join('/sdb/nlp21/Project/physical/depression-main/data', f'encoded_GCN_{mode}_{extra_name}data.csv')
    data_list = json.load(open(data_path, 'r', encoding='utf-8'))
    temp_list = []
    if mode == 'whole':
        for data in data_list:
            sentence_list = []
            for sent in data['text']:
                sent.replace('\n','')
                # remove short sentence
                if len(sent)>5:
                    sentence_list.append(sent)
            new_sent = '.'.join(sentence_list)
            if len(new_sent)>5:
                temp_list.append([new_sent,data['label']])
            else:
                continue

    elif mode == 'current':
        for data in data_list:
            sent=data['text'][-1]
            sent.replace('\n','')
            if len(sent)>5:
                temp_list.append([sent,data['label']])
            else:
                continue
    
    elif mode == 'reddit':
        for data in data_list:
            sent=data['text']
            sent.replace('\n','')
            if len(sent)>5:
                temp_list.append([sent,data['label']])
            else:
                continue

    elif mode == 'partial':
        for data in data_list:
            sentence_list=data['text'][:part_num]
            new_sent = '.'.join(sentence_list)
            new_sent.replace('\n','')
            if len(new_sent)>0:
                temp_list.append([new_sent,data['label']])
    
    with open(data_file,'w') as file:
        writer = csv.writer(file)
        writer.writerow(["t","l"])
        for data in temp_list:
            writer.writerow(data)
            
    print(f"total data num: {len(temp_list)}")
    print(f"{mode} csv file is generated!")

def load_pickle(filename):
    completeName = os.path.join("/sdb/nlp21/Project/physical/depression-main/data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("/sdb/nlp21/Project/physical/depression-main/data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))

### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns); cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1,w2] > 0):
            word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]}))
    return word_word

def generate_text_graph(path, window=10, extra_name=''):
    """ generates graph based on text corpus; window = sliding window size to calculate point-wise mutual information between words """
    logger.info("Preparing data...")
    df_data = pd.read_csv(path)
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    
    ### tokenize & remove funny characters
    df_data["t"] = df_data["t"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
    save_as_pickle(f"{extra_name}_df_data.pkl", df_data)
    
    ### Tfidf
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["t"])
    df_tfidf = vectorizer.transform(df_data["t"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
    
    ### PMI between words
    names = vocab
    n_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict((name,index) for index,name in enumerate(names))

    occurrences = np.zeros((len(names),len(names)), dtype=np.int32)
    # Find the co-occurrences:
    no_windows = 0; logger.info("Calculating co-occurences...")
    for l in tqdm(df_data["t"], total=len(df_data["t"])):
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])

            for w in d:
                n_i[w] += 1
            for w1,w2 in combinations(d,2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1

    logger.info("Calculating PMI*...")
    ### convert to PMI
    p_ij = pd.DataFrame(occurrences, index = names,columns=names)/no_windows
    p_i = pd.Series(n_i, index=n_i.keys())/no_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col]/p_i[col]
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
        
    ### Build graph
    logger.info("Building graph (No. of document, word nodes: %d, %d)..." %(len(df_tfidf.index), len(vocab)))
    G = nx.Graph()
    logger.info("Adding document nodes to graph...")
    G.add_nodes_from(df_tfidf.index) ## document nodes
    logger.info("Adding word nodes to graph...")
    G.add_nodes_from(vocab) ## word nodes
    ### build edges between document-word pairs
    logger.info("Building document-word edges...")
    document_word = [(doc,w,{"weight":df_tfidf.loc[doc,w]}) for doc in tqdm(df_tfidf.index, total=len(df_tfidf.index))\
                     for w in df_tfidf.columns]
    
    logger.info("Building word-word edges...")
    word_word = word_word_edges(p_ij)
    save_as_pickle(f"{extra_name}_word_word_edges.pkl", word_word)
    logger.info("Adding document-word and word-word edges...")
    G.add_edges_from(document_word)
    G.add_edges_from(word_word)
    save_as_pickle(f"{extra_name}_text_graph.pkl", G)
    logger.info("Done and saved!")
    
if __name__=="__main__":
    # data_path = '/sdb/nlp21/Project/physical/depression-main/data/total_data.json'
    # gen_csv_data(data_path=data_path, mode='reddit')  
    # gen_csv_data(data_path=data_path, mode='partial') 
    # gen_csv_data(data_path=data_path, mode='partial', extra_name='5c')
    csv_path =  '/sdb/nlp21/Project/physical/depression-main/data/encoded_GCN_reddit_data.csv'
    generate_text_graph(path=csv_path, extra_name='reddit')