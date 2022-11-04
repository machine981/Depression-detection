import csv
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from nltk.corpus import stopwords
import emoji
import random
import json
import os

random.seed(1895)

label_dict = {'minimum':0,'mild':1,'moderate':2,'severe':3}

def general_preprocess(text, text_processor, stopwords):
    text = emoji.demojize(text)
    text_tokens = text_processor.pre_process_doc(text)
    processed_text_tokens = filter_tokens(text_tokens, stopwords)
    processed_text = ' '.join(processed_text_tokens)

    return processed_text
    
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def remove_date(labels, ids):
    new_labels = []
    for idx, label in enumerate(labels):
        if idx not in ids:
            new_labels.append(label)
   
    return new_labels

def get_reddit_content(text_processor, stopwords):
    data_list = []
    with open('/sdb/nlp21/Project/physical/depression-main/raw_data/reddit_depression/Reddit_depression_dataset.csv') as f:
        csv_reader = csv.reader(f)
        for i,row in enumerate(csv_reader):
            if i == 0:
                continue
            temp_dict = {}
            text = general_preprocess(row[0],text_processor,stopwords)
            if text == '':
                continue
            else:
                temp_dict['text'] = text
                temp_dict['label'] = label_dict[row[1]]          
                data_list.append(temp_dict)
    
    return data_list 


if __name__ == "__main__":

    stop_words = stopwords.words('english')

    text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    fix_html=True,  # fix HTML tokens
    segmenter="twitter", 
    corrector="twitter", 
    unpack_hashtags=True,
    unpack_contractions=True, 
    spell_correct_elong=True, 
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    )
    
    data_list = get_reddit_content(text_processor,stop_words)

    json.dump(data_list, 
                open(os.path.join('/sdb/nlp21/Project/physical/depression-main/data', 'total_reddit_data.json'), 'w'), 
                ensure_ascii=False, 
                indent=2)
    
    print(f'Total data num: {len(data_list)}')
    print('*** Have extracted and processed data from raw data!!! ***')