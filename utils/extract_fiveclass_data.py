import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import os
import json
import random
import emoji

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer

from nltk.corpus import stopwords

random.seed(1895)

def read_single_xml(path, cut_num=0):
    tree = ET.parse(path)
    root = tree.getroot()
    text_list = []
    date_list = []

    for child in root:
        if child.tag == 'ID':
            ID = child.text
        if child.tag != 'WRITING':
            continue
        for sub_child in child:
            if sub_child.tag == 'TEXT':
                text_list.append(sub_child.text)
            elif sub_child.tag == 'DATE':
                date_list.append(sub_child.text)
        if cut_num and len(text_list) > cut_num:
            break
    
    if len(text_list) != len(date_list):
        print('Data num not equal to text num. Wrong loading!')
    else: 
        return text_list, date_list

def gen_label_dict(label_path):
    label_dict = {}
    with open(label_path,'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.strip():
            # print(label_path)
            ID, label = line.strip().split()
            label_dict[ID] = label
    
    return label_dict

def general_preprocess(texts, text_processor, stopwords):
    processed_texts = []
    remove_ids = []
    for idx,text in enumerate(texts):
        text = emoji.demojize(text)
        text_tokens = text_processor.pre_process_doc(text)
        processed_text_tokens = filter_tokens(text_tokens, stopwords)
        processed_text = ' '.join(processed_text_tokens)
        if processed_text == '':
            remove_ids.append(idx)
            continue
        else:
            processed_texts.append(processed_text)
    
    return processed_texts, remove_ids
    
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def remove_date(dates, ids):
    new_dates = []
    for idx, date in enumerate(dates):
        if idx not in ids:
            new_dates.append(date)
   
    return new_dates

def gen_data_list(path, text_processor, stopwords, add_label=True, label_dict=None):
    data_list = []
    invalid_list = []

    if add_label:
        file_list = os.listdir(path)
        for file in file_list:
            if not file.endswith('.xml'):
                continue
            temp_dict = {}

            ID = file[:-4]
            try:
                print(os.path.join(path, file))
                text, date = read_single_xml(os.path.join(path, file))
            except:
                print(os.path.join(path, file), ' loading error!')

            label = label_dict[ID]
            text, remove_ids = general_preprocess(text,text_processor,stopwords)
            date = remove_date(date, remove_ids)

            temp_dict['ID'] = ID
            temp_dict['text'] = text
            temp_dict['date'] = date


            if len(temp_dict['text']) != len(temp_dict['date']):
                print('Length Not Equal!')

            temp_dict['label'] = int(label)+1

            data_list.append(temp_dict)
    else:
        neg_path = path
        cut_num = 150
        neg_file_list = os.listdir(neg_path)
        for file in neg_file_list:
            temp_dict = {}
            ID = file[:-4]
            try:
                print(os.path.join(path, file))
                text,date = read_single_xml(os.path.join(neg_path, file),cut_num=cut_num)
            except:
                print(os.path.join(neg_path, file), ' loading error!')

            text, remove_ids = general_preprocess(text[:cut_num],text_processor,stopwords)
            date = remove_date(date[:cut_num], remove_ids)

            temp_dict['ID'] = ID
            temp_dict['text'] = text
            temp_dict['date'] = date
            temp_dict['label'] = 0

            if len(temp_dict['text']) != len(temp_dict['date']):
                print('Length Not Equal!')

            data_list.append(temp_dict)

        # pos_path = os.path.join(path, 'depression')
        # neg_path = os.path.join(path, 'non-depression')

        # pos_file_list = os.listdir(pos_path)
        # for file in pos_file_list:
        #     temp_dict = {}

        #     ID = file[:-4]
        #     try:
        #         text,date = read_single_xml(os.path.join(pos_path, file))
        #     except:
        #         print(os.path.join(pos_path, file), ' loading error!')
            
        #     text, remove_ids = general_preprocess(text,text_processor,stopwords)
        #     date = remove_date(date, remove_ids)

        #     temp_dict['ID'] = ID
        #     temp_dict['text'] = text
        #     temp_dict['date'] = date
        #     temp_dict['label'] = 1

        #     if len(temp_dict['text']) != len(temp_dict['date']):
        #         print('Length Not Equal!')

        #     data_list.append(temp_dict)
        
        # neg_file_list = os.listdir(neg_path)
        # for file in neg_file_list:
        #     temp_dict = {}

        #     ID = file[:-4]
        #     try:
        #         text,date = read_single_xml(os.path.join(neg_path, file))
        #     except:
        #         print(os.path.join(neg_path, file), ' loading error!')

        #     text, remove_ids = general_preprocess(text,text_processor,stopwords)
        #     date = remove_date(date, remove_ids)

        #     temp_dict['ID'] = ID
        #     temp_dict['text'] = text
        #     temp_dict['date'] = date
        #     temp_dict['label'] = 0

        #     if len(temp_dict['text']) != len(temp_dict['date']):
        #         print('Length Not Equal!')

        #     data_list.append(temp_dict)
    
    return data_list


if __name__ == '__main__':

    # # stop words 
    # stop_words = []
    # with open('/sdb/nlp21/Project/physical/depression-main/utils/stop_word.txt','r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         stop_words.append(line.strip()) 
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

    total_data_list = []

    data_2019 = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2019/DATA'
    dl_2019 = '/sdb/nlp21/Project/physical/depression-main/raw_data/2019/eRisk2019-T3 Label.txt'

    data_2020a = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2020/eRisk2020-T2/DATA'
    dl_2020a = '/sdb/nlp21/Project/physical/depression-main/raw_data/2020/eRisk2020-T2/eRisk2020-T2.txt'

    data_2020b = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2020/eRisk2020-T2-TRAINING-DATA/DATA'
    dl_2020b = '/sdb/nlp21/Project/physical/depression-main/raw_data/2020/eRisk2020-T2-TRAINING-DATA/eRisk2020-T2-TRAINING-DATA.txt'

    data_2021 = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2021/DATA'
    dl_2021 = '/sdb/nlp21/Project/physical/depression-main/raw_data/2021/eRisk2021-T3.txt'

    data_2022 = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2022/DATA'


    # total_train_list.extend(gen_data_list(T1_train,True,gen_label_dict(T1_tr_label)))

    total_data_list.extend(gen_data_list(data_2019,text_processor,stop_words,True,gen_label_dict(dl_2019)))
    total_data_list.extend(gen_data_list(data_2020a,text_processor,stop_words,True,gen_label_dict(dl_2020a)))
    total_data_list.extend(gen_data_list(data_2020b,text_processor,stop_words,True,gen_label_dict(dl_2020b)))
    total_data_list.extend(gen_data_list(data_2021,text_processor,stop_words,True,gen_label_dict(dl_2021)))
    total_data_list.extend(gen_data_list(data_2022,text_processor,stop_words,False))


    random.shuffle(total_data_list)
    data_num = len(total_data_list)
    train_list, dev_list, test_list = total_data_list[:8*(data_num//10)],\
                                    total_data_list[8*(data_num//10):9*(data_num//10)],\
                                    total_data_list[9*(data_num//10):]


    json.dump(total_data_list, 
                open(os.path.join('/sdb/nlp21/Project/physical/depression-main/data', 'total_5c_data.json'), 'w'), 
                ensure_ascii=False, 
                indent=2)

    # json.dump(train_list, 
    #             open(os.path.join('/sdb/nlp21/Project/physical/depression-main/data', 'train_data.json'), 'w'), 
    #             ensure_ascii=False, 
    #             indent=2)

    # json.dump(dev_list, 
    #             open(os.path.join('/sdb/nlp21/Project/physical/depression-main/data', 'dev_data.json'), 'w'), 
    #             ensure_ascii=False, 
    #             indent=2)

    # json.dump(test_list, 
    #             open(os.path.join('/sdb/nlp21/Project/physical/depression-main/data', 'test_data.json'), 'w'), 
    #             ensure_ascii=False, 
    #             indent=2)

    print(f'Total  data num: {len(total_data_list)}')
    # print(f'Total train data num: {len(train_list)}')
    # print(f'Total dev data num: {len(dev_list)}')
    # print(f'Total test data num: {len(test_list)}')
    print('*** Have extracted and processed data from raw data!!! ***')
