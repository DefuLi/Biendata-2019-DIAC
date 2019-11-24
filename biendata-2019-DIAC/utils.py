import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from itertools import product, combinations
from transformers import BertTokenizer
from sklearn.metrics import f1_score

# set path
train_data_path = './data/train_set.xml'
test_data_path = './data/dev_set.csv'

# set seed
np.random.seed(520)

# set bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# set hyper parameter
MAX_LEN = 40  # 接近最优

# 加载训练数据
# [[['哪些情形下，不予受理民事诉讼申请？', '民事诉讼中对哪些情形的起诉法院不予受理', ...],
# ['民事诉讼什么情况下不能立案', '哪些案件会给开具民事诉讼不予立案通知书', ...]], [[], []]]
def load_train_data(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    train_data = []
    for Questions in root:
        Questions_list = []
        flag = True
        for Equ_Not_Questions in Questions:
            question_list = []
            for questions in Equ_Not_Questions:
                if flag:
                    question_list.append(questions.text)
                else:
                    if questions.text == None:
                        continue
                    question_list.append(questions.text)
            Questions_list.append(question_list)
            flag = False
        train_data.append(Questions_list)
 
    return train_data

# 生成正负样本 正19228 负61924
def generate_pn_sample(train_data):
    p_list = []  # [sentence1, sentence2, 1]
    n_list = []  # [sentence1, sentence2, 0]
    for item in train_data:
        p_com = [list(sub_item) for sub_item in combinations(item[0], 2)]
        for sub_item in p_com:
            sub_item.append(1)
            p_list.append(sub_item)
        
        n_com = [list(sub_item) for sub_item in product(item[0], item[1])]
        for sub_item in n_com:
            sub_item.append(0)
            n_list.append(sub_item)

    return p_list, n_list

# 合并正负样本 同等比例 去掉一部分负样本
def combine_pn(p_list, n_list):
    p_list = np.array(p_list)
    n_list = np.array(n_list)

    np.random.shuffle(p_list)
    np.random.shuffle(n_list)
    n_list = n_list[:len(p_list)]  # 设置负样本数量与正样本一致
    pn_list = np.vstack((p_list, n_list))
    np.random.shuffle(pn_list)

    return pn_list.tolist()

# 多折交叉验证
def split_data(pn_list):
    pn_len = len(pn_list)
    fold1 = pn_list[:int(pn_len/5)]
    fold2 = pn_list[int(pn_len/5):int(pn_len/5)*2]
    fold3 = pn_list[int(pn_len/5)*2:int(pn_len/5)*3]
    fold4 = pn_list[int(pn_len/5)*3:int(pn_len/5)*4]
    fold5 = pn_list[int(pn_len/5)*4:]
    return [fold1, fold2, fold3, fold4, fold5]
    # fold1 = pn_list[:int(pn_len/3)]
    # fold2 = pn_list[int(pn_len/3):int(pn_len/3)*2]
    # fold3 = pn_list[int(pn_len/5)*2:]

    # return [fold1, fold2, fold3]

# 生成固定长度的ids 对整个数据集进行处理
def bert_input(data_list, islabel=True):
    bert_list = []
    for item in data_list:
        ids1 = tokenizer.encode(item[0])
        ids2 = tokenizer.encode(item[1])
        num1 = MAX_LEN - len(ids1)
        num2 = MAX_LEN - len(ids2)
        if num1 < 0:
            ids1 = ids1[:MAX_LEN]
        else:
            for _ in range(num1):
                ids1.append(0)
        
        if num2 < 0:
            ids2 = ids2[:MAX_LEN]
        else:
            for _ in range(num2):
                ids2.append(0)
        
        ids = [101] + ids1 + [102] + ids2 + [102]  # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
        if islabel:
            bert_list.append([ids, item[2]])
        else:
            bert_list.append([ids])
    return bert_list

# 计算每个batch的准确率
def batch_accuracy(logits, label):
    logits = logits.argmax(dim=1)
    correct = torch.eq(logits, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy

# 计算每个batch的f1值
def batch_f1(logits, label):
    logits = logits.argmax(dim=1).tolist()
    label = label.tolist()
    f1 = f1_score(logits, label, labels=[0, 1], average='macro')

    return f1

# 加载测试数据
def load_test_data(test_data_path):
    test_csv = pd.read_csv(test_data_path, sep='\t')
    test_list = []
    for index in range(len(test_csv)):
        test_list.append([test_csv.iloc[index][1], test_csv.iloc[index][2]])
    
    return test_list

# 5折测试数据投票
def vote(fold_result):
    fold_result = np.array(fold_result).T
    vote_result = []
    index = 0
    for item in fold_result:
        count = np.bincount(item)
        vote_value = np.argmax(count)
        vote_result.append([index, int(vote_value)])
        index += 1

    return vote_result

# 测试结果写入csv
def write_csv(result_list):
    df = pd.DataFrame(data=result_list)
    df.to_csv('result.csv', sep='\t', index=0, header=0)

# main.py中调用的预处理主函数
def main():
    train_data = load_train_data(train_data_path)
    p_list, n_list = generate_pn_sample(train_data)
    pn_list = combine_pn(p_list, n_list)
    fold_all = split_data(pn_list)

    test_list = load_test_data(test_data_path)
    test_bert_list = bert_input(test_list, islabel=False)

    return fold_all, test_bert_list

if __name__ == "__main__":
    pass