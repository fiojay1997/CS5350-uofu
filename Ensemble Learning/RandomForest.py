import csv
import numpy as np
import math
import random

def load_data(filename):
    dataset = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            dataset.append(line)
    return dataset


def convert_data(dataset):
    length = len(dataset[0]) - 1
    for data in dataset:
        for c in range(length):
            data[c] = float(data[c].strip())


def train(dataset, n):
    subset = []
    features = []
    num = np.shape(dataset)[1]
    k = int(math.log(n - 1, 2)) + 1 if n > 2 else 1
    for _ in range(n):
        sample, feature = choose_sample(dataset, k)
        tree = build_tree(sample)
        subset.append(tree)
        features.append(feature)
    return subset, features


def choose_sample(dataset, k):
    m, n = np.shape(dataset)  
    feature = []
    for _ in range(k):
        feature.append(random.randint(0, n - 2)) 
    index = []
    for i in range(m):
        index.append(random.randint(0, m - 1))
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(dataset[index[i]][fea])
        data_tmp.append(dataset[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def build_tree(data):
    if len(data) == 0:
        return node()
   
    # 1、计算当前的Gini指数
    currentGini = cal_gini_index(data)
   
    bestGain = 0.0
    bestCriteria = None  # 存储最佳切分属性以及最佳切分点
    bestSets = None  # 存储切分后的两个数据集
   
    feature_num = len(data[0]) - 1  # 样本中特征的个数
    # 2、找到最好的划分
    for fea in range(0, feature_num):
        # 2.1、取得fea特征处所有可能的取值
        feature_values = {}  # 在fea位置处可能的取值
        for sample in data:  # 对每一个样本
            feature_values[sample[fea]] = 1  # 存储特征fea处所有可能的取值
       
        # 2.2、针对每一个可能的取值，尝试将数据集划分，并计算Gini指数
        for value in feature_values.keys():  # 遍历该属性的所有切分点
            # 2.2.1、 根据fea特征中的值value将数据集划分成左右子树
            (set_1, set_2) = split_tree(data, fea, value)
            # 2.2.2、计算当前的Gini指数
            nowGini = float(len(set_1) * cal_gini_index(set_1) + 
                             len(set_2) * cal_gini_index(set_2)) / len(data)
            # 2.2.3、计算Gini指数的增加量
            gain = currentGini - nowGini
            # 2.2.4、判断此划分是否比当前的划分更好
            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)
   
    # 3、判断划分是否结束
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0], value=bestCriteria[1], 
                    right=right, left=left)
    else:
        return node(results=label_uniq_cnt(data)) 