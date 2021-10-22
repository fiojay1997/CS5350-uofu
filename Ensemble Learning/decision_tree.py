import numpy as np
import operator
from random import randint
import math

class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}

    def is_leaf(self):
        return len(self.children) == 0


def get_label_info(dataset):
    count = len(dataset)
    label_count = {}
    for data in dataset:
        if data[-1] not in label_count.keys():
            label_count[data[-1]] = 1
        else:
            label_count[data[-1]] += 1
    return label_count, count


def get_ent(dataset):
    label_count, count = get_label_info(dataset)
    ent = 0.0
    for data in label_count:
        prob = label_count[data] / count
        ent -= prob * np.log2(prob)
    return ent


def get_me(dataset):
    label_count, count = get_label_info(dataset)
    return 1 - (max(label_count.values()) / count)


def get_gini(dataset):
    label_count, count = get_label_info(dataset)
    gini = 1.0
    for data in label_count:
        prob = label_count[data] / count
        gini -= np.square(prob)
    return gini


def split(dataset, s, value):
    subset = []
    for data in dataset:
        if data[s] == value:
            newset = data[:s]
            newset.extend(data[s+1:])
            subset.append(newset)
    return subset


def choose_feature(dataset, split_func):
    num = len(dataset[0])-1
    base = split_func(dataset)
    bestinfo_gain = 0
    best_feature = 0
    for i in range(num):
        features = [data[i] for data in dataset]
        vals = set(features)
        new_val = 0
        for value in vals:
            subDataSet = split(dataset, i, value)
            prob = len(subDataSet) / len(dataset)
            new_val += prob * split_func(subDataSet)
        info_gain = base - new_val
        if (info_gain > bestinfo_gain):
            bestinfo_gain = info_gain
            best_feature = i
    return best_feature


def get_most_common_feature(columns):
    count = {}
    for vote in columns:
        if vote not in count.keys():
            count[vote] = 0
        count[vote] += 1
    sorted_count = sorted(
        count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_count[0][0]


def create_tree(dataset, labels, split_func, max_depth, cur_depth=0):
    columns = [data[-1] for data in dataset]
    if columns.count(columns[0]) == len(columns):
        return Node(str(columns[0]))
    if len(dataset[0]) == 1:
        return Node(str(get_most_common_feature(columns)))
    if max_depth == cur_depth:
        return Node(str(get_most_common_feature(columns)))

    if split_func == 'gini':
        func = get_gini
    elif split_func == 'entropy':
        func = get_ent
    else:
        func = get_me

    best_feature = choose_feature(dataset, func)
    best_featureLabel = labels[best_feature]
    dt = Node(str(best_featureLabel))
    del(labels[best_feature])
    feature = [data[best_feature] for data in dataset]
    vals = set(feature)
    for value in vals:
        subLabels = labels[:]
        dt.children[value] = create_tree(split
                                         (dataset, best_feature,
                                          value), subLabels, split_func,
                                         max_depth, cur_depth + 1)
    return dt


def load_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(line.strip().split(','))
    return data


def predict(dataset, features, decision_tree):
    label = dataset[-1]
    cur_tree = decision_tree
    while not cur_tree.is_leaf():
        feature = cur_tree.name
        attr_val = dataset[features.index(feature)]
        cur_tree = cur_tree.children[attr_val]
    return cur_tree.name == label


def adaboost(dataset, features, round):
    classifier = create_tree(dataset, features, 'gini', 1)
    length = len(dataset)
    weight = 1 / length
    predictions = [np.zeros(length)]

    for _ in range(round):
        predict = predict(dataset, features, classifier)
        miss = [int(x) for x in (predict != features)]
        err = np.dot(weight, miss)
        alpha = 0.5 * np.log((1 - err) / float(err))
        filter_miss = [x if x == 1 else - 1 for x in miss]
        weight = np.multiply(weight, np.exp([float(x) * alpha for x in filter_miss]))
        weight = sum(weight)
        prediction = [1 if x == 1 else -1 for x in predict]
        predictions = predictions + np.multiply(alpha, prediction)
    
    predictions = (predictions > 0) * 1
    return predictions


def bagged(dataset, features, round):
    classifier = create_tree(dataset, features, 'gini', 1)
    predictions = []
    votes = []
    for _ in range(round):
        samples = []
        for _ in range(len(dataset)):
            samples.append(dataset[randint(0, len(dataset) - 1)].copy())
        predict = predict(dataset, features, classifier) 
        predictions.append(predict)

        err = 0
        for i in range(dataset):
            if predictions[i] == True:
                err += 1
        vote = (1 / 2) * math.log((1 - err) / err)
        votes.append(vote)
    
    sum = 0
    for i in range(round):
        if predictions[i] == True:
            sum += votes[i]
        else:
            sum += -(votes[i])
    
    if np.sign(sum) == 1:
        return True
    else:
        return False

def randomforest(dataset, features, round):
    pass