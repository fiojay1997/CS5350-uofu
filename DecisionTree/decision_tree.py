from math import log
from numpy import log2, square
import operator


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
        ent -= prob * log2(prob)
    return ent


def get_me(dataset):
    label_count, count = get_label_info(dataset)
    return 1 - (max(label_count.values()) / count)


def get_gini(dataset):
    label_count, count = get_label_info(dataset)
    gini = 1.0
    for data in label_count:
        prob = label_count[data] / count
        gini -= square(prob)
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


def predict_cars_err(depth, function):
    dataset = load_data('../car/train.csv')
    labels = ['buying', 'maint', 'doors',
              'persons', 'lug_boot', 'safety', 'label']
    tree = create_tree(dataset, labels, function, depth)
    predict_labels = ['buying', 'maint', 'doors',
                      'persons', 'lug_boot', 'safety']
    success = 0
    for data in dataset:
        if predict(data, predict_labels, tree):
            success += 1
    return 1 - (success / len(dataset))


def predict_cars_test(depth, function):
    dataset = load_data('../car/test.csv')
    labels = ['buying', 'maint', 'doors',
              'persons', 'lug_boot', 'safety', 'label']
    tree = create_tree(dataset, labels, function, depth)
    predict_labels = ['buying', 'maint', 'doors',
                      'persons', 'lug_boot', 'safety']
    success = 0
    for data in dataset:
        if predict(data, predict_labels, tree):
            success += 1
    return 1 - (success / len(dataset))


def predict_banks(depth, function):
    dataset = load_data('../bank/train.csv')
    labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
              'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    tree = create_tree(dataset, labels, 'entropy', depth)
    predict_labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    success = 0
    for data in dataset:
        if predict(data, predict_labels, tree):
            success += 1
    return 1 - (success / len(dataset))


def predict_banks_test(depth, function):
    dataset = load_data('../bank/test.csv')
    labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
              'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    tree = create_tree(dataset, labels, 'entropy', depth)
    predict_labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    success = 0
    for data in dataset:
        if predict(data, predict_labels, tree):
            success += 1
    return 1 - (success / len(dataset))


def predict_banks_with_missing(depth, function):
    dataset = load_data('../bank/test.csv')
    labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
              'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    tree = create_tree(dataset, labels, 'entropy', depth)
    predict_labels = ['age', 'job', 'material', 'education', 'default', 'balance', 'housing'
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    value_counts = {}
    for i in range(len(labels)):
        for data in dataset:
            if data[i] != 'unknown':
                if data[i] not in value_counts and data[i]:
                    value_counts[data[i]] = 1
                else:
                    value_counts[data[i]] += 1
        most_common_value = max(
            value_counts.items(), key=operator.itemgetter(1))[0]
        for data in dataset:
            if data[i] == 'unknown':
                data[i] = most_common_value

    success = 0
    for data in dataset:
        if predict(data, predict_labels, tree):
            success += 1
    return 1 - (success / len(dataset))


if __name__ == '__main__':
    for i in range(7):
        print('Using Gini index on cars train dataset :',
              predict_cars_err(i, 'gini'))
    for i in range(7):
        print('Using Entropy on cars train dataset:',
              predict_cars_err(i, 'entropy'))
    for i in range(7):
        print('Using Majority Error on cars train dataset:',
              predict_cars_err(i, 'me'))

    for i in range(7):
        print('Using Gini index on cars test dataset:',
              predict_cars_test(i, 'gini'))
    for i in range(7):
        print('Using Entropy on cars test dataset:',
              predict_cars_test(i, 'entropy'))
    for i in range(7):
        print('Using Majority Error on cars test dataset:',
              predict_cars_test(i, 'me'))

    # for i in range(17):
    #     print('Using Gini index on bank train dataset :',
    #           predict_banks(i, 'gini'))
    # for i in range(17):
    #     print('Using Entropy on cars bank dataset:',
    #           predict_banks(i, 'entropy'))
    # for i in range(17):
    #     print('Using Majority Error on bank train dataset:',
    #           predict_banks(i, 'me'))

    # for i in range(17):
    #     print('Using Gini index on bank test dataset:',
    #           predict_banks_test(i, 'gini'))
    # for i in range(17):
    #     print('Using Entropy on bank test dataset:',
    #           predict_banks_test(i, 'entropy'))
    # for i in range(17):
    #     print('Using Majority Error on bank test dataset:',
    #           predict_banks_test(i, 'me'))
