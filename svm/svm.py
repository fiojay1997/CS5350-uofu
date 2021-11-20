import pandas as pd
import numpy as np
import copy


def build_svm(dataset, C, get_learn_rate, T=100):
    dataset_copy = copy.deepcopy(dataset)
    w = [0] * len(dataset_copy[0])
    for t in range(T):
        np.random.shuffle(dataset_copy)
        for data in dataset_copy:
            w0 = w[:-1]
            yi = data[-1]
            xi = data[:-1]
            xi.append(1)
            learn_rate = get_learn_rate(t)
            w1 = w[:-1]
            w1.append(0)
            if np.dot(w, data) * yi <= 1:
                left = list_sub(w, [element * learn_rate for element in w1])
                right = learn_rate * C * len(dataset_copy) * yi
                w = list_add(left, [element * right for element in xi])
            else:
                w = [element * (1 - learn_rate)
                     for element in w0]
                w.append(w[-1])
    return w


def svm_predict(testdata, w):
    value = np.dot(np.transpose(w[:-1]), testdata[:-1]) + w[-1]
    return np.sign(value)


def load_data(file):
    df = pd.read_csv(file).to_numpy().tolist()
    for data in df:
        data[-1] = int(data[-1])
        if data[-1] == 0:
            data[-1] = -1
    return df


def svm_test(traindata, testdata, C, get_rate):
    df = load_data(traindata)
    test_df = load_data(testdata)
    svm_w = build_svm(df, C, get_rate)
    print('Model parameter learned is ', svm_w)

    res = []
    correct = []
    for data in test_df:
        res.append(int(svm_predict(data, svm_w)))
    for data in test_df:
        correct.append(data[-1])

    err = 0
    for i in range(len(correct)):
        if res[i] != correct[i]:
            err += 1
    print(err / len(correct))


def get_rate_1(epoch, learn_rate=0.001, a=0.01):
    return learn_rate / (1 + (learn_rate / a) * epoch)


def get_rate_2(epoch, learn_rate=0.001, a=0.001):
    return learn_rate / (1 + epoch)


def list_sub(l1, l2):
    res = []
    for i in range(len(l1)):
        res.append(l1[i] - l2[i])
    return res


def list_add(l1, l2):
    res = []
    for i in range(len(l1)):
        res.append(l1[i] + l2[i])
    return res


def main():
    print('Using schedule 1')
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 100 / 873, get_rate_1)
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 500 / 873, get_rate_1)
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 700 / 873, get_rate_1)
    print('Using schedule 2')
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 100 / 873, get_rate_2)
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 500 / 873, get_rate_2)
    svm_test('bank-note/train.csv', 'bank-note/test.csv', 700 / 873, get_rate_2)


if __name__ == '__main__':
    main()
