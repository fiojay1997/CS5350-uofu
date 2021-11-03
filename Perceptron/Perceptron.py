import numpy as np
import pandas as pd


def perceptron(data, y, rate=0.01, T=10):
    weights_list = []
    weights = [0] * (len(data[0]))
    for _ in range(T):
        x = data.copy()
        np.random.shuffle(x)
        for i, xi in enumerate(x):
            pred = np.dot(weights, xi)
            if pred * y[i][0] <= 0:
                new_weights = []
                for j in range(len(weights)):
                    new_weights.append(weights[j] + rate * y[i][0] *xi[j])
                if set(new_weights) != set(weights):
                    weights = new_weights
                    weights_list.append(new_weights)
    return weights_list, weights


def voted_perceptron(data, y, rate=0.01, T=10):
    weights_list = []
    weights = [0] * (len(data[0]))
    for _ in range(T):
        x = data.copy()
        np.random.shuffle(x)
        for i, xi in enumerate(x):
            pred = np.dot(weights, xi)
            if pred * y[i][0] <= 0:
                new_weights = []
                for j in range(len(weights)):
                    new_weights.append(weights[j] + rate * y[i][0] *xi[j])
                if set(new_weights) != set(weights):
                    weights = new_weights
                    weights_list.append((new_weights, 1))
            else:
               weights_list[-1] = (weights, weights_list[-1][1] + 1)
    return weights_list, weights


def average_perceptron(data, y, rate=0.01, T=10):
    weights_list = []
    weights = [0] * (len(data[0]))
    average = [0] * (len(data[0]))
    
    for _ in range(T):
        x = data.copy()
        np.random.shuffle(x)
        for i, xi in enumerate(x):
            pred = np.dot(weights, xi)
            if pred * y[i][0] <= 0:
                new_weights = []
                for j in range(len(weights)):
                    new_weights.append(weights[j] + rate * y[i][0] *xi[j])
                if set(new_weights) != set(weights):
                    weights = new_weights
                    weights_list.append(new_weights) 
                average += weights
    return weights_list, weights


def get_perceptron_err(x, y, weights):
    pred = np.sign(np.dot(x, weights))
    err = 0
    for i in range(len(pred)):
        if pred[i] != y[i]:
            err += 1
    print('The perceptron has err rate of {}'.format(1 - err / len(y)))


def load_data(filename):
    df = pd.read_csv(filename)
    x = df.iloc[:, :-1].to_numpy().tolist()
    y = df.iloc[:, -1:].to_numpy().tolist()
    return x, y


if __name__ == '__main__':
    x, y = load_data('bank-note/train.csv')
    testx, testy = load_data('bank-note/test.csv')
    weights_list, weights = perceptron(x, y)
    voted_weights_list, voted_weights = voted_perceptron(x, y)
    average_weights_list, average_weights = average_perceptron(x, y)
    get_perceptron_err(testx, testy, weights)
    get_perceptron_err(testx, testy, voted_weights)
    get_perceptron_err(testx, testy, average_weights)

    with open('StandardPerceptronWeightVectors.txt', 'w') as f:
        for weights in weights_list:
            f.write('{}\n'.format(weights))

    with open('VotedPerceptronWeightVectors.txt', 'w') as f:
        for vweights in voted_weights_list:
            f.write('{}\n'.format(vweights))

    with open('AveragePerceptronWeightVectors.txt', 'w') as f:
        for aweights in average_weights_list:
            f.write('{}\n'.format(aweights))