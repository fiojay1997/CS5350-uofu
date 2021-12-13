import random
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def load_data(filename):
    df = pd.read_csv(filename).to_numpy().tolist()
    for data in df:
        data[-1] = int(data[-1])
        if data[-1] == 0:
            data[-1] = -1
    return df


def get_rate_1(epoch, learn_rate=0.001, a=0.01):
    return learn_rate / (1 + (learn_rate / a) * epoch)


class BPNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, init):
        self.input_n = input_dim
        self.hidden_n = hidden_dim
        self.output_n = output_dim
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        self.init = init
        if self.init == 0:
            self.input_layer = np.zeros(self.input_n)
            self.hidden_layer = np.zeros(self.hidden_n)
            self.output_layer = np.zeros(self.output_n)
            self.input_weights = np.zeros(
                (self.input_n, self.hidden_n))
            self.output_weights = np.zeros(
                (self.hidden_n, self.output_n))
        else:
            self.input_layer = np.random.rand(self.input_n)
            self.hidden_layer = np.random.rand(self.hidden_n)
            self.output_layer = np.random.rand(self.output_n)
            self.input_weights = np.random.rand(
                (self.input_n, self.hidden_n))
            self.output_weights = np.random.rand(
                (self.hidden_n, self.output_n))

        self.input_correction = np.random.randn(self.input_n, self.hidden_n)
        self.output_correction = np.random.randn(self.hidden_n, self.output_n)

    def forward(self, inputs):
        for i in range(self.input_n - 1):
            self.input_layer[i] = inputs[i]
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_layer[i] * self.input_weights[i][j]
            self.hidden_layer[j] = sigmoid(total)
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_layer[j] * self.output_weights[j][k]
            self.output_layer[k] = sigmoid(total)
        return self.output_layer[:]

    def back_propagate(self, case, label, rate, correct):
        self.forward(case)
        output_deltas = np.zeros(self.output_n)
        for o in range(self.output_n):
            error = label[o] - self.output_layer[o]
            output_deltas[o] = sigmoid_derivative(self.output_layer[o]) * error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_layer[h]) * error
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_layer[h]
                self.output_weights[h][o] += rate * change + \
                    correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_layer[i]
                self.input_weights[i][h] += rate * change + \
                    correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_layer[o]) ** 2
        return error

    def train(self, cases, labels, T=10000, correct=0.1):
        for j in range(T):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                learn_rate = get_rate_1(i)
                error += self.back_propagate(case, label, learn_rate, correct)

    def test(self, filename):
        x, y = load_data(filename)
        rate = get_rate_1(10000)
        self.forward(x, y, rate, 0)
        self.train(x, y, 10000)
        for case in x:
            print(self.forward(case))


if __name__ == '__main__':
    nn = NN()
    nn.test()
