import pandas as pd
import numpy as np
from scipy import optimize
from copy import deepcopy


def build_svm(dataset, C, y, kernel):
    dataset_copy = deepcopy(dataset)
    x = dataset_copy[:, :-1]
    y = dataset_copy[:, -1]

    cons = [{
        'type': 'eq',
        'fun': cons_ay,
        'agrs': y
    }]

    bounds = [(0, C)] * len(x)

    opt = optimize.minimize(
        minimize,
        x0=np.zeros(len(x)),
        args=(x, y, kernel, y),
        method="SLSQP",
        constraints=cons,
        bounds=bounds)

    res = opt.x


def gaussian(x, y):
    return np.exp(-np.sqrt(np.linalg.norm(x - y)) ** 2)


def cons_ay(a, y):
    return np.dot(a, y)


def linear(dataset):
    return np.dot(dataset, np.transpose(dataset))


def minimize(a, x, y, kernel, gamma):
    aout = np.outer(a, a)
    yout = np.outer(y, y)

    if kernel == 'gaussian':
        xk = gaussian(x, gamma)
    else:
        xk = linear(x)

    return 0.5 * (aout * yout * xk).sum() - a.sum()


def load_data(file):
    df = pd.read_csv(file).to_numpy().tolist()
    for data in df:
        data[-1] = int(data[-1])
        if data[-1] == 0:
            data[-1] = -1
    return df
