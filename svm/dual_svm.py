import numpy as np
import pandas as pd
import scipy.optimize


def build_dual_svm(dataset, alpha, C, kernel):
    def cons(alpha, y):
        return np.dot(alpha, y).sum()

    def opt_func(alpha, x, y):
        newAlpha = np.outer(alpha, alpha)
        newY = np.outer(y, y)
        if kernel == 'linear':
            newX = linear_func(x)
        else:
            newX = gaussian(x)
        return (newX * newY * newAlpha).sum() * 0.5 - alpha.sum()

    x = np.array([row[:-1] for row in dataset])
    y = np.array([row[-1] for row in dataset])

    bound = scipy.optimize.Bounds(0, C)
    res = scipy.optimize.minimize(fun=opt_func, x0=np.zeros(len(dataset)), args=(
        x, y), method='SLSQP', bounds=bound, constraints={'type': 'eq', 'fun': cons, 'args': [y]})

    a = res.x
    print(a)
    w = np.dot(np.dot(a, y), kernel(x, x.transpose*()))
    b = y - np.dot(w.transpose(), x)
    return w, b


def linear_func(x):
    return x.dot(np.transpose(x))


def gaussian(x, g):
    return np.exp(-1 * x / g)


def load_data(file):
    df = pd.read_csv(file).to_numpy().tolist()
    for data in df:
        data[-1] = int(data[-1])
        if data[-1] == 0:
            data[-1] = -1
    return df


def predict(test_x, test_y, w, g):
    predict = 0
    for i in range(test_x):
        predict += w[i] * test_y[i](-1 * np.linalg(test_x[i]) ** 2 / g)

    return np.sign(predict)


if __name__ == '__main__':
    df = load_data('bank-note/train.csv')
    test_df = load_data('bank-note/test.csv')

    print(build_dual_svm(df, 0.1, 100/873, 'linear'))
