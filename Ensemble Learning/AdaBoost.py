import numpy as np

def adaboost(dataset, data_labels, round):
    classifier = create_tree(dataset, data_labels, 'gini', 1)
    length = len(dataset)
    weight = 1 / length
    predictions = [np.zeros(length)]

    for _ in range(round):
        predict = dt.predict(dataset, data_labels, classifier)
        miss = [int(x) for x in (predict != data_labels)]
        err = np.dot(weight, miss)
        alpha = 0.5 * np.log((1 - err) / float(err))
        filter_miss = [x if x == 1 else - 1 for x in miss]
        weight = np.multiply(weight, np.exp([float(x) * alpha for x in filter_miss]))
        weight = sum(weight)
        prediction = [i if x == 1 else -1 for x in predict]
        predictions = predictions + np.multiply(alpha, prediction)
    
    predictions = (predictions > 0) * 1


