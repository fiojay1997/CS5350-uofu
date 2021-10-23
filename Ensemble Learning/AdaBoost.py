import numpy as np

class AdaBoost:
    def __init__(self, round, dataset, labels):
        self.round = round
        self.dataset = dataset
        self.labels = labels
        self.data_amount = len(dataset)
        self.feature_amount = len(labels)
        self.weight = [1 / self.data_amount] * self.data_amount
        self.classifiers = []
        self.alpha = []
    
    def load_data(self, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(line.strip().split(','))
        return data
    

