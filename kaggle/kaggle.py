import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import tree
import csv
import random 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def load_data(filename):
    return pd.read_csv(filename)


def preprocess_numeric(df, column):
    data = df[column].copy()
    k = 2
    w = [1.0*i/k for i in range(k+1)]
    w = data.describe(percentiles = w)[4:4+k+1] 
    w[0] = w[0]*(1-1e-10)
    d2 = pd.cut(data, w, labels = range(k))
    #d2.to_csv('test_new.csv', encoding='utf-8', index=False)
    

def process_missing(df):
    imr = SimpleImputer(missing_values='?', strategy='most_frequent')
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    #pd.DataFrame(imputed_data).to_csv('converted_test.csv')


def mapping_data(df, tf):
    df.workclass = pd.factorize(df.workclass)[0]
    df.marital = pd.factorize(df.marital)[0]
    df.education = pd.factorize(df.education)[0]
    df.occupation = pd.factorize(df.occupation)[0]
    df.relationship = pd.factorize(df.relationship)[0]
    df.race = pd.factorize(df.race)[0]
    df.sex = pd.factorize(df.sex)[0]
    df.country = pd.factorize(df.country)[0]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    tf.workclass = pd.factorize(tf.workclass)[0]
    tf.marital = pd.factorize(tf.marital)[0]
    tf.education = pd.factorize(tf.education)[0]
    tf.occupation = pd.factorize(tf.occupation)[0]
    tf.relationship = pd.factorize(tf.relationship)[0]
    tf.race = pd.factorize(tf.race)[0]
    tf.sex = pd.factorize(tf.sex)[0]
    tf.country = pd.factorize(tf.country)[0]
     
    ad = AdaBoostClassifier(n_estimators=10000)
    ad.fit(x, y)
    predict = ad.predict(tf)

    header = ['ID', 'Prediction'] 
    ids = []
    total = 23843
    for i in range(1, total):
        ids.append(i)
    
    with open('result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(total - 1):
            res = []
            res.append(ids[i])
            res.append(predict[i])
            writer.writerow(res)


def predict(x, y, test_x):
    dt = tree.DecisionTreeClassifier()
    clf = dt.fit(x, y)
    return clf.predict(test_x)


def write_result():
    header = ['ID', 'Prediction']
    ids = []
    predictions = []
    total = 23843
    for i in range(1, total):
        ids.append(i)
        ran = random.randint(0, 1)
        predictions.append(ran)

    with open('result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(total - 1):
            res = []
            res.append(ids[i])
            res.append(predictions[i])
            writer.writerow(res)


def adaboost(data, target, test):
    ad = AdaBoostClassifier(n_estimators=1000)
    print('fitting')
    ad.fit(data, target)



if __name__ == '__main__':
    df = load_data('converted.csv')
    tdf = load_data('converted_test.csv')
    # features = ['age', 'workclass', 'fnlwgt', 'education', 'marital.status', 
    # 'occupation', 'relationship', 'race', 'sex', 'hour.per.week', 
    # 'native.country', 'income>50K']
    # x = df.iloc[:, -1:]
    # y = df.iloc[:, :-1]
    # test_x = tdf.iloc[:, -1:]
    # #preprocess_numeric(tdf)
    # mapping_data(tdf)
    #predict(x, y, test_x)
    mapping_data(df, tdf) 