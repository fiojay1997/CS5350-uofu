from os import replace
import pandas as pd
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cv
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as TTS
from sklearn import preprocessing


def load_data(filename):
    return pd.read_csv(filename)


def convert_missing_val(df):
    df['workclass'] = df['workclass'].replace('?', np.nan)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    df['native'] = df['native'].replace('?', np.nan)


def transform_data(df):
    le = preprocessing.LabelEncoder()
    le.fit(df.workclass)
    df['workclass'] = le.transform(df.workclass)
    le.fit(df.education)
    df['education'] = le.transform(df.education)
    le.fit(df.marital)
    df['marital'] = le.transform(df.marital)
    le.fit(df.occupation) 
    df['occupation'] = le.transform(df.occupation)
    le.fit(df.relationship)
    df['relationship'] = le.transform(df.relationship)
    le.fit(df.race)
    df['race'] = le.transform(df.race)
    le.fit(df.sex)
    df['sex'] = le.transform(df.sex)
    le.fit(df.native)
    df['native'] = le.transform(df.native)


def get_data_distribution(df):
    print('Data distribution: ------ ')
    print('work class data: ')
    print(df['workclass'].value_counts())
    print('--------')
    print('fntwgt data: ')
    print(df['fnlwgt'].value_counts())
    print('--------')
    print('eudcation data:')
    print(df['education'].value_counts())
    print('--------')
    print('education-level data:')
    print(df['education-level'].value_counts())
    print('--------')
    print('marital data:')
    print(df['marital'].value_counts())
    print('--------')
    print('occupation data:')
    print(df['occupation'].value_counts())
    print('--------')
    print('relationship data:')
    print(df['relationship'].value_counts())
    print('--------')
    print('race data:')
    print(df['race'].value_counts())
    print('--------')
    print('sex data:')
    print(df['sex'].value_counts())
    print('--------')
    print('capital-gain data:')
    print(df['capital-gain'].value_counts())
    print('--------')
    print('capital data:')
    print(df['capital'].value_counts())
    print('--------')
    print('hours data:')
    print(df['hours'].value_counts())
    print('--------')
    print('native data:')
    print(df['native'].value_counts())
 

def check_relation(df):
    capital = df['capital']
    income = df['income']
    zero_count = 0
    capital_count = 0
    zero_with_less = 0

    capital_with_less = 0
    for i in range(len(capital)):
        if capital[i] == 0:
            zero_count += 1
        else:
            capital_count += 1

        if capital[i] == 0 and income[i] == 0:
            zero_with_less += 1
        
        if capital[i] != 0 and income[i] == 1:
            capital_with_less += 1

    print("There are ", zero_count, " people with 0 capital")
    print("For the people with no capital gain, there are ", zero_with_less / zero_count, " percent people with less than 50k income")
    
    print("There are ", capital_count, " people with capital")
    print("For the people with capital gain, there are ", capital_with_less / capital_count, " percent people with less than 50k income")


def nn(df, tdf):
    transform_data(df)
    transform_data(tdf)
    #convert_missing_val(df)
    train_x = df.iloc[:, :-1]
    train_y = df.iloc[:, -1]
    tdf_x = tdf.iloc[:, 1:]

    Xtrain, Xtest, Ytrain, Ytest = TTS(train_x, train_y, test_size=0.3, random_state=400)
    dnn = DNN(hidden_layer_sizes=(100,),random_state=400)
    print("cross validation score",cv(dnn, train_x, train_y, cv=5).mean())

    dnn.fit(Xtrain, Ytrain)
    print("DNN socre ", dnn.score(Xtest, Ytest))
    print("number of layers_：",dnn.n_layers_)

    # dnn = DNN(hidden_layer_sizes=(200,),random_state=400)
    # dnn = dnn.fit(Xtrain,Ytrain)
    # print("New DNN socre", dnn.score(Xtest, Ytest))

    # s = []
    # layers = [(100,),(100,100),(100,100,100),(100,100,100,100),(100,100,100,100,100),
    # (100,100,100,100,100,100)]
    # for i in layers:
    #     dnn = DNN(hidden_layer_sizes=(i),random_state=420).fit(Xtrain,Ytrain)
    #     s.append(dnn.score(Xtest,Ytest))
    #     print(i,max(s))

    result = dnn.predict(tdf_x)
    print(result) 

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
            res.append(result[i])
            writer.writerow(res)

if __name__ == '__main__':
    df = load_data('train_final.csv')
    tdf = load_data('test_final.csv')
    # get_data_distribution(df)
    # check_relation(df)
    # print(df.value_counts())
    # convert_missing_val(df)
    # print(df['workclass'].value_counts())
    nn(df, tdf)
    