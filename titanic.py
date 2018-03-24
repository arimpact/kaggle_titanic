import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

train_data_object = csv.reader(open('train.csv'))
test_data = csv.reader(open('test.csv'))

train_data = []
labels = []
y = []

del_indexes = [0,1,3,8,10]

count = 0

for x in train_data_object:
    if count == 0:
        x = [i for j, i in enumerate(x) if j not in del_indexes] 
    else:
        x = ['0' if v == '' else v for v in x]
        labels.append(int(x[1]))

        if x[4] == 'male':
            x[4] = 1
        elif x[4] == 'female':
            x[4] = 2
        else:
            x[4] = 0
        if x[11] == 'C':
            x[11] = 1
        elif x[11] == 'Q':
            x[11] = 2
        elif x[11] == 'S':
            x[11] = 3
        else:
            x[11] = 0

        x = [i for j, i in enumerate(x) if j not in del_indexes] 
#         print(x)
        y = []
        for i in x:
            y.append(float(i))
        train_data.append(y)

    count += 1

y = []
test = []
# for x in train_data:
#     y.append(x[1])
#     y.append(x[2])
#     test.append(y)
#     y = []

# scaler = MinMaxScaler()
# scaler.fit(test)

# test = scaler.transform(test)
# count = 0 
# for x in test:
#     train_data[count][1] = x[0] + x[1]
#     del train_data[count][2]
#     count += 1
    
# print(train_data)

train_data = np.array(train_data)


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train_data, labels, test_size=0.3, random_state=42)
        

    
# clf = svm.SVC()
# clf = tree.DecisionTreeClassifier()
# clf = GaussianNB()
# clf = clf.fit(features_train, labels_train)

for x in features_train:
    print(x)
# print(features_train)
# print(labels_train)
# pred = clf.predict(features_test)

# # print(pred)
# # print(labels_test)
# print(accuracy_score(labels_test, pred))
# print(precision_score(labels_test, pred))
# print(recall_score(labels_test, pred))  



#tree
# 0.75
# 0.696428571429
# 0.702702702703

#bayes
# 0.798507462687
# 0.747826086957
# 0.774774774775




