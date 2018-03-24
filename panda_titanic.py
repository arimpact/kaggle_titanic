
import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

#['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

from sklearn.preprocessing import Imputer

df = pd.read_csv('train.csv', usecols=['Survived', 'Fare', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Pclass' ])
# print(df)
sex = {'male': 1, 'female': 2}
nan = {'NaN':0}
embarked = {'C': 1, 'Q': 2, 'S': 3}

df['Sex'] = df['Sex'].map(sex)
df['Embarked'] = df['Embarked'].map(embarked)
imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
df[['Age']] = imputer.fit_transform(df[['Age']])

def replace_most_common(x):
    if pd.isnull(x):
        return 3
    else:
        return x

df['Embarked'] = df['Embarked'].map(replace_most_common)

labels = df['Survived']
del df['Survived']

# print(df)
from sklearn.preprocessing import StandardScaler
# # # features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


train_data = df.loc[:].values
labels = labels.loc[:].values

# print(train_data)
# # # Separating out the target
# x = df.loc[:].values
# y = labels.loc[:].values
# # # Standardizing the features
# x = StandardScaler().fit_transform(x)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit_transform(x)

# pca.fit(x)
# print(pca.explained_variance_ratio_)  
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train_data, labels, test_size=0.3, random_state=42)
        

    
# clf = svm.SVC()
# clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=10)
# clf = GaussianNB()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(df.columns, clf.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'importance'})
# print(importances.sort_values(by='importance', ascending = False))

# print(importances)
# print(clf.feature_importances_)
    
# print(clf)
# print(labels_train)
# pred = clf.predict(features_test)

# print(pred)
# print(labels_test)
print(accuracy_score(labels_test, pred))
print(precision_score(labels_test, pred))
print(recall_score(labels_test, pred))  
