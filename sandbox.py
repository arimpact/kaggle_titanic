import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline

# machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

titanic_df = pd.read_csv("train.csv")

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

rand_1 = np.random.randint(10 - 3, 10 + 3, size = 2)

print(titanic_df["Age"].isnull().sum())

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# print(titanic_df['Age'])