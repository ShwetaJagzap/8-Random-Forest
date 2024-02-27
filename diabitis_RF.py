#______________________
# Diabetes 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#2.Load the Dataset
df=pd.read_csv("C:/MLAlgo/Diabetes.csv")


# EDA
df.head(5)
df.tail(5)
df.isnull().sum()
df.info()
df.columns
df.describe()


le=LabelEncoder()
df[" Class variable"]=le.fit_transform(df[" Class variable"])
df[" Class variable"].value_counts()
df.head()

inputs=df.drop([" Class variable"],axis=1)
target=df[" Class variable"]

x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred

acc=accuracy_score(y_test, y_pred)
acc

