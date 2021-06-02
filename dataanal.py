# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:27:41 2021

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

df=pd.read_csv("data.csv")
print(df)

f=["x","y","z"]
X=df[f]
print(X)

Y=df["apot"]
Y2=df["onoff"]

print(Y)



#cc=np.corrcoef(np.array([df["x"],df["y"], df["z"], df["apot"]]))
#print(cc)

Xtrain=X.iloc[0:60]
Ytrain=Y.iloc[0:60]
Y2train=Y2.iloc[0:60]

reg = linear_model.LinearRegression()
reg.fit(Xtrain,Ytrain)

print(reg.coef_)
y=reg.predict([[9,1,7]]  )


Xtest=X.iloc[61:99]
Ytest=Y.iloc[61:99]
Y2test=Y2.iloc[61:99]

print(reg.score(Xtest,Ytest))


regr = MLPRegressor(random_state=1,hidden_layer_sizes=(10,20), max_iter=1000).fit(Xtrain, Ytrain)
regr.fit(Xtrain,Ytrain)

print(regr.score(Xtest,Ytest))

reg = linear_model.Ridge(alpha=.5)
reg.fit(Xtrain,Ytrain)

print(reg.coef_)
print(reg.score(Xtest,Ytest))



regr = MLPClassifier(random_state=1,hidden_layer_sizes=(10,), max_iter=1000)
regr.fit(Xtrain,Y2train)

print(regr.score(Xtest,Y2test))
print(regr.predict([[6,3,2]]))

from sklearn.tree import DecisionTreeClassifier


regr2 = DecisionTreeClassifier()
regr2.fit(Xtrain,Y2train)

print(regr2.score(Xtest,Y2test))
