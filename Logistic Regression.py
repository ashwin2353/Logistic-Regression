# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:06:53 2022

@author: ashwi
"""
import pandas as pd
df = pd.read_csv("bank-full.csv",sep=";")
df
df.head()
list(df)
df.dtypes
df.shape
#=================================================================

Y = df.iloc[:,16:]
X = df.iloc[:,0:16]

#=================================================================
# Data visualization

import matplotlib.pyplot as plt
plt.figure(figsize=(30,10))
plt.scatter(X["age"],X["day"],color = "black")
plt.show()

plt.figure(figsize=(30,7))
plt.scatter(X["age"],X["balance"],color = "black")
plt.show()

plt.figure(figsize=(30,7))
plt.scatter(X["age"],X["duration"],color = "black")
plt.show()
#=====================================================================
# Boxplot

df.boxplot(column="age",vert = False)
import numpy as np
Q1 = np.percentile(df["age"],25)
Q1
Q2 = np.percentile(df["age"],50)
Q3 = np.percentile(df["age"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["age"]<LW) | (df["age"]>UW)]
len(df[(df["age"]<LW) | (df["age"]>UW)])
# out layers are 487
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))
len(df[(df["age"]<LW) | (df["age"]>UW)])
# out layers zero


df.boxplot(column="balance",vert = False)
import numpy as np
Q1 = np.percentile(df["balance"],25)
Q1
Q2 = np.percentile(df["balance"],50)
Q3 = np.percentile(df["balance"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["balance"]<LW) | (df["balance"]>UW)]
len(df[(df["balance"]<LW) | (df["balance"]>UW)])
# out layres are 4729
df["balance"]=np.where(df["balance"]>UW,UW,np.where(df["balance"]<LW,LW,df["balance"]))
len(df[(df["balance"]<LW) | (df["balance"]>UW)])
# out lauers are became zreo


df.boxplot(column="campaign",vert = False)
import numpy as np
Q1 = np.percentile(df["campaign"],25)
Q1
Q2 = np.percentile(df["campaign"],50)
Q3 = np.percentile(df["campaign"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["campaign"]<LW) | (df["campaign"]>UW)]
len(df[(df["campaign"]<LW) | (df["campaign"]>UW)])
# out layres are 3064
df["campaign"]=np.where(df["campaign"]>UW,UW,np.where(df["campaign"]<LW,LW,df["campaign"]))
len(df[(df["campaign"]<LW) | (df["campaign"]>UW)])
# out layres are beame zero


df.boxplot(column="previous",vert = False)
import numpy as np
Q1 = np.percentile(df["previous"],25)
Q1
Q2 = np.percentile(df["previous"],50)
Q3 = np.percentile(df["previous"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["previous"]<LW) | (df["previous"]>UW)]
len(df[(df["previous"]<LW) | (df["previous"]>UW)])
# out layres 8257
df["previous"]=np.where(df["previous"]>UW,UW,np.where(df["previous"]<LW,LW,df["previous"]))
len(df[(df["previous"]<LW) | (df["previous"]>UW)])
# out layres 0


df.boxplot(column="duration",vert = False)
import numpy as np
Q1 = np.percentile(df["duration"],25)
Q1
Q2 = np.percentile(df["duration"],50)
Q3 = np.percentile(df["duration"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[(df["duration"]<LW) | (df["duration"]>UW)]
len(df[(df["duration"]<LW) | (df["duration"]>UW)])
# out layres are 3235
df["duration"]=np.where(df["duration"]>UW,UW,np.where(df["duration"]<LW,LW,df["duration"]))
len(df[(df["duration"]<LW) | (df["duration"]>UW)])
#out layres 0



df.boxplot(column="day",vert = False)
import numpy as np
Q1 = np.percentile(df["day"],25)
Q1
Q2 = np.percentile(df["day"],50)
Q3 = np.percentile(df["day"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[df["day"]<LW]
df[df["day"]>UW]
len(df[(df["day"]<LW) | (df["day"]>UW)])
# out layres 0



df.boxplot(column="pdays",vert = False)
import numpy as np
Q1 = np.percentile(df["pdays"],25)
Q1
Q2 = np.percentile(df["pdays"],50)
Q3 = np.percentile(df["pdays"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)
df[df["pdays"]<LW]
df[df["pdays"]>UW]
len(df[(df["pdays"]<LW) | (df["pdays"]>UW)])
# out layres are 8257
df["pdays"]=np.where(df["pdays"]>UW,UW,np.where(df["pdays"]<LW,LW,df["pdays"]))
len(df[(df["pdays"]<LW) | (df["pdays"]>UW)])
# outlayres are became 0


df.groupby("job").size()
t1 = df.groupby("job").size()
t1.plot(kind = "bar")

#==============================================================================
# Data Transformation

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X['job'] = LE.fit_transform(df['job'])
X['marital'] = LE.fit_transform(df['marital'])
X['education'] = LE.fit_transform(df['education'])
X['default'] = LE.fit_transform(df['default'])
X['housing'] = LE.fit_transform(df['housing'])
X['loan'] = LE.fit_transform(df['loan'])
X['contact'] = LE.fit_transform(df['contact'])
X['month'] = LE.fit_transform(df['month'])
X['poutcome'] = LE.fit_transform(df['poutcome'])
X['y'] = LE.fit_transform(df['y'])

X.dtypes

#=============================================================================
# Data partision

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,random_state=33)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#=========================================================================
# Model fittling

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

#===========================================================================
# Metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print("Training accuracy",accuracy_score(Y_train,Y_pred_train).round(3))
print("Test accuracy",accuracy_score(Y_test,Y_pred_test).round(3))

cm =confusion_matrix(Y_train, Y_pred_train)
cm











