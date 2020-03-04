# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:34:19 2020

@author: Asus
"""

import pandas as pd
import numpy as np

bank = pd.read_csv("Churn_Modelling.csv")

X = bank.iloc[:, 3:13].values
Y = bank.iloc[:,13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#input layer

classifier.add(Dense(6,input_shape=(11,),activation = "relu"))

# hidden layer

classifier.add(Dense(6,activation = "relu"))

# output layer

classifier.add(Dense(1,activation = "sigmoid"))


#compile the classifier

classifier.compile(optimizer = "adam",loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit classifier to the input

classifier.fit(X_train,y_train,batch_size = 8,epochs = 70)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

y_test = classifier.predict()

