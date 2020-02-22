# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:27:37 2020

@author: Asus
"""

import tarfile

tarfile.open("D:\\Machine Learning\\Data sets\\Movie review\\movie_review.tar.gz", "r:gz").extractall()

# import libraries
import pandas as pd
from sklearn.utils import shuffle
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model.logistic import LogisticRegression

nltk.download("stopwords")
#import pyprind

movie = pd.DataFrame()

basefolder = "tokens"
#pbar = pyprind.ProgBar(700)
labels = {"pos":1, "neg":0}

# loadng the text files in the dataframe
for s in ("pos","neg"):
    path = os.path.join(basefolder,s)
    for file in os.listdir(path):

        txt = open(os.path.join(path,file),"r").read()
    
        movie = movie.append([[txt,labels[s]]],ignore_index = True)
    
    #pbar.update()

movie.columns = ["review","sentiment"]

movie = shuffle(movie,random_state = 0)

movie.head()

# cleaning the text
def textclean(col):
    
    return re.sub('^[A-Za-z0-9]+()\'', ' ', col)

movie["review"] = movie["review"].apply(textclean)

movie.loc[50][0]

porter = PorterStemmer()

def tokenize_porter(text):
    return [porter.stem(word) for word in text.split()]

stop = stopwords.words("english") 

vectorizer = TfidfVectorizer(stop_words= "english",ngram_range=(1,1))

X = vectorizer.fit_transform(movie["review"]) 
print(vectorizer.get_feature_names)
#print(vectorizer.get_feature_names())
train_x, test_x, train_y, test_y = train_test_split(X, movie["sentiment"],test_size= 0.2,random_state=42)

  

print(train_x.shape, train_y.shape)
clf = LogisticRegression()
clf.fit(train_x,train_y)

#predict result
print(clf.predict(test_x ))


#crossval score
scores = cross_val_score(clf, test_x, test_y, cv=5)

acc = scores.mean()
print ("Accuracy: %0.2f percent" % (acc *100))