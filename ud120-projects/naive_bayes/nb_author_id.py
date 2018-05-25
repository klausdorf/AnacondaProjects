#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import nltk
import numpy
import scipy
import sklearn
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
clf = GaussianNB()
t0 = time()
clf.fit (features_train,labels_train)
print ("training time: %s S" %round(time()-t0, 3))
t0 = time()
test_results = clf.predict(features_test)
print ("test time: %s S" %round(time()-t0, 3))
acc = clf.score(features_test,labels_test)
print (acc)
#########################################################