# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:20:30 2020

@author: PC
"""

from mod_plots import multiclass_roc_curve, plot_confusion_matrix, multiclass_precision_recall
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as mtr

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# shuffle and split training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

smc = svm.SVC(kernel='linear', probability=True, random_state=10)

fig, ax = plt.subplots(figsize=(12,8))

#### Multilabel ROC ####

rc = multiclass_roc_curve(x_train, x_test, y_train, y_test,
                          n_classes=n_classes, standardize=True,
                          model=smc, title='Support Vector Classifier', ax=ax)


#### Multiclass confusion matrix ####

X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

smcc = svm.SVC(kernel='linear', probability=True, random_state=10, decision_function_shape='ovr')
    
fit = smcc.fit(x_train,y_train)
pred = smcc.predict(x_test)
prob = smcc.predict_proba(x_test)

cm = mtr.confusion_matrix(y_test, pred) #, labels = ['clas 0', 'clas 1', 'clas 2'])

fig, ax = plt.subplots(figsize=(12,8))

pcm = plot_confusion_matrix(y_test, pred, target_names = ['Setosa', 'Versicolor', 'Virginica'],
                            normalize = False, multiclass=True, ax=ax)

#### Precision Recall Curve ####

fig, ax = plt.subplots(figsize=(12,8))

X = iris.data
y = iris.target

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

rc = multiclass_precision_recall(x_train, x_test, y_train, y_test,
                                 n_classes=n_classes, standardize=True,
                                 model=smc, title='Support Vector Machine', ax=ax)
