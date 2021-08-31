# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:35:18 2021

@author: ttommasini
"""

import time
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from get_best_model import cross_val_report

X, Y = make_classification(n_samples = 300)
lr = LogisticRegression()
rf = RandomForestClassifier()
models = [lr, rf]

### NO PARALLELIZATION ###
start = time.time()
prova2 = cross_val_report(X, Y, n_jobs = 1, models = models)
end = time.time()
print(end - start)

### PARALLELIZATION ###

start = time.time()
prova2 = cross_val_report(X, Y, n_jobs = -1, models = models)
end = time.time()
print(end - start)

### INTEL EXTENTION COMPARISON ###

X, Y = make_classification(n_samples = 300000)
rf = RandomForestClassifier(n_jobs = -1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2)

print('WITHOUT INTEL EXTENTION')
start = time.time()
_ = rf.fit(x_train, y_train)
end = time.time()
print(f'Time required to train the model: {end - start}')

from sklearnex import patch_sklearn
patch_sklearn()

rf = RandomForestClassifier(n_jobs = -1)

print('WITH INTEL EXTENTION')
start = time.time()
_ = rf.fit(x_train, y_train)
end = time.time()
print(f'Time required to train the model: {end - start}')










