# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:11:11 2020

@author: PC
"""

from get_best_model import get_best_model, get_report_df
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

rf = RandomForestClassifier(random_state=10)
gb = GradientBoostingClassifier(random_state=10)

pr = [{
    'bootstrap': [True,False],
    'n_estimators': np.arange(10, 100, 10)
    },{
     'n_estimators': np.arange(10,100,10)
       }]
      
md = [rf, gb] 

bm = get_best_model(X,y,random_state=10, models=md, params=pr, score='roc_auc')

rp = get_report_df(bm)