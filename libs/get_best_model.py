# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:25:40 2020

@author: Tobia
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def get_best_model(x, 
                   y,
                   test_size = .3,
                   random_state = None,
                   normalize = True,
                   goal = 'Classification',
                   cv = None,
                   models = [],
                   params = [],
                   n_iter = 100,
                   score = 'neg_mean_squared_error'):
    
    if normalize and goal == 'Classification':
    
        sd = StandardScaler()
        
        x = sd.fit_transform(x)
        
    elif normalize and goal == 'Regression':
        
        sd = StandardScaler()
        
        x = sd.fit_transform(x)
        y = sd.fit_transform(y.reshape(-1, 1))
        
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, 
                                                        random_state = random_state)
    
    best_models = {}
    preds = {}
    prob = {}
    for n, mod in enumerate(models):
        gr = RandomizedSearchCV(mod, params[n], random_state = random_state, cv = cv,  
                                n_iter = n_iter, 
                                scoring = score)
        _ = gr.fit(x_train, y_train)
        pred = gr.best_estimator_.predict(x_test)
        preds[gr.best_estimator_] = gr.best_estimator_.predict(x_test)
        
        if goal == 'Classification':
            cr = classification_report(y_test, pred, output_dict = True)
            best_models[gr.best_estimator_] = cr
            prob[gr.best_estimator_] = gr.best_estimator_.predict_proba(x_test)
            
        elif goal == 'Regression':
            scr = mean_squared_error(y_test, pred)
            best_models[gr.best_estimator_] = scr
        
    return best_models, preds, prob
        
def get_report_df(rep):
    
    rp = {}
    for key, val in zip(rep.keys(), rep.values()):
        rp[str(key).split('(')[0]] = val
        
    crs = []
    models = []
    for mds, cr in rp.items():
        models.append(mds)
        crs.append(pd.DataFrame(cr))
        
    ind = np.array([[mod]*crs[0].shape[0] for mod in models]).flatten()
    
    full_crs = pd.concat(crs, axis=0)
    
    full_crs['Model'] = ind
    full_crs = full_crs.reset_index().rename({'index':'score'}, axis=1)
    
    max_classes = [str(cl) for cl in range(0,100)]
    
    vals = [num_class for num_class in full_crs.columns if num_class in max_classes]
    cl_scores = pd.pivot_table(full_crs, index='Model', columns='score', values=vals)
    
    acc = full_crs.groupby('Model').agg({'accuracy':'mean'})
    cl_scores['accuracy'] = acc['accuracy']
    
    return cl_scores

    
    
        
        