# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:49:57 2021

@author: ttommasini
"""

import pickle
from get_best_model import refit_by_importance, get_report_df

with open('importance_selector.pickle', 'rb') as file:
    imp = pickle.load(file)
    
