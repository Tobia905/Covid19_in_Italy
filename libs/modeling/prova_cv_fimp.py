# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:59:09 2021

@author: PC
"""

import pickle
import pandas as pd
import numpy as np
from get_best_model import smote_cv_score, refit_by_importance
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def missing_row_sum(df, tresh):
    nan_cols = {}
    for col in df.columns:
        nan_cols[f'{col}_nans'] = df[col].isna()
    nan_cols = pd.DataFrame(nan_cols)
    nan_cols['sum_nan'] = nan_cols.sum(axis = 1)
    return nan_cols[nan_cols.sum_nan.div(nan_cols.shape[1]) >= tresh].index

path = 'C:/Users/PC/Documents/Python Scripts/covid_los_let_prediction/bm_cw_interaction_transf.pickle'
data_path = 'C:/Users/PC/Documents/Python Scripts/covid_los_let_prediction/data_ews_covid.csv'

with open(path, 'rb') as file:
    bm = pickle.load(file)

selected_features = ['eta', 'MCV', 'PLT', 'RBC', 'MCHC', 'MCH', 'RDW', 'HGB', 
                     'HCT', 'LYT', 'MO', 'MOT', 'NE', 'NET', 'LY', 'charleston',
                     'MPV', 'WBC', 'UREB', 'CREB', 'ALTB', 'CPKB', 'GLUB', 'PCRB', 'CALB', 
                     'PTR', 'PTINR', 'PTTR', 'PTTS', 'PTS', 'POTB', 'SODB', 'TROBI', 'AMIB', 
                     'FETB', 'FIBB', 'BNPB', 'ALPB', 'ASTB', 'BILT', 'LDHB', 'TRIB', 'POTURG', 
                     'GGTB', 'PCTB', 'TSHB', 'LPSB', 'IL6', 'BILIN', 'CLOB', 'ALBB', 'BILD', 
                     'DDPB', 'frequenza_cardiaca', 'frequenza_respiratoria', 'pressione_massima', 
                     'pressione_minima', 'saturimetria', 'P/F ratio']

data = pd.read_csv(data_path, index_col = 0)

X = data.drop(missing_row_sum(data, .6)).reset_index().drop('index', axis = 1)[selected_features]
Y = data.drop(missing_row_sum(data, .6)).reset_index().decesso

for col in X.columns:
    X[col] =X[col].apply(lambda x: np.nan if x == '\\' else x)

imp = IterativeImputer(missing_values=np.nan, initial_strategy = 'median', max_iter=100)
X = pd.DataFrame(imp.fit_transform(X), columns = X.columns)
ifo = IsolationForest(random_state = 10)
out = ifo.fit_predict(X)
X['is_out'] = out
rem = X[X.is_out == -1].index
X = X[X.is_out != -1].reset_index().drop(['index', 'is_out'], axis = 1)
Y = pd.DataFrame(Y).drop(rem, axis = 0).decesso
rf = list(bm.keys())[1]
lr = list(bm.keys())[0]
# fimp = smote_cv_score(X, Y, model = rf, imbalanced = False, 
#                       random_state = 10, scoring = 'feature_importance')
# fimp_mean = np.mean(np.array(fimp), axis = 0)

prova = refit_by_importance(X, Y, scores_on = 'cross_val', models = [lr, rf])





