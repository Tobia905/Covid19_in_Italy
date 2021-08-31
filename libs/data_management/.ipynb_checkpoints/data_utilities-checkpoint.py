# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:59:30 2020

@author: Tobia Tommasini
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from functools import reduce

def _get_mapper(x, mapper):
    return [mapper[t] if t in mapper.keys() else t for t in x]

def merge_all(db, on = ['ID_PAZIENTE', 'CODDEG']):
    if isinstance(db, dict):
        db = list(db.values())
    return reduce(lambda df1,df2: pd.merge(df1, df2, on = on), db)
    
def open_file_db(path, 
                 files = [], 
                 sep = ',', 
                 fmt = '*.csv',
                 index_col = None,
                 to_list = False):
    
    db = {}    
    for file in Path(path).glob(fmt):
        if fmt == '*.csv':
            df = pd.read_csv(file, sep = sep, index_col = index_col)
        elif fmt == '*.xlsx':
            df = pd.read_excel(file, index_col = index_col)
            
        if len(files) == 0:
            db[file.name.split('.')[0]] = df
        else:
            if file.name.split('.')[0] in files:
                db[file.name.split('.')[0]] = df
    
    if to_list:
        return list(db.keys()), list(db.values())
    else:
        return db
            
def nan_int(x, 
            int_vars = []):
    
    def nan_helper(x):
        return np.isnan(x), lambda z: z.nonzero()[0]
    
    if isinstance(x, pd.core.frame.DataFrame):
        for col in int_vars:
            nans, y = nan_helper(x[col].values)
            x[col].values[nans] = np.interp(y(nans), y(~nans), x[col].values[~nans])

    else:
        x = np.array(x)
        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        
    return x

def get_one_grouper(df, 
                    x='', 
                    y='', 
                    reindex = True, 
                    rename = 'grouper', 
                    drop = False):
    
    if reindex:
        df = df.reset_index()
     
    df[rename] = [g1+' '+g2 for g1,g2 in zip(df[x].astype('str'),df[y].astype('str'))]
    if drop:
        return df.set_index(rename).drop([x,y], axis=1)
    else:
        return df.set_index(rename)

class dum_to_group():
    
    def __init__(self, 
                 df, 
                 ind = None, 
                 renm = ''):
        
        self.df = df
        self.ind = ind
        self.renm = renm
    
    def stacker(self):
        
        if self.ind is not None:
            self.df.set_index(self.ind, inplace=True)
            
        self.df = self.df[self.df == 1].stack().reset_index().drop(0,1).rename({'level_1':self.renm}, 
                                                                               axis = 1)
        
        return self.df
    
    def unstacker(self, 
                  drop=False):
        
        un_df = pd.DataFrame(self.df.groupby([self.ind,
                                           self.renm])[self.ind].nunique()).rename({self.ind:'Count'}, 
                                                                                   axis=1).reset_index()
        dum_df = pd.get_dummies(un_df.loc[:, self.renm]).dropna()
        dum_df['index'] = un_df[self.ind]
        dum_cor = dum_df.set_index('index').sum(axis=0, level = 'index').reset_index()
        if drop:
            dum_cor.drop('index', axis=1, inplace=True)
                    
        return dum_cor
    
def prep_ct(df1,df2, on=''):
    
    df_tot = pd.merge(df1,df2, on=on)    
    df1.drop(on, axis=1, inplace=True)
    df2.drop(on, axis=1, inplace=True)
    
    df1_ct = df_tot.loc[:, df1.columns]
    df2_ct = df_tot.loc[:, df2.columns]
    
    return df1_ct, df2_ct
    
def get_crosstabs(df1, 
                  df2=None):
    
    rep = []
    for col in df1.columns:
        rep.append(df1[col])
        
    num = df1.shape[0]
    crostb = []
    if df2 is not None:
        dim = []
        for col in df1.columns:
            dim.append(df1[col].sum(axis=0))
            
        for col,n in zip(df1.columns, dim):
            cor_val = []
            for g in df2.columns:
                try:
                    cor_val.append(100*pd.crosstab(df1[col], df2[g])[1][1]/n)
                except:
                    raise ValueError("{} is never 1".format(g)) 
            crostb.append(cor_val)
            
        co_occ = pd.DataFrame(crostb, index = df1.columns, columns = df2.columns)
    else:
        for group in rep:
            cor_val = []
            for n in range(len(rep)):
                cor_val.append(100*(pd.crosstab(rep[n],group))[1][1] / num)  
            crostb.append(cor_val)
            
        co_occ = pd.DataFrame(crostb, index = df1.columns, columns = df1.columns)

    co_occ = pd.DataFrame([round(co_occ[col],3) for col in co_occ.columns])
    
    return co_occ

def get_better_latex(df, perc = True):
    
    tl = df.round(3).to_latex()
    if perc:
        tl = tl.replace('\\\\','\\').replace('\n',' ').replace('\%', '%') 
    else:
        tl = tl.replace('\\\\','\\').replace('\n',' ')
        
    tl = re.sub(r'(\\toprule|\\midrule|\\bottomrule)', '', tl)
    
    return tl

### FEATURE EXPANSION ###
    
def get_window(df, window, start = 0, step = 1):
    
    if window == 1:
        return df
    
    else:
        shifts = []
        if window > 0:
            rng = range(window)
        else:
            rng = range(window, start, step)
        for w in rng:
            df_shift = df.shift(w)
            if isinstance(df, pd.core.frame.DataFrame):
                df_shift = df_shift.rename({col:col+'_shift_{}'.format(w) 
                                            for col in df_shift.columns}, axis = 1)
            shifts.append(df_shift)
            
        if window > 0:
            shifts = pd.concat(shifts[1:], axis = 1)
        else:
            shifts = pd.concat(shifts, axis = 1)
        
        if isinstance(df, pd.core.series.Series):
            if window > 0:
                names = [col+'_shift_{}'.format(w+1) for col, w in zip(shifts.columns, rng)]
            else:
                names = [col+'_shift_{}'.format(w) for col, w in zip(shifts.columns, rng)]
            shifts = pd.DataFrame(shifts.values, columns = names)
            
    if window > 0:        
        return pd.concat((df, shifts), axis = 1)
    
    else:
        return pd.concat((shifts, df), axis = 1)
    
def pairwise_interaction_expansion(df, 
                                   tresh = .5, 
                                   method = 'pearson'):
    
    df = df.copy()
    pair_cor = df.corr(method = method)
    pair_cor_triang = pd.DataFrame(np.tril(pair_cor), 
                                   columns = pair_cor.columns, index = pair_cor.index)
    pairs = []
    for i in range(pair_cor_triang.values.shape[0]):
        for j in range(pair_cor_triang.values.shape[1]):
            if abs(pair_cor_triang.values[i][j]) >= tresh \
            and (pair_cor_triang.iloc[i].name != pair_cor_triang.iloc[:, j].name):
                pairs.append((pair_cor_triang.iloc[i].name, pair_cor_triang.iloc[:, j].name))
    
    for feat1, feat2 in pairs:
        df[f'{feat1}*{feat2}'] = df[feat1] * df[feat2]
    
    return df
    
def feature_expansion(X, 
                      transf = ['poly'],
                      power = [2],
                      interaction = False,
                      tresh = .5,
                      method = 'pearson'):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    nans = ['nan', 'NaN', 'NAN', '-inf', 'inf']
    new_X = X.copy()
    feat_exp = {'poly': np.power,
                'log': np.log,
                'sqrt': np.sqrt}
    
    if not isinstance(new_X, pd.core.frame.DataFrame):
        new_X = pd.DataFrame(new_X)
    
    for col in new_X.columns:
        un_val = new_X[col].unique()
        if len(un_val) == 2 or (len(un_val) == 3 and np.nan in un_val):
            pass
        else:
            for n, tr in enumerate(transf):
                try:
                    if tr == 'poly':
                        new_X[f'{tr}{power[n]}_{col}'] = feat_exp[tr](new_X[col], power[n])
                    else:
                        new_X[f'{tr}_{col}'] = feat_exp[tr](new_X[col]).apply(lambda x: np.nan if str(x) in nans else x)
                except TypeError:
                    pass
    
    if interaction:
        new_X = pairwise_interaction_expansion(new_X, tresh = tresh, method = method)
    
    return new_X
    