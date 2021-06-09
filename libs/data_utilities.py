# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:59:30 2020

@author: Tobia Tommasini
"""
import pandas as pd
import numpy as np
from pathlib import Path

def open_files(path, 
               files = [], 
               sep = ',', 
               fmt = '*.csv', 
               to_dict = False,
               index_col = None,
               check_index = False): 
    
    ds_list = []
    names = []
    
    for file in Path(path).glob(fmt):
        if file.name.split('.')[0] in files:
            names.append(file.name.split('.')[0])
                
        if fmt == '*.csv':
            ds_list.append(pd.read_csv(file, sep = sep, index_col = index_col))
            
        elif fmt == '*.xlsx':
            ds_list.append(pd.read_excel(file, index_col = index_col))
                
    if check_index:
    
        for n, df in enumerate(ds_list):
            if isinstance(df.index, pd.core.indexes.base.Index):
                ds_list[n] = df.reset_index()

    if to_dict:
        
        db = {}
        for name,file in zip(names, ds_list): 
            db[name] = file
                
        return db
    
    else:
        
        return ds_list, names
    
def open_file_db(path, 
                 files = [], 
                 sep = ',', 
                 fmt = '*.csv',
                 index_col = None):
    
    db = {}
    for file in Path(path).glob(fmt):
        if file.name.split('.')[0] in files:
            if fmt == '*.csv':
                db[file.name.split('.')[0]] = pd.read_csv(file, sep = sep, index_col = index_col)
            
            elif fmt == '*.xlsx':
                db[file.name.split('.')[0]] = pd.read_excel(file, index_col = index_col)
            
    return db
            
            
def rename_all_cols(db, renames):
    
    db_cols = [pd.Series(df.columns) for df in db.values()]
    mapped_db_cols = [db_col.map(renames) for db_col in db_cols]
    
    renamer = []
    for old, new in zip(db_cols, mapped_db_cols):
        rn = {old_name:new_name for old_name, new_name in zip(old, new)}
        renamer.append(rn)
        
    for n, (ren, df_name) in enumerate(zip(renamer, db.keys())):
        db[df_name] = list(db.values())[n].rename(ren, axis = 1)
        
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
    
def cleaner(df, 
            nan_like = [], 
            sel_vars = [],
            conv = 'float94'):
    
    na_conv = lambda x: np.nan if x in nan_like else x
    
    for col in sel_vars:
        df[col] = df[col].apply(na_conv)
        df[col] = [str(i).replace(',','.') for i in df[col]]
        df[col] = df[col].astype(conv)
        
    return df

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
                cor_val.append(100*pd.crosstab(df1[col], df2[g])[1][1]/n)
                
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
        
    
    
    
