# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:59:30 2020

@author: Tobia Tommasini
"""
import pandas as pd
import numpy as np

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
        
    
    
    
