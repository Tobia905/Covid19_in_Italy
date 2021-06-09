# -*- coding: utf-8 -*-
"""
Series of functions useful to obtain
a crosstab matrix.

Created on Wed Jun  3 16:30:42 2020

@author: Tobia Tommasini
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
                    raise ValueError("{} never has 1 value".format(g))
                
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

def heat_plot(df, 
              triang = False, 
              dim = (12,8), 
              linewidth = 0.0, 
              annot = True,
              cmap = 'YlGnBu',
              ylab_deg = 90,
              title = None):
    
    fig, ax = plt.subplots(figsize=dim)
    
    if triang:
        mask = np.triu(np.ones_like(df))
        _ = sns.heatmap(df, cmap=cmap, linewidth=linewidth, mask=mask, annot=annot, ax=ax)
        
    else:
        _ = sns.heatmap(df, cmap=cmap, linewidth=linewidth, annot=annot, ax=ax)
      
    if title is None:
        _ = ax.set_title('Co-Occurrences (%)')
        
    else:
        _ = ax.set_title(title)
        
    _ = ax.set_yticklabels(df.columns, rotation = ylab_deg)
        
    return fig, ax
        
    
    
    