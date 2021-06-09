# -*- coding: utf-8 -*-
"""
Series of functions useful to EDA plots.

@author: Tobia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
import scipy.stats as ss
import seaborn as sns
import itertools

def treemap(df, 
            grouper = '', 
            y = '', 
            meas = 'nunique', 
            size = 'linear',
            ax = None, 
            rate = None,
            subs = None,
            ascending = True,
            linewidth = None,
            title = ''):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if rate is not None:
        
        if meas == 'nunique':
            g_df = pd.DataFrame(df.groupby(grouper)[rate].sum() / 
                                df.groupby(grouper)[y].nunique()).rename({0:rate+'_rate'}, axis=1)
            g_df[y+'_Count'] = df.groupby(grouper)[y].nunique()
            
            
        elif meas == 'count':
            g_df = pd.DataFrame(df.groupby(grouper)[rate].sum() / 
                                df.groupby(grouper)[y].count()).rename({0:rate+'_rate'}, axis=1)
            g_df[y+'_Count'] = df.groupby(grouper)[y].count()
            
        g_df = g_df.reset_index().sort_values(by=rate+'_rate', ascending=ascending)
        
    else:
        if meas == 'nunique':
            g_df = pd.DataFrame(df.groupby(grouper)[y].nunique()).rename({y:y+'_Count'}, axis=1)
            
        elif meas == 'count':
            g_df = pd.DataFrame(df.groupby(grouper)[y].count()).rename({y:y+'_Count'}, axis=1)

        g_df = g_df.reset_index().sort_values(by=y+'_Count', ascending=ascending)  
         
    if subs is not None:
        g_df = g_df[g_df[y+'_Count'] >= subs]
        
    if rate:
        labels = g_df.apply(lambda x: str(x[0]) + "\n " + str(round(100*x[1],2)) , axis=1)
        
    else:
        labels = g_df.apply(lambda x: str(x[0]) + "\n " + str(round(x[1],2)) , axis=1)
        
    if size == 'linear':
        sizes = g_df[y+'_Count'].values.tolist()
        
    elif size == 'log':
        sizes = np.log((1/2)*g_df[y+'_Count']).values.tolist()
        
    elif size == 'rad':
        sizes = np.sqrt(g_df[y+'_Count']).values.tolist()

    colors = [plt.cm.Spectral(i/(len(labels))) for i in range(len(labels))]
    
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.9, ax=ax, linewidth=linewidth)
    _ = ax.axis('off')
    _ = ax.set_title(title)
    
    return fig, ax

def scatter_fit(x, 
                y, 
                degree = 1,
                coef = False,
                linecol = 'r',
                ax = None,
                label = ''):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if degree == 1:
        m1, b1 = np.polyfit(x, y, degree)
        
        if coef:
            corr, _ = ss.pearsonr(x, y)
            
    else:
        coefs = np.polyfit(x, y, degree)
        poly = np.poly1d(coefs)
        
        if coef:
            corr, _ = ss.spearmanr(x, y)
            
    _ = ax.scatter(x, y)
    
    if degree == 1:
        _ = ax.plot(np.sort(x), m1*np.sort(x)+b1, c=linecol, label = label+' Linear Fit')
        
        
    else:
        _ = ax.plot(np.sort(x), poly(np.sort(x)), c=linecol, label = label+' Polynomial Fit')
    
    if coef:
        
        # needs to be modified
        x0, xmax = ax.get_xlim()
        y0, ymax = ax.get_ylim()
        data_width = xmax - x0
        data_height = ymax - y0
        _ = ax.text(x0 + data_width * 0.85, y0 + data_height * 0.8, 'corr:'+str(round(corr, 4)))
        
    _ = ax.legend()
    _ = ax.grid(alpha=.3)
        
    return fig, ax

def nan_plot(df, 
             cbar = False, 
             show = 'nan', 
             ax = None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if show == 'nan':
        
        to_show = df.isnan()
        
    elif show == 'null':
        
        to_show = df.isnull()
        
    _ = sns.heatmap(to_show, cbar = cbar)
    _ = ax.set_title('NaN / Null Position')
    
    return fig, ax

def grouped_barplot(df, 
                    x = '', 
                    grouper = '', 
                    hue = '', 
                    meas = 'nunique', 
                    width = .35, 
                    title = '',
                    ax = None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    lb = np.arange(len(df[x].unique()))
    labels = df[x].unique()
    width = width
    
    g1 = str(df[hue].unique[0])
    g2 = str(df[hue].unique[1])
    
    df_g1 = df[df[hue] == g1]
    df_g2 = df[df[hue] == g2]
    
    df_pl1 = df_g1.groupby(x).agg({grouper:meas}).reset_index()
    df_pl2 = df_g2.groupby(x).agg({grouper:meas}).reset_index()
    
    ax.bar(lb - width/2, df_pl1[grouper], width, label = g1)
    ax.bar(lb + width/2, df_pl2[grouper], width, label = g2)
    
    _ = ax.set_ylabel('#')
    _ = ax.set_xticks(lb)
    _ = ax.set_xticklabels(labels)
    _ = ax.legend()
    _ = ax.grid(alpha=.3)
    _ = ax.set_title(title)
    
    return fig, ax

def plot_cdf(df, 
             x = '',
             grouper = '',
             meas = 'nunique',
             marker = 's',
             ax = None,
             xlab = '',
             ylab = '',
             title = '',
             label = None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
    
    N = df.shape[0]
    
    if meas == 'nunique':
        df = df.groupby(grouper)[x].nunique()
        
    else:
        df = df.groupby(grouper)[x].count()
        
    df.cumsum().div(N).plot(ax = ax, label = label, marker = 's')
    
    _ = ax.grid(alpha=.3)
    _ = ax.set_xlabel(xlab)
    _ = ax.set_ylabel(ylab)
    _ = ax.set_title()
    _ = ax.legend()
    
    return fig, ax

def heat_plot(mat, 
              xlab = None, 
              ylab = None,
              xlab_rot = None,
              ha = 'center',
              rotation = None,
              interpolation = 'nearest',
              cmap = 'Blues',
              annot = False,
              drop = -1,
              aspect = 'auto',
              title = 'Co-Occurrences (%)',
              ax = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    else:
        fig = []
    
    if xlab is None:
        xlab = mat.columns
    
    if ylab is None:
        ylab = mat.index
    
    mat = mat.values
    
    marksx = np.arange(mat.shape[1])
    marksy = np.arange(mat.shape[0])
    
    mt = ax.imshow(mat, cmap = cmap, aspect = aspect)
    _ = ax.set_xticks(marksx)
    _ = ax.set_yticks(marksy)
    _ = ax.set_yticklabels(ylab)
    _ = ax.set_xticklabels(xlab, rotation = xlab_rot, ha = ha)
    _ = ax.set_title(title)
    
    _ = fig.colorbar(mt)

    if annot:
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            if drop is not None:
                if mat[i][j] > drop:
                    _ = ax.text(j, i, "{:,}".format(round(mat[i][j], 3)), 
                                horizontalalignment="center")
    
    return fig, ax


    


    
    
    


