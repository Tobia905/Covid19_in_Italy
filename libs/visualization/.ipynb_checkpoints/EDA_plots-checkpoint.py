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
            subset = None,
            ascending = True,
            linewidth = None,
            title = ''):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if rate is not None:
        perc = 100
        if meas == 'mean':
            raise ValueError(f"rate parameter is not valid with measure: {meas}")
        else:
            grouped = df.groupby(grouper).agg({rate:'sum'}) \
                        .div(df.groupby(grouper).agg({y:meas})).rename({0:f'{rate}_rate'}, axis = 1)
            grouped[f'{y}_{meas}'] = df.groupby(grouper).agg({y:meas})
            grouped = grouped.reset_index().sort_values(by = f'{rate}_rate', ascending = ascending)
    else:
        perc = 1
        grouped = df.groupby(grouper).agg({y:meas}).rename({y:f'{y}_{meas}'}, axis = 1)
        grouped = grouped.reset_index().sort_values(by = f'{y}_{meas}', ascending = ascending) 
    if subset is not None:
        grouped = grouped[grouped[f'{y}_{meas}'] >= subset]
        
    labels = grouped.apply(lambda x: str(x[0]) + "\n " + str(round(perc*x[1],2)) , axis = 1)
    if size == 'linear':
        sizes = grouped[f'{y}_{meas}'].values.tolist()
    elif size == 'log':
        sizes = np.log((1/2)*grouped[f'{y}_{meas}']).values.tolist()
    elif size == 'rad':
        sizes = np.sqrt(grouped[f'{y}_{meas}']).values.tolist()
    colors = [plt.cm.Spectral(i/(len(labels))) for i in range(len(labels))]
    squarify.plot(sizes = sizes, label = labels, 
                  color = colors, alpha = .9, ax = ax, linewidth = linewidth)
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
              ax = None,
              fig = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
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


    


    
    
    


