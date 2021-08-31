# -*- coding: utf-8 -*-
"""
Get Survival Curves

Created on Mon May 11 13:01:30 2020

@author: Tobia Tommasini
"""

import pandas as pd
import matplotlib.pyplot as plt
from date_handle import diff_dates
from lifelines.statistics import logrank_test

class make_surv_study():
    
    def __init__(self, df):
        self.df = df
        
    def sel_data(self, 
                 col1 = [], 
                 col2 = [], 
                 to_dt = True,
                 grouper = None, 
                 sogg = None, 
                 val = None, 
                 meas = 'count'):
        
        self.grouper = grouper
        
        if to_dt:
            
            self.df[grouper] = pd.to_datetime(self.df[self.grouper])
        
        if val is not None:
            self.df_g1 = self.df[(self.df[col1[0]] == col2[0]) & (self.df[col1[1]] == val)] \
                                                                      .sort_values(by = self.grouper)
            self.df_g2 = self.df[(self.df[col1[0]] == col2[1]) & (self.df[col1[1]] == val)] \
                                                                      .sort_values(by = self.grouper)
        else:
            self.df_g1 = self.df[self.df[col1[0]] == col2[0]]
            self.df_g2 = self.df[self.df[col1[0]] == col2[1]]
            
        if meas == 'count':
            self.df_g1_surv = self.df_g1.groupby(self.grouper)[sogg] \
                                        .count().reset_index().rename({sogg:'Deaths'}, axis = 1)
            self.df_g2_surv = self.df_g2.groupby(self.grouper)[sogg] \
                                        .count().reset_index().rename({sogg:'Deaths'}, axis = 1)
        elif meas == 'nunique':
            self.df_g1_surv = self.df_g1.groupby(self.grouper)[sogg] \
                                        .nunique().reset_index().rename({sogg:'Deaths'}, axis = 1)
            self.df_g2_surv = self.df_g2.groupby(self.grouper)[sogg] \
                                        .nunique().reset_index().rename({sogg:'Deaths'}, axis = 1)
        
        return self.df_g1_surv, self.df_g2_surv
    
    def make_rates(self):
        
        self.df_g1_surv['Cumulative Deaths'] = self.df_g1_surv['Deaths'].cumsum()
        self.df_g2_surv['Cumulative Deaths'] = self.df_g2_surv['Deaths'].cumsum()
        
        self.df_g1_surv['Cumulative Death Rate'] = self. \
                                                   df_g1_surv['Cumulative Deaths'] / self.df_g1.shape[0]
        self.df_g2_surv['Cumulative Death Rate'] = self. \
                                                   df_g2_surv['Cumulative Deaths'] / self.df_g2.shape[0]
        
        self.df_g1_surv['Cumulative Survival Rate'] = 1 - self.df_g1_surv['Cumulative Death Rate']
        self.df_g2_surv['Cumulative Survival Rate'] = 1 - self.df_g2_surv['Cumulative Death Rate']
        
        self.df_g1_surv['Risk'] = self.df_g1_surv['Cumulative Death Rate'] / self. \
                                       df_g1_surv['Cumulative Survival Rate']
        self.df_g2_surv['Risk'] = self.df_g2_surv['Cumulative Death Rate'] / self. \
                                       df_g2_surv['Cumulative Survival Rate']
        
        return self.df_g1_surv, self.df_g2_surv
    
    def get_time(self, 
                 time = 'years'):
        
        date1 = pd.to_datetime(self.df_g1_surv[self.grouper])
        date2 = pd.to_datetime(self.df_g2_surv[self.grouper])

        if date1[0] < date2[0]:
            start = date1[0]
        else:
            start = date2[0]
            
        self.df_g1_surv[f'{time} to Start'] = diff_dates(start, date1, time = time)
        self.df_g2_surv[f'{time} to Start'] = diff_dates(start, date2, time = time)
        
        return self.df_g1_surv, self.df_g2_surv
    
def kaplan_plot(time1,
                time2,
                dr1,
                dr2,
                RR = True,
                annote = True,
                xlab = 'years',
                title = 'Survival Curves',
                titlesize = 15,
                text_coords = (0.56,0.6),
                text_size = 13,
                size=(20,8),
                labels=[]):
    
    sr1 = 1 - dr1
    sr2 = 1 - dr2
    Risk_ratio = dr1 / dr2
    results = logrank_test(time1, time2, 
                           sr1, sr2, alpha=95)
    if RR:
        fig, ax = plt.subplots(1,3, figsize=size)
    else:
        fig, ax = plt.subplots(1,2, figsize=size)

    ax[0].plot(time1, dr1, label=labels[0])
    ax[0].plot(time2, dr2, label=labels[1], c='red')
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel('Perc. Death')
    ax[0].set_title('Cumulative Death Rate')
    ax[0].legend()
    ax[1].plot(time1, sr1, label=labels[0])
    ax[1].plot(time2, sr2, label=labels[1], c='red')
    
    if annote:
        ax[1].annotate('LR P='+str(round(results.p_value,3)), xy=text_coords, xytext=(30,0), 
                       textcoords="offset points", fontsize=text_size)
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel('Perc. Surv')
    ax[1].set_title('Cumulative Survival Rate')
    ax[1].legend()
    
    if RR:
        ax[2].plot(Risk_ratio, c='k')
        ax[2].set_xlabel('Time')
        ax[2].set_ylabel('Ratio')
        ax[2].set_title('Risk Ratio')
    
    fig.suptitle(title, fontsize=titlesize)
    
    return fig, ax, results