# -*- coding: utf-8 -*-
"""
Series of fuctions to plot models
results / validation.

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats as ss
import sklearn.metrics as mtr
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

    
def plot_confusion_matrix(y_test,
                          pred,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    cm = mtr.confusion_matrix(y_test, pred)

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    _ = ax.set_title(title)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        _ = ax.set_xticks(tick_marks)
        _ = ax.set_yticks(tick_marks)
        _ = ax.set_xticklabels(target_names)
        _ = ax.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            _ = ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            _ = ax.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    # plt.tight_layout()
    _ = ax.set_ylabel('True label')
    _ = ax.set_xlabel('Predicted label')
    
    return fig, ax

def roc_curve(y_test,
              prob,
              ax=None,
              label='Model'):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    auc = mtr.roc_auc_score(y_test, prob[:,1])
    ns_probs = [0 for _ in range(len(y_test))]
    
    ns_fpr, ns_tpr, _ = mtr.roc_curve(y_test, ns_probs)
    md_fpr, md_tpr, _ = mtr.roc_curve(y_test, prob[:,1])
    
    ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', c='k')
    ax.plot(md_fpr, md_tpr, marker='.', label=label+' AUC: '+str(round(auc,4)))
    _ = ax.grid()
    _ = ax.legend()
    _ = ax.set_title('ROC Curve')
    _ = ax.set_xlabel('FPR')
    _ = ax.set_ylabel('TPR')
    _ = ax.set_xlim(0.0,1.0)
    _ = ax.set_ylim(0.0,1.0)
    
    return fig, ax

def multiclass_roc_curve(x_train,
                         x_test,
                         y_train,
                         y_test,
                         n_classes = 3,
                         standardize = False,
                         ax = None,
                         model = None,
                         title='Model'):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
    
    classifier = OneVsRestClassifier(model)
    
    if standardize:
        
        sd = StandardScaler()
        ppl = Pipeline([
            ('sd', sd),
            ('cla', classifier)
            ])
    
        prob = ppl.fit(x_train, y_train).predict_proba(x_test)
        
    else:
        prob = classifier.fit(x_train, y_train).predict_proba(x_test)
        
    fpr = {}
    tpr = {}
    auc = {}
    
    for cl in range(n_classes):
        fpr[cl], tpr[cl], _ = mtr.roc_curve(y_test[:, cl], prob[:, cl])
        auc[cl] = mtr.auc(fpr[cl], tpr[cl])
        
        ax.plot(fpr[cl], tpr[cl], label = 'AUC of class {0} (area = {1:0.4f})'
             ''.format(cl, auc[cl]), zorder=2)
        
    ax.plot([0,1], [0,1], linestyle='--', label='No Skill', c='k')
    
    _ = ax.grid()
    _ = ax.legend()
    _ = ax.set_title('ROC Curve '+str(title))
    _ = ax.set_xlabel('FPR')
    _ = ax.set_ylabel('TPR')
    _ = ax.set_xlim([-0.02, 1.0])
    _ = ax.set_ylim([0.0, 1.02])
    
    return fig, ax

def precision_recall_curve(y_test,
                           prob,
                           cl,
                           ax = None,
                           label = 'Model'):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    precision, recall, _ = mtr.precision_recall_curve(y_test, prob[:, cl])
    
    ax.plot(recall, precision, label=label)
    _ = ax.legend()
    _ = ax.grid()
    _ = ax.set_xlabel('Recall')
    _ = ax.set_ylabel('Precision')
    _ = ax.set_title('Precision vs Recall Curve')
    
    return fig, ax
        
def multiclass_precision_recall(x_train,
                                x_test,
                                y_train, 
                                y_test, 
                                n_classes = 3, 
                                standardize = False,
                                ax = None,
                                model = None,
                                title = 'Model'):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    precision = {}
    recall = {}
    
    classifier = OneVsRestClassifier(model)
    
    if standardize:
        
        sd = StandardScaler()
        ppl = Pipeline([
            ('sd', sd),
            ('cla', classifier)
            ])
    
        prob = ppl.fit(x_train, y_train).predict_proba(x_test)
        
    else:
        prob = classifier.fit(x_train, y_train).predict_proba(x_test)
    
    for cl in range(n_classes):
        precision[cl], recall[cl], _ = mtr.precision_recall_curve(y_test[:, cl], prob[:, cl])
        ax.plot(recall[cl], precision[cl], lw=2, label='class {}'.format(cl))
    
    _ = ax.set_xlabel('Recall')
    _ = ax.set_ylabel('Precision')
    _ = ax.legend()
    _ = ax.grid()
    _ = ax.set_title('Precision vs Recall Curve '+title)
    _ = ax.set_xlim([0.0, 1.02])
    _ = ax.set_ylim([0.0, 1.02])
    
    return fig, ax

def reg_repo_plot(mod_res, 
                  residual,
                  line='s',
                  ax=None):
    
    val, p = ss.shapiro(residual)
    y_max = np.max(residual)
    
    if ax is None:
        fig, ax = plt.subplots(1,2)
    else:
        fig = []
        
    ax[0].scatter(residual, mod_res)
    sm.qqplot(residual, ax=ax[1], line=line, c='#1f77b4')
    _ = ax[1].text(-2, y_max-0.5, 'Shap. p:'+' '+str(round(p),4))
    _ = ax[0].set_xlabel('Residuals')
    _ = ax[0].set_ylabel('Fitted Values')
    
    for t,x in zip(['Fitted vs Residuals','Residuals QQ plot'], ax.flatten()):
        _ = x.grid(alpha=.3)
        _ = x.set_title(t)
    
    return fig, ax
    
def stepwise_plot(x,
                  y,
                  goal = 'regression',
                  ax = None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    R = []
    AIC = []
    BIC = []
    mods = [x.iloc[:, 0:n+1] for n,i in enumerate(x.columns)]
    
    for data in mods:
        _ = sm.add_constant(data)
        
        if goal == 'regression':
            model = sm.OLS(y,data)
            results = model.fit()
            R.append (results.rsquared_adj)
            lb = 'Std. Adjusted R-squared'
            
        else:
            model = sm.Logit(y,data)
            results = model.fit(disp=False)
            R.append (results.prsquared)
            lb = 'Std. Pseudo R-squared'
        
        AIC.append(results.aic)
        BIC.append(results.bic)
        
    
    std_AIC = (AIC - np.min(AIC)) / (np.max(AIC) - np.min(AIC))
    std_BIC = (BIC - np.min(BIC)) / (np.max(BIC) - np.min(BIC))
    std_R = (R - np.min(R)) / (np.max(R) - np.min(R))
    
    ax.plot(x.columns, std_AIC, marker='o', label='Std. Aikake Information Criterion')
    ax.plot(x.columns, std_BIC, marker='o', label='Std. Bayesian Information Criterion')
    ax.plot(x.columns, std_R, marker='s', label=lb)
        
    _ = ax.legend()
    _ = ax.grid(alpha=.3)
    _ = ax.set_xticks(np.arange(len(x.columns)))
    _ = ax.set_xticklabels(x.columns, rotation = 45)
    _ = ax.set_title('Behavior of Model Scores Adding One Feature at Each Step')
        
    return fig, ax
        
def cv_score_plot(x, 
                  y, 
                  mods = [], 
                  folds=10,
                  score='neg_mean_squared_error',
                  ax=None):
    
    import sklearn
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
    
    if score == 'neg_mean_squared_error':
        ylab = 'MSE'
    else:
        ylab = 'AUC'
    
    # needs to be modified for pipelines with length greater than 2
    labs = []
    for mod in mods:
        if isinstance(mod, sklearn.pipeline.Pipeline):
            labs.append(str(mod[1]).split('(')[0])
        else:
            labs.append(str(mod).split('(')[0])
            
    for leg, mod in zip(labs, mods):
        cv_sc = cross_val_score(mod, x, y, cv=folds, scoring=score)
        if score == 'neg_mean_squared_error':
            ax.plot(range(1,folds+1), -cv_sc, marker='o', label=leg)
            _ = ax.legend()
        else:
            ax.plot(range(1,folds+1), cv_sc, marker='o', label=leg)
            _ = ax.legend()
            
    _ = ax.grid(alpha=.3)
    _ = ax.set_xlabel('Folds')
    _ = ax.set_ylabel(ylab)
    _ = ax.set_title(str(folds)+' Fold Cross Validation '+ylab)
    
    return fig, ax

def explained_variance_plot(expl, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    components = ['Comp '+str(i+1) for i in range(len(expl))]
    
    ax.plot(components, np.cumsum(expl * 100), marker = 's', c = 'k', label = 'Cumsum')
    ax.bar(components, expl * 100)
    
    _ = ax.set_xticklabels(components, rotation = 45)
    _ = ax.grid(alpha=.3)
    _ = ax.legend()
    _ = ax.set_ylabel('Explained Variance Ratio (%)')
    _ = ax.set_title('Explained Variance Plot')
    
    return fig, ax
        

    
            