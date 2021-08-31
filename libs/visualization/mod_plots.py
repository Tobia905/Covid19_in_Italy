# -*- coding: utf-8 -*-
"""
Series of fuctions to plot models
results / validation.

@author: Tobia Tommasini.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import scipy.stats as ss
import sklearn.metrics as mtr
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as imPipeline
from libs.modeling.get_best_model import _check_parallel, _get_estimator

def plot_confusion_matrix(
        y_test,
        pred = None,
        target_names = None,
        cv = None,
        n_jobs = 1,
        pre_dispatch = '1.5*n_jobs',
        x = None,
        y = None,
        imbalanced = False,
        standardize = False,
        stratified = True,
        shuffle = False,
        strategy = None,
        under_strategy = None,
        method = 'smote',
        model = None,
        title = 'Confusion Matrix',
        cmap = None,
        normalize = True,
        random_state = None,
        ax = None,
        **kwargs
    ):
    
    """Plots the confusion matrix of a classification
       model. Supports Multiclass problems.
       
       Parameters
       ----------
       y_test : array-like - true values of the outcome.
       
       pred : array-like - predicted values of the outcome.
       
       target_names : list - the names of the classes, default = None.
       
       title : str - title of the plot, default = "Confusion Matrix".
       
       cmap : str - cmap for imshow, if None, default = "Blues".
       
       normalize : bool - if true, results will be reported as %.
       
       ax : array - the axes on which to plot, default = None
       
       Returns
       -------
       fig: matplotlib.figure.
       
       ax : array - axes with modification added.
    """
    if standardize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if cv is not None:
        _cv_score, cv_args = _check_parallel(n_jobs = n_jobs,
                                             pre_dispatch = pre_dispatch,
                                             imbalanced = imbalanced,
                                             strategy = strategy,
                                             under_strategy = under_strategy,
                                             method = method,
                                             **kwargs)
        
        cms = _cv_score(x, y, model = model, cv = cv, stratified = stratified, shuffle = shuffle,
                        random_state = random_state, scoring = 'confusion_matrix', **cv_args)
        cm = np.mean(cms, axis=0)
        
    else:
        cm = mtr.confusion_matrix(y_test, pred)

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
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
    
    """Plots the roc curve for binary cassification problems.
    
       Parameters
       ----------
       y_test : array-like - true values of the outcome.
       
       prob : array-like - predicted probabilities for a class.
       
       ax : array - the axes on which to plot, default = None.
       
       label : str - the legend of the plot, default = "Model"
       
       Returns
       -------
       fig: matplotlib.figure.
       
       ax : array - axes with modification added.
    """
    
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
    
    """Roc curve for multiclass problems.
    
       Parameters
       ----------
       x_train : array-like - training input.
       
       x_test : array-like - testing input.
       
       y_train : array-like - training output.
       
       y_test : array-like - true values of the outcome.
       
       n_classes : int - number of classes, default = 3.
       
       standardize : bool - if true, standard scaling will be performed.
       
       ax : array - the axes on which to plot, default = None.
       
       model : scikit-learn classifier - the model you want to train, default = None.
    """
    
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

def CV_roc_curve(x,
                 y,
                 cv = 5,
                 n_jobs = 1,
                 pre_dispatch = '1.5*n_jobs',
                 model = None,
                 imbalanced = False,
                 normalize = False,
                 stratified = True,
                 shuffle = False,
                 strategy = None,
                 under_strategy = None,
                 method = 'smote',
                 random_state = None,
                 showall = True,
                 ax = None,
                 **kwargs):
    
    if isinstance(model, Pipeline) or isinstance(model, imPipeline):
        label = _get_estimator(model).__class__.__name__
    
    if normalize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    _cv_score, cv_args = _check_parallel(n_jobs = n_jobs,
                                         pre_dispatch = pre_dispatch,
                                         imbalanced = imbalanced,
                                         strategy = strategy,
                                         under_strategy = under_strategy,
                                         method = method,
                                         **kwargs)
    
    int_tpr_mean = []
    rocs = _cv_score(x, y, model = model, cv = cv, stratified = stratified, 
                     shuffle = shuffle, random_state = random_state, scoring = 'roc_curve', **cv_args)
    aucs = _cv_score(x, y, model = model, cv = cv, stratified = stratified, 
                     shuffle = shuffle, random_state = random_state, scoring = 'roc_auc', **cv_args)
    mean_fpr = np.linspace(0, 1, 100)
    for n, _ in enumerate(rocs):
        interp_tpr = np.interp(mean_fpr, rocs[n][0], rocs[n][1])
        interp_tpr[0] = 0.0
        int_tpr_mean.append(interp_tpr)
        if showall:
            if cv <= 5:
                lab = 'Fold {0} AUC: {1:.4f}'.format(n+1, aucs[n])
            else:
                lab = None
            ax.plot(rocs[n][0], rocs[n][1], label = lab, alpha = .3)
            
    mean_tpr = np.mean(int_tpr_mean, axis=0)
    std_tpr = np.std(int_tpr_mean, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = mtr.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.plot(mean_fpr, mean_tpr, linewidth = 2, c = 'b', label = 'CV Mean ROC - AUC: {0:.2f} $\pm$ {1:.2f}'.format(mean_auc, std_auc))
    ax.plot([0,1], [0,1], c = 'k', linestyle = '--')
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = .2, label = r'$\pm$ Std. Dev.')
    ax.legend()
    ax.set_xlim(-.02,1.02)
    ax.set_ylim(-.02,1.02)
    ax.grid(alpha = .3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('{0} CV ROC Curve - Mean AUC: {1:.4f}'.format(label, np.mean(aucs)))
    
    return fig, ax

def CV_precision_recall(x,
                        y,
                        cv = 5,
                        n_jobs = 1,
                        pre_dispatch = '1.5*n_jobs',
                        model = None,
                        imbalanced = False,
                        normalize = False,
                        stratified = True,
                        shuffle = False,
                        strategy = None,
                        under_strategy = None,
                        method = 'smote',
                        random_state = None,
                        showall = True,
                        ax = None,
                        **kwargs):
    
    if isinstance(model, Pipeline) or isinstance(model, imPipeline):
        label = _get_estimator(model).__class__.__name__
    
    if normalize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    _cv_score, cv_args = _check_parallel(n_jobs = n_jobs,
                                         pre_dispatch = pre_dispatch,
                                         imbalanced = imbalanced,
                                         strategy = strategy,
                                         under_strategy = under_strategy,
                                         method = method,
                                         **kwargs)
        
    int_pre_mean = []
    prec = _cv_score(x, y, model = model, cv = cv, stratified = stratified, shuffle = shuffle,
                     random_state = random_state, scoring = 'precision_recall_curve', **cv_args)
    mean_rec = np.linspace(0, 1, 100)
    for n, _ in enumerate(prec):
        interp_pre = np.interp(mean_rec, prec[n][0], prec[n][1])
        int_pre_mean.append(interp_pre)
        if showall:
            ax.plot(prec[n][0], prec[n][1], alpha = .3)
    
    mean_pre = np.mean(int_pre_mean, axis=0)
    std_pre = np.std(int_pre_mean, axis=0)
    prec_upper = np.minimum(mean_pre + std_pre, 1)
    prec_lower = np.maximum(mean_pre - std_pre, 0)
    ax.plot(mean_rec, mean_pre, linewidth = 2, c = 'b', label = 'CV Mean PR')
    ax.fill_between(mean_rec, prec_lower, prec_upper, color = 'grey', alpha = .2, label = r'$\pm$ Std. Dev.')
    ax.legend()
    ax.set_xlim(-.02,1.02)
    ax.set_ylim(-.02,1.02)
    ax.grid(alpha = .3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('{0} CV PR Curve'.format(label))
    
    return fig, ax    

def all_models_results(y_true,
                       preds, 
                       probs,
                       x_train = None,
                       x_test = None,
                       y_train = None,
                       models = [],
                       multiclass = False,
                       standardize = True,
                       st_cmat = False,
                       normalize = True,
                       n_classes = None,
                       cm_target = [],
                       ax = None):
    
    if isinstance(models, dict):
        models = list(models.keys())
    
    if ax is None:
        fig, ax = plt.subplots(len(models),3, figsize=(18,len(models)*5))
    else:
        fig = []
        
    if isinstance(preds, dict):
        preds = list(preds.values())
    if isinstance(probs, dict):
        probs = list(probs.values())
        
    for n, mod in enumerate(models):
        if multiclass:
            if standardize:
                sd = StandardScaler()
                ppl = Pipeline([
                        ('sd',sd),
                        ('mod',mod)
                        ])
        
                _ = ppl.fit(x_train, y_train)
                pred = ppl.predict(x_test)
                
            else:
                _ = mod.fit(x_train, y_train)
                pred = mod.predict(x_test)
    
            multiclass_roc_curve(x_train, x_test, y_train, y_true, n_classes=n_classes, 
                                 standardize=standardize, model=mod, 
                                 ax=ax[n][0], title=str(mod).split('(')[0]) 
            multiclass_precision_recall(x_train, x_test, y_train, y_true, 
                                        n_classes=n_classes, standardize=standardize, 
                                        model=mod, ax=ax[n][1], title=str(mod).split('(')[0])
            plot_confusion_matrix(y_true, pred, target_names = cm_target, 
                                  normalize=normalize, ax=ax[n][2])
        else:
            roc_curve(y_true, probs[n], label = str(mod).split('(')[0], ax = ax[n][0])
            precision_recall_curve(y_true, probs[n], cl=1, label = str(mod).split('(')[0], ax=ax[n][1])
            plot_confusion_matrix(y_true, preds[n], target_names=cm_target, standardize = st_cmat, 
                                  normalize=normalize, ax=ax[n][2])
        
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
    _ = ax[1].text(-2, y_max-0.5, 'Shap. p: {0:.4f}'.format(p))
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
    _ = ax.set_xticklabels(x.columns, rotation = 45, ha = 'right')
    _ = ax.set_title('Behavior of Model Scores Adding One Feature at Each Step')
        
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
    
    print('\033[1m'+'Total Explained Variance:'+'\033[0m'+
          ' {}'.format(sum(expl) * 100))
    
    return fig, ax

def scatter_clust_matrix(clust_res, 
                         x = '', 
                         y = '', 
                         clust_dict = {}, 
                         centroids = None,
                         to_del = None,
                         ax = None,
                         fig = None):
    
    # this works only with pandas dataframes
    # needs to be debugged
    
    if ax is None:
        fig, ax = plt.subplots()

    x = clust_res[x]
    y = clust_res[y]
    
    if not isinstance(clust_res, pd.core.frame.DataFrame):
        raise TypeError("clust_res should be a pandas DataFrame")
        
    flax = ax.flatten()
    
    for n, (cl, vals) in enumerate(clust_dict.items()):  
        _ = flax[n].grid(alpha=.3)
        _ = flax[n].set_xlabel(x)
        _ = flax[n].set_ylabel(y)
        
        for n_cl in vals:
            flax[n].scatter(clust_res[clust_res[cl] == n_cl][x], 
                            clust_res[clust_res[cl] == n_cl][y], 
                            label='Cluster {}'.format(n_cl+1))
        _ = flax[n].legend()
        _ = flax[n].set_title('K = {}'.format(len(vals)))
    
    if centroids is not None:
        for n, cents in enumerate(centroids):
            for i in range(len(cents)):
                _ = flax[n].scatter(centroids[n][i][0], 
                                    centroids[n][i][1], c='k', s=80)
        
    _ = fig.suptitle('Clusters Distribution by Changing the Number of K')
    _ = fig.set_tight_layout()
        
    if to_del is not None:
        if len(to_del) < 2:
            _ = fig.delaxes(flax[to_del])
        
        else:
            for dropax in to_del:
                _ = fig.delaxes(flax[dropax])
                
    return fig, ax

def plot_calibration_curve(y_true, 
                           prob, 
                           n_bins = 10, 
                           ax = None,
                           label = 'Model'):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
    
    ax1 = ax.twinx()
    bs = mtr.brier_score_loss(y_true, prob)
    frac_pos, mean_pred_val = calibration_curve(y_true, prob, n_bins = n_bins)
    ax.plot(np.sort(mean_pred_val), np.sort(frac_pos), 
             'o-', label = '{0} Brier: {1:.04f}'.format(label, bs), c = '#ff7f0e', zorder = 2)
    ax.plot([0,1], [0,1], linestyle = '--', c = 'k')
    ax1.hist(prob, alpha = .3, zorder = -1)
    _ = ax1.set_ylabel('#')
    _ = ax.legend(loc = 'upper center')
    _ = ax.grid(alpha = .3)
    _ = ax.set_title('{0} Calibration Curve, N. Bins: {1}'.format(label, n_bins))
    _ = ax.set_xlabel('Predicted Probability')
    _ = ax.set_ylabel('Fraction of Positive Samples')
    
    return fig, ax

def multimodel_calibration_plot(y_true, 
                                prob = [], 
                                models = [],
                                cal_bins = 10,
                                ax = None):
    
    if ax is None:
        fig, ax = plt.subplots(2,1, figsize = (15,10))
    else:
        fig = []
        
    ax[0] = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax[1] = plt.subplot2grid((3, 1), (2, 0))
        
    names = [mod.__class__.__name__ for mod in models]
    
    if isinstance(cal_bins, int):
        n_bins = [cal_bins]*len(models)

    mod_bins = {mod:bn for mod,bn in zip(names, n_bins)}
    for n, (mod, bn) in enumerate(mod_bins.items()):
        bs = mtr.brier_score_loss(y_true, prob[n])
        frac_pos, mean_pred_val = calibration_curve(y_true, prob[n], n_bins = bn)
        ax[0].plot(np.sort(mean_pred_val), np.sort(frac_pos), 
                   'o-', label = '{0}, Brier: {1:.04f}'.format(mod, bs))
        ax[0].plot([0,1], [0,1], lnestyle = '--', c = 'k', label = mod)
        ax[1].hist(prob[n], bins = np.arange(min(prob[n]), max(prob[n])+1), 
                   label=mod, histtype="step", lw=2)
        
    _ = ax[0].set_ylabel('Fraction of Positive Samples')
    _ = ax[0].grid(alpha = .3)
    _ = ax[0].legend(loc='lower right')
    _ = ax[0].set_title('Calibration plots  (reliability curve)')

    _ = ax[1].set_xlabel('Predicted Probabilities')
    _ = ax[1].set_ylabel('#')
    _ = ax[1].legend(loc='upper center', ncol=2)
    
    return fig, ax
    

    
            