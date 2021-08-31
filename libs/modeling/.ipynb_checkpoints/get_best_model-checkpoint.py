# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:25:40 2020

@author: Tobia Tommasini
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from math import floor, modf
from pandas.core.groupby.groupby import DataError
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, get_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as imPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def _check_smote(x_train,
                 y_train,
                 strategy = None,
                 random_state = None,
                 under_strategy = None):
    
    """If Imbalanced is set to True in Optimization functions, 
       it returns the over/undesampling strategy, computed on 
       x_train and y_train.
       
       Parameters
       ----------
       x_train : array-like - training input.
       
       y_train : array-like - training output.
       
       strategy : float, dict - oversampling strategy for SMOTE
       
       random_state : int - set a random seed.
       
       under_strategy : float, dict - undersampling strategy.
       
       Returns
       -------
       x_train : array-like - training input with over/undersampling performed.
       
       y_train :array-like - training output with over/undersampling performed.
    """
    
    if isinstance(strategy, dict) or isinstance(strategy, float):
        smote = SMOTE(sampling_strategy = strategy, random_state = random_state)
        
    else:
        smote = SMOTE(random_state = random_state)
    
    if under_strategy is not None:
        under = RandomUnderSampler(sampling_strategy = under_strategy, random_state = random_state)
        steps = [('o', smote), ('u', under)]
        ppl = imPipeline(steps = steps)
        x_train, y_train = ppl.fit_resample(x_train, y_train)
        
    else:
        x_train, y_train = smote.fit_sample(x_train, y_train)
        
    return x_train, y_train

def get_best_model(x, 
                   y,
                   test_size = .3,
                   random_state = None,
                   normalize = True,
                   goal = 'Classification',
                   cv = None,
                   refit = True,
                   imbalanced = False,
                   strategy = None,
                   under_strategy = None,
                   models = [],
                   params = [],
                   n_iter = 100,
                   score = 'neg_mean_squared_error'):
    
    """Performs hyperparameter optimization using randomized 
       search for a list of selected models. Works for both 
       classification and regression problems and can take 
       into account standard scaling of features and over/
       undersampling strategies.
       
       Parameters
       ----------
       x : array-like - Input data.
       
       y : array-like - Output data.
       
       test_size : float - size of test set.
       
       random_state : int - set a random seed.
       
       normalize : bool - if true, standard scaling will be performed on data.
       
       goal : str - classification or regression.
       
       cv : int - folds for cross-validation.
       
       refit : bool, str o callable - if not false, refit the estimator using 
       the best found parameters.
       
       imbalanced : bool - if true, over/undersampled will be performed.
       
       strategy : float, dict - oversampling strategy for SMOTE.
       
       under_strategy : float, dict - undersampling strategy.
       
       models : list - models to be hyperparametrized.
       
       params : list - parameters to be optimized.
       
       n_iter : int - number of iterations.
       
       score : str, callable, list, dict - score(s) to evaluate the prediction.
       
       Returns
       -------
       bm : dict - dictionary with best models as keys and classification
       reports as values.
       
       pred : dict - best models as keys and predicted values as values
       
       prob : dict - best models as keys and predicted probabilities as values.
       
       scores : dict - best models as keys and scores as values - returned only 
       if refit is not bool.
    """
    
    if normalize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, 
                                                        random_state = random_state)
    
    if imbalanced:
        x_train, y_train = _check_smote(x_train, y_train, strategy = strategy, 
                                        under_strategy = under_strategy, random_state = random_state)
            
    prob = {}
    preds = {}
    best_models = {}
    
    if not isinstance(refit, bool):
        scores = {}
    for n, mod in enumerate(models):
        gr = RandomizedSearchCV(mod, params[n], refit = refit, 
                                random_state = random_state, cv = cv, n_iter = n_iter, scoring = score)
        _ = gr.fit(x_train, y_train)
        pred = gr.best_estimator_.predict(x_test)
        preds[gr.best_estimator_] = gr.best_estimator_.predict(x_test)
        
        if goal == 'Classification':
            cr = classification_report(y_test, pred, output_dict = True)
            best_models[gr.best_estimator_] = cr
            prob[gr.best_estimator_] = gr.best_estimator_.predict_proba(x_test)
            if not isinstance(refit, bool):
                scores[gr.best_estimator_] = gr.best_score_
            
        elif goal == 'Regression':
            scr = mean_squared_error(y_test, pred)
            best_models[gr.best_estimator_] = scr
            if not isinstance(refit, bool):
                scores[gr.best_estimator_] = gr.best_score_
            
    if not isinstance(refit, bool) and goal == 'classification':
        return scores, best_models, preds, prob
    
    elif isinstance(refit, bool) and goal == 'classification':
        return best_models, preds, prob
    
    else:
        return best_models, preds
    
def get_model_score(x,
                    y,
                    models = [],
                    test_size = .3,
                    normalize = True,
                    goal = 'Classification',
                    imbalanced = False,
                    strategy = None,
                    random_state = None,
                    under_strategy = None):
    
    """Trains a list of models and compute the classification
       report for classification problems and mean squared
       error for regression. Can handle standard scaling
       and over/undersampling strategy.
       
       Parameters
       ----------
       x : array-like - input data.
       
       y : array-like - output data.
       
       models : list - models to be trained.
       
       test_size : float - size of test set.
       
       normalize : bool - if true, standard scaling is performed.
       
       goal : str - could be classification or regression.
       
       imbalanced : bool - if true, over/undersampled will be performed.
       
       strategy : float, dict - oversampling strategy for SMOTE.
       
       random_state : int - set a random seed.
       
       under_strategy : float, dict - undersampling strategy.
       
       Returns
       -------
       cr : dict - dictionary with models as keys and classification
       report as values.
    """
    
    if normalize:
        sd = StandardScaler()
        
        if goal == 'Classification':
            x = sd.fit_transform(x)
        
        elif goal == 'Regression':
            x = sd.fit_transform(x)
            y = sd.fit_transform(y.reshape(-1, 1))
        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, 
                                                        random_state = random_state)
    
    if imbalanced:
        x_train, y_train = _check_smote(x_train, y_train, strategy = strategy, 
                                        under_strategy = under_strategy, random_state = random_state)
        
    cr = {}
    for mod in models:
        _ = mod.fit(x_train, y_train)
        pred = mod.predict(x_test)
        prob = mod.predict_proba(x_test)
        if goal == 'Classification':
            cr[mod] = classification_report(y_test, pred, output_dict = True)
        
        elif goal == 'Regression':
            cr[mod] = mean_squared_error(y_test, pred)
            
    return cr, pred, prob
        
def get_report_df(rep, 
                  be = False):
    
    """Takes the output of get_best_model and rearranges
       the dictionaries to show the reports in a more readable 
       format.
       
       Parameters
       ----------
       rep : dict - results of RandomizedSearchCV computed on a list 
       of models (get_best_model output).
       
       be : bool - set it to true if you are passing a composed 
       estimator (es: CalibratedClassifierCV)
       
       Returns
       -------
       cl_scores : DataFrame - results of classification report for each
       class and a column for accuracy.
    """
    
    rp = {}
    for key, val in zip(rep.keys(), rep.values()):
        if be:
            rp[key.base_estimator.__class__.__name__] = val
        else:
            rp[key.__class__.__name__] = val
    
    crs = []
    models = []
    for mds, cr in rp.items():
        models.append(mds)
        crs.append(pd.DataFrame(cr))
        
    ind = np.array([[mod]*crs[0].shape[0] for mod in models]).flatten()
    
    full_crs = pd.concat(crs, axis=0)
    full_crs['Model'] = ind
    full_crs = full_crs.reset_index().rename({'index':'score'}, axis=1)
    
    max_classes = [str(cl) for cl in range(0,100)]
    
    try:
        vals = [num_class for num_class in full_crs.columns if num_class in max_classes]
        cl_scores = pd.pivot_table(full_crs, index='Model', columns='score', values=vals)
    except DataError:
        cols = [str(col) for col in full_crs.columns]
        vals = [num_class for num_class in cols if num_class in max_classes]
        vals = [int(cl) for cl in vals]
        cl_scores = pd.pivot_table(full_crs, index='Model', columns='score', values=vals)
    
    acc = full_crs.groupby('Model').agg({'accuracy':'mean'})
    cl_scores['accuracy'] = acc['accuracy']
    
    return cl_scores

def _score_renamer():
    
    """Renames the most used classification scores. 
       
       Returns
       -------
       renamer : dict - maps original score's name to new name.
    """
    
    renamer = {'f1_macro':'F1 Macro',
               'f1_micro':'F1 Micro',
               'f1_samples':'F1 Samples',
               'f1_weighted':'F1 Weighted',
               'accuracy':'Accuracy',
               'balanced_accuracy':'Balanced Accuracy',
               'recall':'Recall',
               'precision':'Precision',
               'average_precision':'Average Precision',
               'roc_auc':'ROC AUC',
               'roc_auc_ovr':'ROC AUC One vs Rest',
               'roc_auc_ovo':'ROC AUC One vs One'
              }
    
    return renamer

def _score_mapper(y_test, 
                  pred, 
                  prob = None, 
                  pos_label = 1):
    
    """Maps a score from a string to a function.
       
       Parameters
       ----------
       y_test : array-like - true values of outcome.
       
       pred : array-like - predicted values of outcome.
       
       prob : array-like - predicted probabilities. Is not None,
       roc-auc is computed.
       
       pos_label : int, str - label of positive class.
       
       Returns:
       --------
       mapper : dict - the mapping from string to function.
    """
    
    import sklearn.metrics as mtr
    
    def get_support(y_test, pos_label = 1):
        return np.bincount(y_test)[pos_label]
    
    try:
        single_scores = {'f1':mtr.f1_score(y_test, pred, pos_label = pos_label),
                         'precision':mtr.precision_score(y_test, pred, pos_label = pos_label),
                         'recall':mtr.recall_score(y_test, pred, pos_label = pos_label)}
    except ValueError:
        single_scores = {'f1':mtr.f1_score(y_test, pred, average = None),
                         'precision':mtr.precision_score(y_test, pred, average = None),
                         'recall':mtr.recall_score(y_test, pred, average = None)}
    
    mapper = {# 'f1':mtr.f1_score(y_test, pred, pos_label = pos_label),
              'f1_macro':mtr.f1_score(y_test, pred, average = 'macro'),
              'f1_micro':mtr.f1_score(y_test, pred, average = 'micro'),
              'f1_weighted':mtr.f1_score(y_test, pred, average = 'weighted'),
              # 'precision':mtr.precision_score(y_test, pred, pos_label = pos_label),
              'precision_macro':mtr.precision_score(y_test, pred, average = 'macro'),
              'precision_micro':mtr.precision_score(y_test, pred, average = 'micro'),
              'precision_weighted':mtr.precision_score(y_test, pred, average = 'weighted'),
              # 'recall':mtr.recall_score(y_test, pred, pos_label = pos_label),
              'recall_macro':mtr.recall_score(y_test, pred, average = 'macro'),
              'recall_micro':mtr.recall_score(y_test, pred, average = 'micro'),
              'recall_weighted':mtr.recall_score(y_test, pred, average = 'weighted'),
              'accuracy':mtr.accuracy_score(y_test, pred),
              'balanced_accuracy':mtr.balanced_accuracy_score(y_test, pred),
              'support':get_support(y_test, pos_label = pos_label)}
    
    mapper = {**single_scores, **mapper}
    if prob is not None:
        try:
            roc_map = {'roc_auc':mtr.roc_auc_score(y_test, prob[:, 1]),
                       'roc_auc_ovr':mtr.roc_auc_score(y_test, prob, multi_class = 'ovr'),
                       'roc_auc_ovo':mtr.roc_auc_score(y_test, prob, multi_class = 'ovo')}
        except ValueError:
            roc_map = {'roc_auc_ovr':mtr.roc_auc_score(y_test, prob, multi_class = 'ovr'),
                       'roc_auc_ovo':mtr.roc_auc_score(y_test, prob, multi_class = 'ovo')}
        mapper = {**mapper, **roc_map}
        
    return mapper

def _smote_cv_score(x, 
                    y, 
                    model = None,
                    strategy = None,
                    under_strategy = None,
                    random_state = None,
                    folds = 5, 
                    scoring = 'roc_auc'):
    
    if isinstance(x, pd.core.frame.DataFrame):
        x = x.values
        y = y.values
        
    kf = KFold(n_splits = folds)
    
    if isinstance(strategy, dict) or isinstance(strategy, float):
        smote = SMOTE(sampling_strategy = strategy, random_state = random_state)
    else:
        smote = SMOTE(random_state = random_state)
        
    if under_strategy is not None:
        under = RandomUnderSampler(sampling_strategy = under_strategy, random_state = random_state)
        steps = [('o', smote), ('u', under)]
        ppl = imPipeline(steps = steps)
    
    scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        if under_strategy is not None:
            x_train, y_train = ppl.fit_resample(x_train, y_train)
        else:
            x_train, y_train = smote.fit_sample(x_train, y_train)
            
        _ = model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        
        score = get_scorer(scoring)(model, x_test, y_test)
        scores.append(score)
        
    return scores

def smote_cv_score(x, 
                   y, 
                   model = None,
                   stratified = False,
                   imbalanced = True,
                   strategy = None,
                   under_strategy = None,
                   random_state = None,
                   shuffle = False,
                   folds = 5, 
                   scoring = 'roc_auc',
                   pos_label = 1):
    
    """Performs cross validation for the selected score(s)
       taking into account the over/undersampling strategy.
       
       Parameters
       ----------
       x : array-like - input data.
       
       y : array-like - output data.
       
       model : base_estimator - the model you want to cross-validate.
       
       stratified : bool - if True, stratified k-fold is considered.
       
       imbalanced : bool - if true, over/undesampling strategy is considered.
       
       strategy : float, dict - oversampling strategy for SMOTE.
       
       under_strategy : float, dict - undersampling strategy.
       
       random_state : int - set a random seed.
       
       folds : int - folds for cross-validation.
       
       scoring : str, callable - the score you want to cross-validate.
       
       Returns
       -------
       scores : list - the selected score computed for each fold.
    """
    
    if isinstance(x, pd.core.frame.DataFrame) or isinstance(y, pd.core.series.Series):
        x = x.values
        y = y.values
        
    if stratified:
        if shuffle:
            kf = StratifiedKFold(n_splits = folds, 
                                 shuffle = shuffle, 
                                 random_state = random_state)
        else:
            kf = StratifiedKFold(n_splits = folds, 
                                 random_state = None)
        spl = kf.get_n_splits(x, y)
    else:
        if shuffle:
            kf = KFold(n_splits = folds, 
                       shuffle = shuffle, 
                       random_state = random_state)
        else:
            kf = KFold(n_splits = folds,
                       random_state = None)
        spl = kf.split(x)
    
    if imbalanced:
        if isinstance(strategy, dict) or isinstance(strategy, float):
            smote = SMOTE(sampling_strategy = strategy, random_state = random_state)
        else:
            smote = SMOTE(random_state = random_state)
            
        if under_strategy is not None:
            under = RandomUnderSampler(sampling_strategy = under_strategy, random_state = random_state)
            steps = [('o', smote), ('u', under)]
            ppl = imPipeline(steps = steps)
    
    scores = []
    for train_index, test_index in spl:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        if imbalanced:
            if under_strategy is not None:
                x_train, y_train = ppl.fit_resample(x_train, y_train)
            else:
                x_train, y_train = smote.fit_sample(x_train, y_train)
    
        _ = model.fit(x_train, y_train)
        pred = model.predict(x_test)
    
        if scoring in ['roc_auc', 'roc_auc_ovr', 'roc_auc_ovo']:
            prob = model.predict_proba(x_test)
            score = _score_mapper(y_test, pred, prob = prob)[scoring]
        else:
            score = _score_mapper(y_test, pred, pos_label = pos_label)[scoring]
            scores.append(score)
    
    return scores

def standard_classifier_builder(random_state = 10, 
                                no_conv_risk = False):
    
    """Prepares LogisticRegression, RandomForestClassifier, 
       GradientBoostingClassifier and XGBoost for the RandomizedSearchCV 
       process.
       
       Parameters
       ----------
       random_state : int - set a random seed.
       
       no_conv_risk : bool - set it to true if you think there is a risk that
       the logistic regression will not converge.
       
       Returns
       -------
       models : list - the four models.
       
       params : list - the parameters to be optimized for the models.
    """
    
    lr = LogisticRegression(random_state = random_state, max_iter=10000)
    rf = RandomForestClassifier(random_state = random_state)
    gb = GradientBoostingClassifier(random_state = random_state)
    xb = xgb.XGBClassifier(random_state = random_state)
    models = [lr, rf, gb, xb]
    
    if no_conv_risk:
        lr_pars = {
        'class_weight': [None, 'balanced'],
        'C':[0.01, 0.1, 1.0],
        }
        
    else:
        lr_pars = {
        'class_weight': [None, 'balanced'],
        'solver':['liblinear', 'saga'],
        'C':[0.01, 0.1, 1.0],
        'penalty':['l1','l2']
        }
    
    params = [lr_pars, 
       {
        'n_estimators':np.arange(10,100,10),
        'criterion':['gini','entropy'],
        'min_samples_leaf':np.arange(1,11),
        'max_features':['auto','sqrt','log2'],
        'bootstrap':[True, False],
        'class_weight':[None, 'balanced', 'balanced_subsample'] 
    }, {
        'n_estimators':np.arange(10,100,10),
        'learning_rate':np.arange(.1,1),
        'min_samples_leaf':np.arange(1,11),
        'max_features':['auto','sqrt','log2']
    }, {
        'learning_rate':np.arange(.1,1),
        'max_depth':np.arange(1,11),
        'min_child_weight':np.arange(1,11),
        'gamma':np.arange(.01,.1),
        'colsample_bytree':np.arange(.1,1),
    }]
        
    return models, params

def get_prob_calibration(x,
                         y,
                         test_size = .3,
                         random_state = 10,
                         normalize = True,
                         models = [], 
                         method = 'isotonic', 
                         cv = 10,
                         imbalanced = False,
                         strategy = None,
                         under_strategy = None):
    
    """Performs the calibration of probabilities for a
       list of selected models.
       
       Parameters
       ----------
       x : array-like - input data.
       
       y : array-like - output data.
       
       test_size : float - size of test set.
       
       random_state : int - set a random seed.
       
       normalize : bool - if true, standard scaling will be performed.
       
       models : list - list of models to be calibrated.
       
       method : str - isotonic or logistic - method for calibration.
       
       cv : int - number of folds for cross-validation.
       
       imbalanced : bool - if true, over/undersampled will be performed.
       
       strategy : float, dict - oversampling strategy for SMOTE.
       
       under_strategy : float, dict - undersampling strategy.
       
       Raturns
       -------
       calib_probs : dict - dictionary with models as keys and calibrated
       probabilities as values.
    """
    
    if normalize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, 
                                                        random_state = random_state)
    
    if imbalanced:
        x_train, y_train = _check_smote(x_train, y_train, strategy = strategy, 
                                        under_strategy = under_strategy, random_state = random_state)
    
    calib_probs = {}
    for mod in models:
        calibration = CalibratedClassifierCV(mod, cv = cv, method = method)
        _ = calibration.fit(x_train, y_train)
        prob = calibration.predict_proba(x_test)
        calib_probs[mod] = prob
    
    return calib_probs

def cross_val_report(x,
                     y,
                     folds = 10,
                     models = [],
                     normalize = True,
                     scores = ['f1', 'precision', 'recall', 'support'],
                     imbalanced = False,
                     strategy = None,
                     under_strategy = None,
                     mean_cv = False,
                     random_state = 10):
    
    """Returns a cross-validated classification report (as a dict). 
       Can handle standard scaling and over/undersampling strategy.
    """
    
    if normalize:
        sd = StandardScaler()
        x = sd.fit_transform(x)
    
    classes = np.unique(y)
    avgs = ['macro avg', 'micro avg', 'weighted avg']
    macro_scores = ['f1_macro', 'precision_macro', 'recall_macro']
    micro_scores = ['f1_micro', 'precision_micro', 'recall_micro']
    weighted_scores = ['f1_weighted', 'precision_weighted', 'recall_weighted']
    
    cv_scores = {}
    for mod in models:
        rep = {}
        for cl in classes: 
            rep[cl] = {score:smote_cv_score(x, y, scoring = score,
                                            random_state = random_state,
                                            imbalanced = imbalanced,
                                            model = mod, folds = folds, 
                                            pos_label=cl) for score in scores}
            if mean_cv:
                for score in scores:
                    if score == 'support':
                        rep[cl][score] = np.mean(rep[cl][score])
                        if modf(rep[cl][score])[0] > 0.5:
                            rep[cl][score] = round(rep[cl][score]) 
                        else:
                            rep[cl][score] = floor(np.mean(rep[cl][score]))
                    else:
                        rep[cl][score] = np.mean(rep[cl][score])
        
        for avg, avg_score in zip(avgs, [macro_scores, micro_scores, weighted_scores]):
            rep[avg] = {score.split('_')[0]:smote_cv_score(x, y, 
                                                           scoring = score, 
                                                           random_state = random_state,
                                                           model = mod, 
                                                           folds = folds) for score in avg_score}
            
            rep['accuracy'] = smote_cv_score(x, y, scoring = 'accuracy',
                                             random_state = random_state, model = mod, folds = folds)
            
            if mean_cv:
                for s in avg_score:
                    rep[avg][s.split('_')[0]] = np.mean(rep[avg][s.split('_')[0]])
                
                rep['accuracy'] = np.mean(rep['accuracy'])
        
        ords = [i for i in classes] + avgs + ['accuracy']
        rep = {k : rep[k] for k in ords}  
        cv_scores[mod] = rep
    
    return cv_scores
    
    

        
        
    
    
    
    
        