import math
import re
from typing import List
import multiprocessing
from multiprocessing import Process, Pool

import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error \
    , accuracy_score, classification_report, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.linear_model import ElasticNet, Lasso, MultiTaskElasticNet
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel, SelectFpr, SequentialFeatureSelector, RFE
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from imblearn.over_sampling import RandomOverSampler

import seaborn as sb


from statsmodels.nonparametric.smoothers_lowess import lowess

def mmp(x, prob, actual, title):
    """
    Marginal Model Plots for Binary Model Fit Validation
    Based on methodology established by Weisberg, 2005.
    """
    yhat = lowess(endog=prob, exog=x, frac=.2, it=0)
    x1 = yhat[:,0]
    y1 = yhat[:,1]

    Y = lowess(endog=actual, exog=x, frac=.2, it=0)
    x2 = Y[:,0]
    y2 = Y[:,1]
    
    fig1, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x1, y1, color='blue', alpha=0.7)
    ax2.plot(x2, y2, color='red', alpha=0.7)
    ax1.set_title(title)
    ax1.set_ylabel('Actual + Prob')
    ax1.set_xlabel('X')
    plt.show()

def get_balanced_accuracy(tpr, fpr):
    """
    Return average of Sensitivity and Specificity.
    """
    return (tpr + (1-fpr)) / 2

def get_tpr_fpr(cm):
    """
    Sensitivity: TruePos / (True Pos + False Neg) 
    Specificity: True Neg / (False Pos + True Neg)
    TN | FP
    -------
    FN | TP
    @param 2D array <list<list>>
    @returns <list<float>>
    """
    tn = float(cm[0][0])
    fp = float(cm[0][1])
    fn = float(cm[1][0])
    tp = float(cm[1][1])

    pos = (tp + fn)
    neg = (fp + tn)
    tpr = 0 if pos == 0 else (tp/pos)
    fpr = 0 if neg == 0 else 1-(tn/neg)

    return [tpr, fpr]

def get_best_cutoff(actual, prob):  
    """
    Get the best cutoff according to Balanced Accuracy
    'Brute-force' technique - try all cutoffs from 0.01 to 0.99 in increments of 0.01

    @param actual <list<float>>
    @param prob <list<tuple<float, float>>>
    @returns <list<float>>
    """
    best_tpr = 0.0; best_fpr = 0.0; best_cutoff = 0.0; best_ba = 0.0; 
    cutoff = 0.0
    cm = [[0,0],[0,0]]
    while cutoff < 1.0:
        pred = list(map(lambda p: 1 if p >= cutoff else 0, prob))
        _cm = confusion_matrix(actual, pred)
        _tpr, _fpr = get_tpr_fpr(_cm)

        if(_tpr < 1.0):    
            ba = get_balanced_accuracy(tpr=_tpr, fpr=_fpr)

            if(ba > best_ba):
                best_ba = ba
                best_cutoff = cutoff
                best_tpr = _tpr
                best_fpr = _fpr
                cm = _cm

        cutoff += 0.01

    tn = cm[0][0]; fp = cm[0][1]; fn = cm[1][0]; tp = cm[1][1];
    return [best_tpr, best_fpr, best_cutoff, tn, fp, fn, tp]
    
# create confusion matrix
def score_model(actual, prob, model_name='Logit'):
    """
    Compute predicted based on estimated probabilities and best threshold. 
    Output predictions and confusion matrix.
    """
    # calculate TPR, FPR, best probability threshold
    tpr, fpr, cutoff, tn, fp, fn, tp = get_best_cutoff(actual, prob)
    yhat = list(map(lambda p: 1 if p >= cutoff else 0, prob))  
    stats = {
        'Model': model_name
        , 'TP': tp
        , 'FP': fp
        , 'TN': tn
        , 'FN': fn
        , 'Sensitivity': tpr
        , 'Specificity': (1-fpr)
        , 'Cutoff': cutoff
        , 'Accuracy': get_balanced_accuracy(tpr, fpr)
        , 'F1 Score': f1_score(actual, yhat)
        , 'Recall Score': recall_score(actual, yhat)
        , 'AUC': roc_auc_score(actual, prob)
    }
    return stats

def score_model_non_prob(actual, yhat, model_name='Logit'):
    """
    Compute predicted based on estimated probabilities and best threshold. 
    Output predictions and confusion matrix.
    """
    # calculate TPR, FPR, best probability threshold
    #fpr, tpr, cutoffs = roc_curve(actual, yhat) 
    cm = confusion_matrix(actual, yhat)
    tpr, fpr = get_tpr_fpr(cm)
    ba = get_balanced_accuracy(tpr, fpr)
    tn = float(cm[0][0])
    fp = float(cm[0][1])
    fn = float(cm[1][0])
    tp = float(cm[1][1])

    stats = {
        'Model': model_name
        , 'TP': tp
        , 'FP': fp
        , 'TN': tn
        , 'FN': fn
        , 'Sensitivity': tpr
        , 'Specificity': (1-fpr)
        , 'Cutoff': 0.5
        , 'Accuracy': ba
        , 'F1 Score': f1_score(actual, yhat)
        , 'Recall Score': recall_score(actual, yhat)
        , 'AUC': roc_auc_score(actual, yhat)
    }
    return stats

def plot_roc(actual, prob):
    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(actual, prob)

    # plot ROC curve
    fig = plt.figure(figsize=(10, 10))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.show()
    
def safe_log(col_data):
    return list(map(lambda x: 0 if x == 0 else math.log(x), col_data.values))

def get_numeric_columns(df):
    return list(df.describe().columns)

def display_box_plots(df, groupby=None):
    
    def _fn(col):
        if groupby == None:
            ax = df.boxplot(column=col)
        else:
            ax = df.boxplot(column=col, by=groupby)
            
        ax.set_title(col)
        plt.show()
            
    num_cols = get_numeric_columns(df)
    _ = list(map(lambda col: _fn(col), num_cols))

def get_corrs(df, figsize=(10, 10)):
    num_cols = get_numeric_columns(df)
    #Compute Percentage Change
    rets = df[num_cols].pct_change()
    #Compute Correlation
    corr = rets.corr()
    #Plot Correlation Matrix using Matplotlib
    #%pylab inline
    plt.figure(figsize=figsize)
    plt.imshow(corr, cmap='Dark2', interpolation='none', aspect='auto') 
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr)), corr.columns);
    plt.suptitle('Correlations Heat Map', fontsize=15, fontweight='bold')
    plt.show()
    
def replace_missing(data, ceiling=0.05):
    def _impute(col_name):
        _type = str(data[col_name].dtype)
        if re.match(r'^(int|float)', _type):
            data[col_name].fillna(data[col_name].median(), inplace=True)
        else:
            data[col_name].fillna(data[col_name].mode().to_list()[0], inplace=True)
    
    df_missing = pd.DataFrame(columns=['col_name'], data=data.columns)
    perc_missing = list(data.isnull().sum() / len(data) )
    df_missing['missing'] = perc_missing
    df_missing.sort_values('missing', ascending=False, inplace=True)
    df_missing.query('missing > 0', inplace=True)
    
    if len(df_missing) > 0:
        for row in df_missing.values:
            col_name = str(row[0])
            missing = float(row[1])
            if missing <= ceiling:
                _impute(col_name)
            else:
                data.drop(labels=[col_name], axis=1, inplace=True)
        
        print(df_missing.head(25))
        replace_missing(data, ceiling)
        
    else:
        print('Missing values were resolved.')
        
def split_train_val(data, y_col, train_size=0.7):
    features = list(data.columns)
    if y_col in features:
        features.remove(y_col)
        
    df_train = data.sample(frac=train_size, random_state=42)
    df_val = data.drop(df_train.index, inplace=False)
    x_train = df_train[features]
    y_train = df_train[y_col]
    x_val = df_val[features]
    y_val = df_val[y_col]
    return x_train, y_train, x_val, y_val, features

def get_n_features_to_select(X):
    """
        A general rule of thumb is to use as many features as 
        a square root of the number of observations.
    """
    n_obs, n_cols = X.shape
    n_features_to_select = math.floor( math.sqrt(n_obs) )
    return n_features_to_select if n_features_to_select < n_cols else n_cols
        
def get_train_val_data(data:DataFrame, y:str, target_columns:List, train_size:float=0.7, normalize=True):
    df = data.copy()
    to_drop = target_columns.copy()
    if y in to_drop:
        to_drop.remove(y)
    
    try:
        df.drop(labels=to_drop, axis=1, inplace=True)
    except:
        pass
    
    x_train, y_train, x_val, y_val, features = split_train_val(df, y, train_size)
    pos_orig = len([y for y in y_train if y == 1])
    neg_orig = len(y_train) - pos_orig
    
    if normalize:
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.fit_transform(x_val)
    
    pos_ratio = round( pos_orig/len(y_train), 6 )
    oversample = True if pos_ratio < 0.02 else False
    
    if oversample:
        ros = RandomOverSampler(random_state=42)
        x_over, y_over = ros.fit_resample(x_train, y_train)
        #pos = len([y for y in y_over if y == 1])
        #neg = len(y_over) - pos
        #print(f'Before ROS - #True: {pos_orig} #False: {neg_orig} Ratio: {pos_ratio} Num rows: {len(y_train)}')
        #print(f'After ROS  - #True: {pos} #False: {neg} Ratio: {pos/len(y_over)} Num rows: {len(y_over)}')
        return x_over, y_over, x_val, y_val, features
    
    return x_train, y_train, x_val, y_val, features

def get_selector(X, y, max_iter=10000, is_lasso=True, random_state=42) -> SelectFromModel:
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
        #'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0],
        'fit_intercept': [True, False]
    }
    estimator = ( Lasso(max_iter=max_iter, random_state=random_state) if is_lasso 
                 else ElasticNet(max_iter=max_iter, random_state=random_state) ).fit(X, y)
    
    grid = GridSearchCV(estimator=estimator, param_grid=params, n_jobs=-1)
    grid.fit(X, y)
    sfm = SelectFromModel(grid.best_estimator_)
    #sfm.fit(X, y)
    return sfm

def get_elasticnet_selector(X, y, random_state=42) -> SelectFromModel:
    return get_selector(X=X, y=y, is_lasso=False)

def get_lasso_selector(X, y, random_state=42) -> SelectFromModel:
    return get_selector(X=X, y=y)

def get_features_names(sfm: SelectFromModel, features: List) -> List:
    return sfm.get_feature_names_out(features)

def get_selectkbest(X, y):
    n_features_to_select = get_n_features_to_select(X)
    return SelectKBest(k=n_features_to_select)

def get_rfe_selector(X, y, random_state=42):
    n_features_to_select = get_n_features_to_select(X)
    return RFE(RandomForestClassifier(random_state=random_state), n_features_to_select=n_features_to_select)

def get_seq_feature_selector(X, y, random_state=42):
    n_features_to_select = get_n_features_to_select(X)
    return SequentialFeatureSelector(
        RandomForestClassifier(random_state=random_state), 
        direction='forward', 
        scoring='recall', 
        n_jobs=-1
    )

def plot_coef(features, coef_list, title=None):
    data = [[row[0], row[1]] for row in zip(features, coef_list)]
    df_features = pd.DataFrame(columns=['X', 'Coef'], data=data)
    df_features.sort_values('Coef', ascending=True, inplace=True)
    fig_ht = math.ceil(len(features)*0.3)
    x_ticks_labels = df_features.X.values   
    ax = df_features.plot(kind='barh', figsize=(10, fig_ht), title=title)
    ax.set_yticklabels(x_ticks_labels, fontsize=10)
    ax.plot()