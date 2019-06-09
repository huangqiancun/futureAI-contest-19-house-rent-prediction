# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:38:42 2019

@author: huangqiancun
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product
from scipy import stats
from scipy.stats import skew, kurtosis
def get_mode(row):
    return stats.mode(row)[0][0]  
def get_unique_num(row):
    if len(row) == 0:
        return 0
    return len(np.unique(row))
def get_len(row):
    return len(row)
def get_unique_freq(row):
    if len(row) == 0:
        return 0
    return float(len(np.unique(row))) / len(row)
def get_max(row):
    return np.max(row)
def get_min(row):
    return np.min(row)
def get_mean(row):
    return np.mean(row)
def get_std(row):
    return np.std(row)
def get_skew(row):
    return skew(row)
class MeanEncoder:
    def __init__(self, categorical_features,stats="mean", flag="mean",n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
        :param n_splits: the number of splits used in mean encoding
        :param target_type: str, 'regression' or 'classification'
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        '''
        >>>example:
        mean_encoder = MeanEncoder(
                        categorical_features=['regionidcity',
                          'regionidneighborhood', 'regionidzip'],
                target_type='regression'
                )

        X = mean_encoder.fit_transform(X, pd.Series(y))
        X_test = mean_encoder.transform(X_test)


        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
        self.stats=stats
        self.flag=flag
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func,stats,flag):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
       
        if target is not None:
            nf_name = '{}_pred_{}_{}'.format(variable, target,stats)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred_{}'.format(variable,stats)
            X_train['pred_temp'] = y_train  # regression
#         prior = X_train['pred_temp'].mean()
        stats_dict={"mean":get_mean,
                    "min":get_min,
                    "max":get_max,
                    "std":get_std,
                    "skew":get_skew,
                    "mode":get_mode,
                    "kurt":kurtosis,
                    "unique":get_unique_num,
                    "freq":get_unique_freq
                   
                   } 
        if stats=="mode":
            col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({stats: get_mode, 'beta': 'size'})
        if stats=="kurt":
            col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({stats: kurtosis, 'beta': 'size'})
        if stats=="unique":
            prior = get_unique_num(X_train['pred_temp'])
            col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({stats: get_unique_num, 'beta': 'size'})
        if stats=="freq":
            prior = get_unique_freq(X_train['pred_temp'])
            col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({stats: get_unique_freq, 'beta': 'size'})
        
        if stats in ["mean","max","min","sum","std","var","median","skew"]: 
            col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({stats: stats, 'beta': 'size'})
            
        if flag=="mean":
            prior = stats_dict['mean'](X_train['pred_temp'])
        elif flag=="median":
            prior = X_train['pred_temp'].median()
        else:
            prior = stats_dict[stats](X_train['pred_temp'])
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y[stats]
        col_avg_y.drop(['beta', stats], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        stats=self.stats
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}_{}'.format(variable, target,stats): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}_{}'.format(variable, target,stats)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func,self.stats,self.flag)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred_{}'.format(variable,stats): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred_{}'.format(variable,stats)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func,self.stats,self.flag)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        stats=self.stats
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}_{}'.format(variable, target,stats)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred_{}'.format(variable,stats)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new