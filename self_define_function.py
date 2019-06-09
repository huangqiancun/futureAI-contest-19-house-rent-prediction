# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:39:33 2019

@author: huangqiancun
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def feature_importance(feature_importance_df):
    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:1000].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    
    feature_rank = best_features.sort_values(by="importance",ascending=False)
    plt.figure(figsize=(14,40))
    sns.barplot(x="importance",
                y="feature",
                data=feature_rank)
    
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    
    return cols


def lgb_eval(params, train, target, test):
    
    features = train.columns
    
    folds = KFold(n_splits=5, shuffle=True, random_state=2333)

    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], 
                        verbose_eval=2000, early_stopping_rounds = 200)

        oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

    print("CV Score: {:<8.5f}".format(r2_score(target, oof_lgb)))
    score = 1- np.sum(np.square(target - oof_lgb)) / np.sum(np.square(target - np.mean(target)))
    print("Score: {:<8.5f}".format(score))

    return score, feature_importance_df, predictions_lgb, oof_lgb