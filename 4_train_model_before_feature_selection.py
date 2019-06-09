# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:23:47 2019

@author: huangqiancun
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import self_define_function #自定义函数

combine = pd.read_csv('output/test-a/data_featureEngineering_2.csv')


train = combine[combine.tradeMoney.notnull()]
test = combine[combine.tradeMoney.isnull()]


params = {
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'min_child_samples':20,
    'objective': 'regression',
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.85,
    "bagging_seed": 23,
    "metric": 'rmse',
    "lambda_l1": 0.2,
    "nthread": 4,
}


# baseline  0.90952
target = train.pop('tradeMoney')
test.drop(['tradeMoney'], axis = 1, inplace = True)

object_feats = [x for x in train.select_dtypes(include = ['object']).columns]
for col in object_feats:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    
score, feature_importance_df,predictions_lgb, oog_lgb = self_define_function.lgb_eval(params, train, target, test)

feature_importance = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)
#feature_importance = self_define_function.feature_importance(feature_importance_df)


pd.DataFrame(predictions_lgb).apply(round).to_csv('submit/submit-baseline.csv',na_rep='\\n',index=False,encoding='utf8',header=False)

# =============================================================================
# 15个重要特征进行特征交叉
# =============================================================================
def add(x, y):
    return x + y
def substract(x, y):
    return x - y
def times(x, y):
    return x * y
def divide(x, y):
    return (x + 0.001)/(y + 0.001)
CrossMethod = {
#                '+':add,
#                '-':substract,
                '*': times,
                '/': divide, 
                }
def get_Cross_feat(Cross_col, df):
    df_after = pd.DataFrame()
    for i in range(len(Cross_col)):
        for j in range(i+1, len(Cross_col)):
            for k in CrossMethod:
                df_after[Cross_col[i]+k+Cross_col[j]] = CrossMethod[k](df[Cross_col[i]], df[Cross_col[j]])
                print(Cross_col[i]+k+Cross_col[j],"done")
    return df_after

imp_feature = feature_importance[:15]
#imp_feature = ['area', 'per_room', 'communityName_pred_mean', 'area_mean',
#       'communityName_pred_max', 'communityName_pred_min', 'tradeTime',
#       'tradeDay_pred_std', 'communityName_pred_std', 'tradeDay_pred_freq',
#       'tradeDay_pred_mean', 'area_skew', 'area_min', 'houseType', 'day_skew',
#       'tradeDay_pred_kurt', 'tradeMeanPrice', 'client', 'tradeDay_pred_skew',
#       'plate_pred_mean', 'area_std', 'month_std', 'Floor_',
#       'communityName_pred_skew', 'month_skew', 'tradeDay_pred_unique',
#       'tradeNewMeanPrice', 'day_std', 'communityName_pred_kurt',
#       'plate_pred_std'][:15]
Cross_train = get_Cross_feat(imp_feature,train)
Cross_test = get_Cross_feat(imp_feature,test)


train_after_cross = train.join(Cross_train)
test_after_cross = test.join(Cross_test)



# =============================================================================
# 特征交叉后 重新训练  0.90921
# =============================================================================

score, feature_importance_df,predictions_lgb, oog_lgb = self_define_function.lgb_eval(params, train_after_cross, target, test_after_cross)
feature_importance = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False).index)
#feature_importance = self_define_function.feature_importance(feature_importance_df)



pd.DataFrame(predictions_lgb).apply(round).to_csv('submit/submit.csv',na_rep='\\n',index=False,encoding='utf8',header=False)

# =============================================================================
# 保存所有特征 给下一步做特征选择
# =============================================================================
train_after_cross['tradeMoney'] = target
combine7 = pd.concat([train_after_cross,Cross_test],axis=0, ignore_index=True)

combine7.to_csv('output/test-a/data_featureCross.csv',index=None)