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

import self_define_function

#combine = pd.read_csv('output/test-a/data_featureSelection.csv')
#train = combine[combine.tradeMoney.notnull()]
#test = combine[combine.tradeMoney.isnull()]

# 特征选择太耗时，没跑完，直接使用前100个重要的特征训练模型
combine = pd.read_csv('output/test-a/data_featureCross.csv')

selected_features = feature_importance[:100]

train = combine[combine.tradeMoney.notnull()]
test = combine[combine.tradeMoney.isnull()]
train = train[selected_features]
test = test[selected_features]

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


#  特征选择后训练  0.90848  
target = train.pop('tradeMoney')
test.drop(['tradeMoney'], axis = 1, inplace = True)

object_feats = [x for x in train.select_dtypes(include = ['object']).columns]
for col in object_feats:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    
score, feature_importance_df,predictions_lgb, oog_lgb = self_define_function.lgb_eval(params, train, target, test)

#feature_importance = self_define_function.feature_importance(feature_importance_df)

pd.DataFrame(predictions_lgb).apply(round).to_csv('submit/submit-after_feature_selection.csv',na_rep='\\n',index=False,encoding='utf8',header=False)

