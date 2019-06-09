# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 04:24:08 2019

@author: huangqiancun
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
 

combine = pd.read_csv('output/test-a/data_featureCross.csv')

train= combine[combine.tradeMoney.notnull()]
test= combine[combine.tradeMoney.isnull()]



def lgb_feature(train, target):
    
    folds = KFold(n_splits=3, shuffle=True, random_state=2333)

    oof_lgb = np.zeros(len(train))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=3000, early_stopping_rounds = 200)

        oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    print("CV Score: {:<8.5f}".format(r2_score(target, oof_lgb)))
    score = 1- np.sum(np.square(target - oof_lgb)) / np.sum(np.square(target - np.mean(target)))
    print("Score: {:<8.5f}".format(score))
    
    return score

# 参考 https://stackoverflow.com/questions/43258188/cant-reproduce-xgb-cv-cross-validation-results
def Forward_Feature_Selection(train, target):

    best = 0
    selected_features = []
    
    previous_best_score = 0
    
    res_score = []
    threshold = 0
    gain = threshold + 1
    features = train.columns.values
    selected = np.zeros(len(features))
    scores = np.zeros(len(features))
    
    while (gain > threshold):    # we start a add-a-feature loop

#     while len(selected_features) < 3:
        
        for i in range(0,len(features)):
            print("last added feature: " + features[best] + "**number:" + str(len(selected_features)) + "**Launching XGBoost for feature:= "+ features[i])
            if (selected[i] == 0):   # take only features not yet selected
                selected_features.append(features[i])
                new_train = train.iloc[:][selected_features]               
                selected_features.remove(features[i])
                
                categorical_subset = new_train.select_dtypes('category')
                categorical_feats = [c for c in categorical_subset.columns]
                
                scores[i] = lgb_feature(new_train, target)
                
            else:
                scores[i] = -1    # discard already selected variables from candidates
                
        # 一次只取最好的一个
#        best = np.argmax(scores)
#        gain = scores[best] - previous_best_score
#        
#        if (gain > 0):
#            previous_best_score = scores[best]
#            res_score.append(scores[best])
#            selected_features.append(features[best])
#            selected[best] = 1
        
        # 一次选择多个特征，加速
        candidates = scores[sorted(scores, reverse = True)[:5]]
        best = candidates[0]
        gain = scores[best] - previous_best_score
        
        if (gain > 0):
            previous_best_score = scores[best]
            res_score.append(scores[candidates])
            selected_features.append(features[candidates])
            selected[candidates] = 1
        
        print("Adding feature: " + features[best] + " increases score by " + str(gain) + ". Final score is now: " + str(previous_best_score)) 
    
    return selected_features, previous_best_score, res_score

target = train.pop('tradeMoney')
test.drop(['tradeMoney'], axis = 1, inplace = True)

object_feats = [x for x in train.select_dtypes(include = ['object']).columns]
for col in object_feats:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    
selected_features, previous_best_score, res_score = Forward_Feature_Selection(train, target)


train = train[selected_features]
test = test[selected_features]

train['tradeMoney'] = target
combine = pd.concat([train,test],axis=0, ignore_index=True)

combine.to_csv('output/test-a/data_featureSelection.csv',index=None)