# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:11:08 2019

@author: huangqiancun
"""
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import MeanEncoder

combine = pd.read_csv('output/test-a/data_featureEngineering_1.csv')


# =============================================================================
# 获取位置 区域+板块+小区
# =============================================================================
def get_location(df):
    df['location'] = df['region']+"_"+df['plate']+"_"+df['communityName']
    return df
combine2 = get_location(combine)

# =============================================================================
# rank编码
# =============================================================================
def rank_feature(df):
    df['communityName_freq'] = df['communityName'].map(df['communityName'].value_counts().rank()/len(df['communityName'].unique()))
    df['region'] = df['region'].map(df['region'].value_counts().rank()/len(df['region'].unique()))
    df['plate'] = df['plate'].map(df['plate'].value_counts().rank()/len(df['plate'].unique()))
    df['location'] = df['location'].map(df['location'].value_counts().rank()/len(df['location'].unique()))
    return df

combine3 = rank_feature(combine2)
# =============================================================================
# LabelEncoder 编码
# =============================================================================
def label_feature(df):
    for col in ['houseType',"houseToward","communityName"]:
        lbl =LabelEncoder()
        df[col]=lbl.fit_transform(df[col])
    return df

combine4 = label_feature(combine3)


train = combine4[combine4.tradeMoney.notnull()]
test = combine4[combine4.tradeMoney.isnull()]
# =============================================================================
# MeanEncoder 需要自定义依赖文件 MeanEncoder.py
# =============================================================================
flag="mean"
print("mean...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("std...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="std",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)

print("max...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="max",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("min...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="min",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("skew...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="skew",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("kurt...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="kurt",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("unique...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="unique",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)
print("freq...")
mean_encoder = MeanEncoder.MeanEncoder(
                        categorical_features=['communityName',"plate","buildYear","totalFloor","tradeDay"],
                stats="freq",
                target_type='regression',flag=flag
                )

train = mean_encoder.fit_transform(train, pd.Series(train['tradeMoney']))
test = mean_encoder.transform(test)


# =============================================================================
# 组合变量
# =============================================================================
combine5 = pd.concat([train,test])
groupby_feat=combine5.groupby("communityName",as_index=False)['area'].agg({"area_mean":"mean","area_std":"std","area_skew":"skew",
                                                         "area_min":"min","area_max":"max",})

groupby_feat1=combine5.groupby("communityName",as_index=False)['tradeDay'].agg({"day_mean":"mean","day_std":"std","day_skew":"skew",
                                                         "day_min":"min","day_max":"max",})

groupby_feat2=combine5.groupby("communityName",as_index=False)['tradeMonth'].agg({"month_mean":"mean","month_std":"std","month_skew":"skew",
                                                         "month_min":"min","month_max":"max"})

train = pd.merge(train,groupby_feat,on="communityName",how="left")
test = pd.merge(test,groupby_feat,on="communityName",how="left")

train = pd.merge(train,groupby_feat1,on="communityName",how="left")
test = pd.merge(test,groupby_feat1,on="communityName",how="left")

train = pd.merge(train,groupby_feat2,on="communityName",how="left")
test = pd.merge(test,groupby_feat2,on="communityName",how="left")

# =============================================================================
# 输出
# =============================================================================
combine6 = pd.concat([train,test])

combine6.drop(['ID', 'city','tradeYear'], axis = 1, inplace = True)

combine6.to_csv('output/test-a/data_featureEngineering_2.csv',index=None)
