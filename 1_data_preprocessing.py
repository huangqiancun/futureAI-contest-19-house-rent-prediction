# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:19:58 2019

@author: huangqiancun
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

train_raw = pd.read_csv('dataset/train_data.csv')
test_raw = pd.read_csv('dataset/test_a.csv')

combine_raw = pd.concat([train_raw, test_raw],axis=0,ignore_index=True)

# =============================================================================
# 将所有中文转化为英文
# =============================================================================
def ZHtoEN(df):
    df['rentType'][df['rentType']=='--'] = '未知方式'
    df['rentType'].replace({'未知方式':'unknown', '合租': 'hezu', '整租':'zhengzu'}, inplace = True)
    df['houseFloor'].replace({'低':'low', '中': 'middle', '高':'high'}, inplace = True)
    df['houseFloor'].replace({'低':'low', '中': 'middle', '高':'high'}, inplace = True)
    df['houseToward'].replace(['暂无数据', '西南', '西北', '西', '南北', '南', '东西', '东南', '东', '北'],
                              ['unknown', 'WS', 'WN', 'W', 'SN', 'S','EW','ES','E','N'] , inplace = True)
    df['houseDecoration'].replace([u'其他','毛坯',u'简装',u'精装'],
                                  ['unknown', 'maopei', 'jianzhuang', 'jingzhuang'],inplace=True)
    df['buildYear'][df['buildYear'] == '暂无信息'] = 'unknown'
    return df

combine_1 = ZHtoEN(combine_raw)

# =============================================================================
# 将houseType转化为室厅卫
# =============================================================================
def houseType_trans(df):
    houseType_ = pd.DataFrame(columns = ['ID', 'room', 'lobby', 'bath'])
    houseType_['ID'] = df['ID'].copy()
    houseType_['room'] = df['houseType'].apply(lambda x: int(x[0]))
    houseType_['lobby'] = df['houseType'].apply(lambda x: int(x[2]))
    houseType_['bath'] = df['houseType'].apply(lambda x: int(x[4]))
    df = df.merge(houseType_,on='ID',how='left')
    return df

combine_2 = houseType_trans(combine_1)

# =============================================================================
# 将tradeTime转化为年月日 和 时间戳
# =============================================================================
def datetime_timestamp(dt):
    s = time.mktime(time.strptime(dt,'%Y/%m/%d'))
    return s

def tradeTime_trans(df):
    df['tradeYear'] = df['tradeTime'].apply(lambda x: int(x.split('/')[0]))
    df['tradeMonth'] = df['tradeTime'].apply(lambda x: int(x.split('/')[1]))
    df['tradeDay'] = df['tradeTime'].apply(lambda x: int(x.split('/')[2]))

    df['tradeTime']=df['tradeTime'].apply(lambda x:datetime_timestamp(x),1)
    return df 

combine_3 = tradeTime_trans(combine_2)

# =============================================================================
# 处理缺失值
# =============================================================================
def fillna_(df):
   # 处理pv和uv的空值
    df['pv'] = df['pv'].fillna(df['pv'].mean())
    df['uv'] = df['uv'].fillna(df['uv'].mean())
    df['pv'] = df['pv'].astype('int')
    df['uv'] = df['uv'].astype('int')
    # 处理buildYear的暂无信息
    df['buildYear'].replace('unknown', np.nan, inplace = True)
    df['buildYear'].fillna(df['buildYear'].mode()[0], inplace = True)
    df['buildYear'] = df['buildYear'].astype('int')
    # 将未知 houseToward 转为为 S
    temp = df['houseToward'].mode()[0]
    df['houseToward'] = df['houseToward'].apply(lambda x:x.replace("unknown",temp),1)
    return df
    
combine_4 = fillna_(combine_3)

# =============================================================================
# 去除test中不存在的小区，区域和板块
# =============================================================================
def remove_(df, train, test):
    samecommunity = set(train.communityName) & set(test.communityName)
    df = df[df.communityName.isin(samecommunity)]

    sameregion = set(train.region) & set(test.region)
    df = df[df.region.isin(sameregion)]

    sameplate = set(train.plate) & set(test.plate)
    df = df[df.plate.isin(sameplate)]
    return df

combine_5 = remove_(combine_4, train_raw, test_raw)

# =============================================================================
# 去除训练集中的异常值
# =============================================================================
train_pre = combine_5[combine_5.tradeMoney.notnull()]
test_pre= combine_5[combine_5.tradeMoney.isnull()]

train_pre1 = train_pre[(train_pre.area<=160)&(train_pre.area>=10)]
train_pre2 = train_pre1[(train_pre1.tradeMoney<=20000)&(train_pre1.tradeMoney>=600)]
train_pre3 = train_pre2[(train_pre2.totalFloor > 1)]

combine_6 = pd.concat([train_pre3, test_pre],axis=0,ignore_index=True)



#combine_6.to_csv('output/test-a/data_preprocessing.csv',index=None)