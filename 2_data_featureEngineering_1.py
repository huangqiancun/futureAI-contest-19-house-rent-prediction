# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:13:37 2019

@author: huangqiancun
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

combine = pd.read_csv('output/test-a/data_preprocessing.csv')

# =============================================================================
# 添加简单特征
# =============================================================================
def simple_feature(df):
    # 房子年龄
    df['buildingAge'] = df['tradeYear'] - df['buildYear']
    # 实际交易时间
    df['tradeDay_real'] = (df['tradeMonth'] -1) * 30 + df['tradeDay']

    # 房间总数
    df['totalRoom'] = df['room'] + df['lobby'] + df['bath']
    # 平均面积
    df['per_room'] = df['area'] / df['totalRoom']
    
    df['room_add_lobby'] = df['room'] + df['lobby']
    df['room_sub_lobby'] = df['room'] - df['lobby']
    
    df['room_add_bath'] = df['room'] + df['bath']
    df['room_sub_bath'] = df['room'] - df['bath']
    
    df['lobby_add_bath'] = df['lobby'] + df['bath']
    df['lobby_sub_bath'] = df['lobby'] - df['bath']
    # pv/uv
    #     pv/uv
    df['pv/uv'] = df['pv'] / (df['uv']+1)
    df['client'] = df['pv/uv'] + df['lookNum']
    
    df['Floor_'] =  np.ceil(df['houseFloor'].map({'low':0.2, 'middle':0.5, 'high':0.8 }) * df['totalFloor'])
    # 配套设施
    df['transportation'] = df['subwayStationNum'] + df['busStationNum']
    df['education'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
    df['health'] = df['hospitalNum'] + df['drugStoreNum'] + df['drugStoreNum'] + df['gymNum']
    df['convenience'] = df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']
    
    df['supportingFacilities'] = df['transportation'] + df['education'] + df['health'] + df['convenience']
    
    # 成交比例
#     新房成交比例
    df['tradeNewRatio'] = df['tradeNewNum'] / (df['tradeNewNum'] + df['remainNewNum'] + 1)
#     新增人口平均每人成交套数
    df['tradeRatio_avg'] = df['newWorkers'] / (df['tradeSecNum'] + df['tradeNewNum']+1)
#     工人占比
    df['workerRatio'] = df['totalWorkers'] / (df['residentPopulation'] + df['totalWorkers'] + 1)
#    本月新增工人占比
    df['NewWorkerRatio'] = df['newWorkers'] / (df['totalWorkers']+1)
    return df

combine_feature_1 = simple_feature(combine)

combine_feature_1.to_csv('output/test-a/data_featureEngineering_1.csv',index=None)