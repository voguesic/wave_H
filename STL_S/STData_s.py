# !/user/bin/env python3
# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from vmdpy import VMD

from adjMatrix import getADJ
from functions import loadData, scale
from torch.utils.data import DataLoader
from MyDataSet import MyDataset
from statsmodels.tsa.seasonal import STL
class CSVDataModule():
    def __init__(self,path,
                 batch_size=32,
                 seq_len=12,
                 pred_len=12,
                 split_ratio=0.7,
                 station=['0','1','2','3'],
                 normalize=False,
                 normalForm='all',
                 vars='swh',auxiVar=None,base='default',k=2):
        super(CSVDataModule,self).__init__()
        self.path=path
        self.batch_size=batch_size
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.split_ratio=split_ratio
        self.normalize=normalize
        self.vars=vars
        self.base=base
        self.auxiVar=auxiVar
        self.station=station
        self.k=k
        self.normalForm=normalForm
        self.feature,self.auxiData=loadData(path,station,vars,auxiVar)
        self.split_size = int(len(self.feature) * self.split_ratio)

    def adfBasedPearson(self,data):
        A = np.eye(len(data[0]))
        for i in range(len(data[0])):
            for j in range(i + 1, len(data[0])):
                # 计算皮尔逊相关系数及 p 值
                pearson_coefficient, p_value = pearsonr(data[:, i], data[:, j])
                A[i, j] = A[j, i] = pearson_coefficient if pearson_coefficient>=0 else 0
        return A
    def composeData(self,data,auxData):
        data_X=[]
        data_Y=[]

        auxiArray = []
        for i in range(data.shape[-2]-self.seq_len-self.pred_len):
            data_X.append(data[:,i:i+self.seq_len,:])

            data_Y.append(auxData[i+self.seq_len:i+self.seq_len+self.pred_len])

        return np.array(data_X),np.array(data_Y)
    def adjacent(self):
        adj=np.full((len(self.station),len(self.station)),0.5)
        np.fill_diagonal(adj, 1)
        return adj

    def vmdFJ(self,signal,date):
        res = []
        for i in range(4):
            ts = pd.Series(signal[:,i], index=date[:,0])

            stl = STL(ts, period=24, seasonal=11)
            result = stl.fit()
            res.append(np.hstack((result.trend.values.reshape(-1,1),
                                  result.seasonal.values.reshape(-1,1),
                                  result.resid.values.reshape(-1,1))))
        return np.stack(res,axis=0)
        # # VMD分解参数设置
        # alpha = 1000  # 数据保真度约束参数
        # tau = 0.  # 噪声容限
        # K = 2  # 分解的模态数
        # DC = 0  # 不包含直流分量
        # init = 0.5  # 初始化中心频率
        # tol = 1e-7  # 收敛容限
        # res=[]
        # for i in range(4):
        #     # 进行VMD分解
        #     u, u_hat, omega = VMD(signal[:,i], alpha, tau, self.k, DC, init, tol)
        #     res.append(u)
        # return np.stack(res,axis=0)

    def getTrainDataLoader(self):

        scalerTest=None
        train_data = self.vmdFJ(self.feature[:8400*2],self.auxiData[:8400*2])
        val_data= self.vmdFJ(self.feature[8400 * 2:8400 * 3],self.auxiData[8400 * 2:8400 * 3])
        test_data = self.vmdFJ(self.feature[8400*3:-1],self.auxiData[8400*3:-1])
        if self.normalize:
            test_data, scalerTest = scale(test_data, self.normalForm)
            train_data,scaler=scale(train_data,self.normalForm)
        #邻接矩阵
        coors = [(34.84,123.345),(35.94,125.37),(33.74,121.12),(33.65,125.42)]
        # allADJ,allLPLS = getADJ(self.auxiData, coors,self.base)
        # train_ADJ=allADJ[:self.split_size]
        # test_ADJ=allADJ[self.split_size:]
        #数据

        train_X, train_Y = self.composeData(train_data, self.feature[:8400 * 2])
        val_X, val_Y = self.composeData(val_data, self.feature[8400 * 2:8400 * 3])
        test_X, test_Y = self.composeData(test_data, self.feature[8400 * 3:-1])
        adj=self.adjacent()
        train_dataset = MyDataset(train_X, train_Y)
        val_dataset = MyDataset(val_X, val_Y)
        test_dataset = MyDataset(test_X, test_Y)
        train_dataLoader=DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataLoader=DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataLoader,val_dataLoader,test_dataLoader,test_X,test_Y,scalerTest,test_data



