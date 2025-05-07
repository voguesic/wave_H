import math
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def delete_all_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return
    folder = Path(folder_path)
    # 遍历文件夹中的所有文件
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()  # 删除文件
            print(f"Deleted file: {file}")
def colsName(station,k):
    df_sub=[]
    for s in range(station):
        df_sub.append('pred_s'+str(s+1))
        df_sub.append('true_s'+str(s+1))

        for vk in range(k):
            df_sub.append('sub_vmd_pred_k'+str(vk+1)+'_s'+str(s+1))
            df_sub.append('sub_vmd_true_k'+str(vk+1)+'_s'+str(s+1))
    return np.array(df_sub)
def loadData(path,stations,vars,auxiVar):
    primaryData=[]
    auxiData=[]
    for i in stations:
        data=pd.read_csv('../dataSet/station_'+i+'_data.csv')
        data=data.iloc[:,:]
        primaryData.append(np.reshape(data[vars].values.astype('float32'),(-1,1)))
        # u10=data['u10'].values.astype('float32')
        # v10=data['v10'].values.astype('float32')
        # wind_direction_rad = np.arctan2(u10,v10)
        #
        # # 转换为度数，并确保结果在 [0, 360) 范围内
        # wind_direction_deg = np.degrees(wind_direction_rad)  # 转换为度
        # wind_direction_deg = (wind_direction_deg + 360) % 360
        mwd=np.reshape(data['time'].values, (-1, 1))
        # wind_direction_deg=np.reshape(wind_direction_deg, (-1, 1))
        auxiData.append(mwd)
    return np.hstack(primaryData),np.hstack(auxiData)

def scale(feature,normalForm='all'):
    if normalForm=='all':
        # 将数据展平为一维数组
        data_flat = feature.flatten().reshape(-1, 1)

        # 使用 StandardScaler 对整个数据进行标准化
        scaler = MinMaxScaler()
        standardized_flat = scaler.fit_transform(data_flat)

        # 将标准化后的数据恢复为原始形状
        standardized_data = standardized_flat.reshape(feature.shape)
        return standardized_data,scaler
    else:
        scaler = MinMaxScaler()
        standardized_feature = scaler.fit_transform(feature)
        return standardized_feature,scaler
def inverse_scale(data,scaler):
    if scaler.n_features_in_>1:

        standardized_data=scaler.inverse_transform(data)
        return standardized_data
    else:
        data_flat = data.flatten().reshape(-1, 1)
        standardized_flat = scaler.inverse_transform(data_flat)

        # 将标准化后的数据恢复为原始形状
        resultData = standardized_flat.reshape(data.shape)
        return resultData
def afterResult(Y,pred,scaler):
    predData=inverse_scale(pred,scaler)
    GTData=inverse_scale(Y,scaler)

    return predData,GTData