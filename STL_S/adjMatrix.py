import math

import numpy as np
import torch


#(34.84,123.345),(35.94,125.37),(33.74,121.12),(33.65,125.42)
def positionInfo(node_num,coor):
    if node_num!=len(coor):
        print("please input right coor!")
        return
    infoArr=np.zeros((node_num,node_num))
    for i in range(node_num):
        for j in range(i+1,node_num):
            # 将角度转换为弧度
            lat1_rad = math.radians(coor[i][0])
            lon1_rad = math.radians(coor[i][1])
            lat2_rad = math.radians(coor[j][0])
            lon2_rad = math.radians(coor[j][1])

            # 计算经度差
            delta_lon = lon2_rad - lon1_rad

            # 计算方向角（方位角）
            angle = math.atan2(
                math.sin(delta_lon) * math.cos(lat2_rad),
                math.cos(lat1_rad) * math.sin(lat2_rad) -
                math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
            )

            # 将角度转换为度
            angle_degrees = math.degrees(angle)

            # 将角度标准化为 0 到 360 度
            angle_degrees = (angle_degrees + 360) % 360


            infoArr[i,j]=angle_degrees
            infoArr[j,i]=(180+angle_degrees)%360
    return infoArr



'''
return 风速，风向
风向的方向是相对于北方
'''
def windComponent(u,v):
    # 计算风速
    wind_speed = np.sqrt(u**2 + v**2)

    # 计算风向
    wind_direction = np.arctan2(v, u)  # 使用 arctan2 考虑象限
    wind_direction_degrees = (np.degrees(wind_direction) + 360) % 360
    return wind_speed,wind_direction_degrees

def distanceA(coors):
    A=np.eye(len(coors))
    for i in range(len(coors)):
        for j in range(i+1,len(coors)):
            lat1,lon1=coors[i]
            lat2,lon2=coors[j]

            d=math.pow(math.pow(lat1-lat2,2)+math.pow(lon1-lon2,2),0.5)
            # d=geodesic(coors[i], coors[j]).kilometers
            A[i,j]=A[j,i]=1/d

    return A


def getADJ(SPdata,coor,base='default'):
    PInfo = positionInfo(len(coor), coor)
    referA=[]
    if base == 'distance':
        referA=distanceA(coor)

    adj_all = []
    lpls_all = []
    for s in range(len(SPdata)):
        adj=adjAtT(SPdata[s], PInfo)
        if base == 'distance':
            adj=adj*referA
        adj_all.append(adj)
        # lpls_all.append(calculate_laplacian_with_self_loop(adj))
        # weight_all.append(b)
    return np.array(adj_all),np.array(lpls_all)

def adjAtT(sd,positionInfo):
    I=np.eye(len(sd))
    adj=np.zeros((len(sd),len(sd)))
    for i in range(len(sd)):
        for j in range(i+1,len(sd)):
            a=sd[i]
            # # a=(sd[i]+360)%360
            # # if sd[i,0]-sd[i,0]
            # rangeRadius=[(a-60)%360,(a+60)%360]
            # # idxs=[]
            # if rangeRadius[0]<rangeRadius[1]:
            #     idxs = np.where((positionInfo[i] > rangeRadius[0]) & (positionInfo[i] < rangeRadius[1]))
            # else:
            #     idxs = np.where((positionInfo[i] > rangeRadius[0]) | (positionInfo[i] < rangeRadius[1]))
            # # print(len(idxs))
            # for id in idxs[0]:
            #     if id==i:
            #         continue
            #     adj[i, id] = 1
            #     adj[id, i] = 1
            # a = (sd[i] + 180) % 360
            # if sd[i,0]-sd[i,0]
            rangeRadius = [(a - 70) % 360, (a + 70) % 360]
            # idxs=[]
            if rangeRadius[0] < rangeRadius[1]:
                idxs = np.where((positionInfo[i] > rangeRadius[0]) & (positionInfo[i] < rangeRadius[1]))
            else:
                idxs = np.where((positionInfo[i] > rangeRadius[0]) | (positionInfo[i] < rangeRadius[1]))
            # print(len(idxs))
            for id in idxs[0]:
                if id == i:
                    continue
                # if np.any(swh[:, i] >= threshhold) or np.any(swh[:, id] >= threshhold):
                adj[i, id] = 1
                adj[id, i] = 1
    # adj=adj+I
    # edge_index = np.array(np.nonzero(adj))
    # edge_weight = adj[edge_index[0], edge_index[1]]
    return adj
def calculate_laplacian_with_self_loop(matrix):
    matrix=torch.tensor(matrix)
    matrix = matrix + torch.eye(matrix.size(0))
    # 计算节点的度
    row_sum = matrix.sum(1)
    #度矩阵的逆平方根
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian.numpy()