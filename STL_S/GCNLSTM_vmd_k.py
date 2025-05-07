import os.path
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from torch import optim
from STData_s import CSVDataModule
from functions import afterResult, delete_all_files_in_folder, colsName
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
def symmetric_normalize(matrix):
    # 计算节点的度
    row_sum = matrix.sum(1)
    # 度矩阵的逆平方根
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # 对称归一化
    normalized_matrix = d_mat_inv_sqrt @ matrix @ d_mat_inv_sqrt
    return normalized_matrix
class LearnableAdjacencyMatrix(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacencyMatrix, self).__init__()
        # 将邻接矩阵初始化为全 1 并定义为可学习参数
        self.adj = nn.Parameter(torch.ones((1, num_nodes, num_nodes)))

    def forward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eye = torch.eye(self.adj.shape[1]).unsqueeze(0).to(device)  # 创建一个对角线为1的矩阵
        # self.adj.data = self.adj.data * (1 - eye) + eye
        adj=(self.adj.data+self.adj.data.transpose(1,2))/2
        self.adj.data = adj * (1 - eye) + eye
        return self.adj
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 定义可学习的权重矩阵
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x, adj):
        # 计算 A_hat * X * W
        # support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, x)
        return out
class GCNcustom(nn.Module):
    def __init__(self,num_nodes,node_num):
        super(GCNcustom,self).__init__()
        self.num_nodes=num_nodes
        self.project=[nn.Linear(1, 16) for i in range(node_num)]
    def forward(self,x,adj):
        H=adj@x
        out=[]
        for i in range(4):
            out.append(F.relu(self.project[i](H[:,i:i+1,:].transpose(1,2))))
        return torch.stack(out,dim=-1)
class GCNTrans(nn.Module):
    def __init__(self,num_nodes, node_features, hidden_dim,lstm_layers,
                 output_dim,initial_edge_weight,if_attention, hidden_dim2=4,k=3):
        super(GCNTrans, self).__init__()
        self.num_nodes=num_nodes
        self.pred_len=output_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj=symmetric_normalize(initial_edge_weight).to(device)
        # GCN layer
        self.gcn = nn.ModuleList([GCNcustom(node_features,4) for i in range(k-1)])
        self.gcn_fc=nn.ModuleList([nn.Linear(hidden_dim,node_features) for i in range(k-1)])
        self.project1=nn.ModuleList([nn.Linear(1,16) for i in range(4)])
        self.project2=nn.ModuleList([nn.Linear(1,16) for i in range(4)])

        # lstm layer
        self.lstm_1=nn.ModuleList([nn.LSTM(32, hidden_dim2, lstm_layers, batch_first=True) for i in range(4)])
        self.lstm_2 = nn.ModuleList([nn.LSTM(1, hidden_dim2, lstm_layers, batch_first=True) for i in range(num_nodes)])
        self.lstm_3 = nn.ModuleList([nn.LSTM(1, hidden_dim2, lstm_layers, batch_first=True) for i in range(num_nodes)])

        # Fully connected layer
        self.fc_1=nn.ModuleList([nn.Linear(hidden_dim2, self.pred_len) for i in range(4)])
        self.fc_2 = nn.ModuleList([nn.Linear(hidden_dim2, self.pred_len) for i in range(num_nodes*2)])
        self.dropout = nn.Dropout(p=0.3)
        self.if_attention=if_attention
        if if_attention:
            # Attention mechanism
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)

    # self.adj:没有自连接的邻接矩阵
    # x:(B,F,N)
    def forward(self, x):
        batch_size = x.size(0)

        # 第一部分
        out=[]
        out_2 = torch.empty((batch_size, self.pred_len, 0)).to(device)
        x_1 = x[:, :, :, 0]
        H = self.adj @ x_1
        out1 = []
        for i in range(4):
            out1.append(F.relu(self.project1[i](H[:, i:i + 1, :].transpose(1, 2))))
        out1=torch.stack(out1, dim=-1)
        # out1 = self.gcn[0](x_1, self.adj)  # (B,F,N)=>(B,N,H)
        # gcn_out = self.gcn_fc[0](gcn_out)  # (B,N,H)=>(B,N,F)

        # (B,N,F)=>(B,F,N)
        # gcn_out = gcn_out.transpose(1, 2)

        for k in range(4):  # range(x.shape[-1]-1):
            # out1 = self.project1[k](gcn_out[:,:,k:k+1])
            out2 = self.project2[k](x_1.transpose(1, 2)[:,:,k:k+1])
            # if k==1:
            #     x_1 = x[:, :, :, k+1]
            X_input=torch.cat((out1[:,:,:,i],out2),dim=-1)
            # features = torch.stack([x_1.transpose(1, 2), gcn_out], dim=0)
            # pooled_feature = torch.mean(features, dim=0)
            # X_input=torch.cat([x_1.transpose(1,2)[:,:,k:k+1],gcn_out[:,:,k:k+1]],dim=-1)
            lstm_out, (hn, _) = self.lstm_1[k](X_input)
            fc_out = self.fc_1[k](hn[0]).reshape(batch_size, self.pred_len, -1)
            out_2 = torch.cat((out_2, fc_out), dim=2)
        out.append(out_2)


        # 第二部分
        x_2=x[:,:,:,1]
        out_2 = torch.empty((batch_size, self.pred_len, 0)).to(device)
        for i in range(self.num_nodes):
            out_N, (hn, _) = self.lstm_2[i](x_2[:,i:i+1].transpose(1,2))
            out_N = self.fc_2[i+4](hn[0])
            # (B,pred_len*N)
            out_N = out_N.view(batch_size, -1, self.pred_len).transpose(1,2)
            out_2 = torch.cat((out_2, out_N), dim=2)
        out.append(out_2)

        # x_2 = x[:, :, :, 2]
        # out_2 = torch.empty((batch_size, self.pred_len, 0)).to(device)
        # for i in range(self.num_nodes):
        #     out_N, (hn, _) = self.lstm_2[i + 8](x_2[:, i:i + 1].transpose(1, 2))
        #     out_N = self.fc_2[i + 8](hn[0])
        #     # (B,pred_len*N)
        #     out_N = out_N.view(batch_size, -1, self.pred_len).transpose(1, 2)
        #     out_2 = torch.cat((out_2, out_N), dim=2)
        # out.append(out_2)
        x_2 = x[:, :, :, 2]
        out_2 = torch.empty((batch_size, self.pred_len, 0)).to(device)
        # gcn_out = F.relu(self.gcn[1](x_1, self.adj))  # (B,F,N)=>(B,N,H)
        # gcn_out = self.gcn_fc[1](gcn_out)  # (B,N,H)=>(B,N,F)
        for i in range(self.num_nodes):
            out_N, (hn, _) = self.lstm_3[i](x_2[:, i:i + 1].transpose(1, 2))
            out_N = self.fc_2[i](hn[0])
            # (B,pred_len*N)
            out_N = out_N.view(batch_size, -1, self.pred_len).transpose(1, 2)
            out_2 = torch.cat((out_2, out_N), dim=2)
        out.append(out_2)

        return out


def customloss(y_true,y_pred,alpha=2):
    loss = (y_true.flatten() - y_pred.flatten()) ** 2
    loss = torch.where(y_pred.flatten() < y_true.flatten(), alpha * loss, loss)
    return loss.mean()
    # loss=0.0
    # r=y_true.flatten()
    # s=y_pred.flatten()
    # for i in range(len(r)):
    #     if s[i]<0:
    #         loss=loss+alpha*(s[i]-r[i])**2
    #     else:
    #         loss=loss+(s[i]-r[i])**2
    # return loss



'''
    固定A、合一起的LSTM、未归一
    四个站点归一化
    GCN的邻接矩阵可学习
    四个站点同时进入LSTM
'''
def distanceA(coors):
    A=np.eye(len(coors))
    for i in range(len(coors)):
        A[i, i]=0
        for j in range(i+1,len(coors)):
            lat1,lon1=coors[i]
            lat2,lon2=coors[j]

            d=math.pow(math.pow(lat1-lat2,2)+math.pow(lon1-lon2,2),0.5)
            # d=geodesic(coors[i], coors[j]).kilometers
            A[i,j]=A[j,i]=1/d

    return A

def inverse_scale(data,scaler,k):
    return_data=[]
    for i in range(4):
        station_data=[]
        for kv in range(k):
            station_data.append(data[kv,:,:,i]*(scaler[i].data_max_[kv]-scaler[kv].data_min_[kv])+scaler[kv].data_min_[kv])
        return_data.append(torch.stack(station_data,dim=0))
    return torch.stack(return_data,dim=-1)
'''
    不归一
'''
savePath=r'E:\LDRes\GCNLSTM\R6'
# delete_all_files_in_folder(savePath)
for d  in [24]:

    for k in [3]:
        r_max = 0
        pred_len =d if d>0 else 1
        seq_len =pred_len*4
        stepByStep=False
        if_one=False

        hidden_dim = 64
        hidden_dim2 = 32
        lstm_layers = 1
        epochTotal = 500
        patience = 30
        batch_size = 256

        AuxiliaryVar=[]
        tarVar='swh'
        coors=[(34.84,123.345),(35.94,125.37),(33.74,121.12),(33.65,125.42)]
        station=['1','2','3','4']
        stations=len(station)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataLoader,val_dataLoader,test_dataLoader,test_X,test_Y,scale,test_data=CSVDataModule('../dataSet/',batch_size,seq_len,pred_len,0.8,
                                                                                          station,normalForm='single',auxiVar='mwd',base='distance',k=k).getTrainDataLoader()
        adj=np.full((len(station),len(station)),0.5)
        np.fill_diagonal(adj, 1)
        # ----------------初始化model------------------
        A = distanceA(coors)
        for e in range(2):
            model = GCNTrans(stations,seq_len, hidden_dim, lstm_layers, 1 if stepByStep else pred_len,torch.Tensor(A),False,
                             hidden_dim2=hidden_dim2,k=3)
            model = model.to(device)

            best_val_loss = float('inf')
            patience_counter = 20
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # criterion = torch.nn.L1Loss().to(device)
            criterion = nn.MSELoss()  # 例如用于回归任务的损失函数
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

            train_l_sum = 0
            batch_count = 0
            start = time.time()
            train_loss = []
            for epoch in range(500):
                loss_batch=0.0
                model.train()
                for batch in train_dataLoader:
                    optimizer.zero_grad()
                    node_features,  labels = batch
                    # (B,seq_len,N)
                    node_features = node_features.to(device)
                    # (B, pred, N)
                    if stepByStep:
                        labels=labels[:,:1,:]
                    labels = labels.to(device).to(torch.float32)

                    y_hat = model(node_features)
                    y_hat = torch.stack(y_hat)
                    # y_hat=inverse_scale(y_hat,train_scale,k)
                    # 沿着第 0 维求和
                    y_hat = torch.sum(y_hat, dim=0)
                    loss=customloss(labels, y_hat)
                    # loss = criterion(labels, y_hat)
                    loss.backward()
                    optimizer.step()
                    loss_num=loss.cpu().item()
                    loss_batch+=loss_num
                    train_l_sum += loss_num
                    batch_count += 1
                if (epoch + 1) % 20 == 0 and epoch > 0:
                    print('epoch %d, loss %.4f, time %.1f sec'
                    % (epoch + 1,  train_l_sum / 20, time.time() - start))
                    start = time.time()
                    train_l_sum = 0
                    batch_count = 0
                train_loss.append(loss_batch)

                # 验证阶段
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_data in val_dataLoader:
                        node_features,  labels = val_data
                        node_features = node_features.to(torch.float32)
                        node_features = node_features.to(device)
                        if stepByStep:
                            labels = labels[:, :1, :]
                        labels = labels.to(device).to(torch.float32)

                        y_hat = model(node_features)
                        y_hat = torch.stack(y_hat)
                        # y_hat = inverse_scale(y_hat, val_scale, k)
                        # 沿着第 0 维求和
                        y_hat = torch.sum(y_hat, dim=0)
                        val_loss += customloss(labels, y_hat)
                        # val_loss += criterion(labels, y_hat)
                avg_val_loss = val_loss / len(val_dataLoader)

                scheduler.step(avg_val_loss)
                # Early Stopping逻辑
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 可以选择保存最佳模型的权重
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered", str(epoch))
                        print("val_loss:%.4f ; train_loss:%.4f" % (avg_val_loss, train_l_sum / (epoch % 20+1)))
                        break


            # -------------------预测------------------------
            print('---------------pred-------------')
            model.eval()
            res = []
            vmd_k=[]
            if not stepByStep:
                with torch.no_grad():
                    p = 0
                    for batch in test_dataLoader:
                        node_features,  _ = batch
                        node_features = node_features.to(device)
                        y_hat = model(node_features)
                        y_hat_sum = torch.stack(y_hat)
                        # y_hat = inverse_scale(y_hat, test_scale, k)
                        y_hat_sum = torch.sum(y_hat_sum, dim=0)
                        res.append(y_hat_sum.cpu().numpy()[:, -1, :])
                        vmd_k.append(torch.stack(y_hat).cpu().numpy()[:,:,-1])
                    res = np.vstack(res)
                    vmd_k=np.concatenate(vmd_k,axis=1)
            else:
                with torch.no_grad():
                    p = 0
                    res_temp = []
                    for batch in test_dataLoader:
                        node_features,  labels = batch
                        for j in range(pred_len):
                            if j % pred_len == 0:
                                if len(res_temp) > 0:
                                    res.append(res_temp[-1])
                                res_temp = []
                                node_features = node_features
                            else:
                                node_features = np.concatenate((node_features.cpu().numpy()[:, 1:, :],
                                                                res_temp[-1]), axis=1)

                            node_features = torch.Tensor(node_features).to(device)
                            y_pred = model(node_features)
                            res_temp.append(y_pred.cpu().numpy()[:, :, :])
                            # res_temp=np.vstack(res_temp)
                    res.append(res_temp[-1])
                    res = np.vstack(res)[:, -1, :]

            # ------------------后处理，保存结果----------------------
            # 结果加和
            predData=res
            GTData=test_Y[:,-1,:]
            # 分信号存储
            cols=colsName(stations,k)
            df_sub = pd.DataFrame(columns=cols)

            for s in range(stations):
                df_sub['pred_s' + str(s+1)] = predData[:, s]
                df_sub['true_s' + str(s+1)] = GTData[:, s]
                for vk in range(k):
                    df_sub['sub_vmd_pred_k'+str(vk+1)+'_s'+str(s+1)]=vmd_k[vk,:,s]
                    df_sub['sub_vmd_true_k' + str(vk+1)+'_s'+str(s+1)] = test_data[s,seq_len+pred_len:,vk]

            predData = predData.flatten().reshape(-1, 1)
            GTData = GTData.flatten().reshape(-1, 1)
            rm=r2_score(GTData,predData)
            print(rm)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            if r_max < rm and r_max > 0:
                file = Path(f'{savePath}/pred_{pred_len}_{str(k)}_{"stepByStep" if stepByStep else "multiStep"}_{r_max :.6f}.csv')
                print('delete file success')
                file.unlink()
                r_max = rm

                df_sub.to_csv(f'{savePath}/pred_{pred_len}_{str(k)}_{"stepByStep" if stepByStep else "multiStep"}_{rm :.6f}.csv')
            elif r_max == 0:
                r_max = rm
                df_sub.to_csv(f'{savePath}/pred_{pred_len}_{str(k)}_{"stepByStep" if stepByStep else "multiStep"}_{rm :.6f}.csv')


