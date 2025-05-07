import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, node_features_list, labels):#, edge_index
        """
        :param node_features_list: 包含多个时间步的节点特征列表，每个元素是 (num_nodes, num_node_features) 的Tensor。
        :param edge_index: 邻接矩阵，shape为 [2, num_edges]。
        :param labels: 每个时间步对应的标签列表。
        """
        self.node_features_list = torch.tensor(node_features_list, dtype=torch.float32)
        # self.edge_index = torch.tensor(edge_index, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        # self.edge_attr = edge_attr

    def __len__(self):
        # 数据集的长度是时间步的数量
        return len(self.node_features_list)

    def __getitem__(self, idx):
        # 获取特定时间步的节点特征和对应标签
        node_features = self.node_features_list[idx]
        label = self.labels[idx]

        # 返回节点特征、边信息和标签
        return node_features, label
        #, self.edge_index[idx]

