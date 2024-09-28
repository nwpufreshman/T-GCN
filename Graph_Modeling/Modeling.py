import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import csv

def model_graph(path, dimension, random_number, k_number):
    # 读取数据
    data = pd.read_csv(path, header=None)  # 确保无 header
    features = data.values  # 将数据转为 numpy 数组

    # 进行MDS降维
    mds = MDS(n_components=dimension, random_state=random_number)
    features = mds.fit_transform(features)

    # 进行0-1标准化
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    # print(features)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(features)

    # KNN处理
    k = k_number  # 选择 K 值
    knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(similarity_matrix)
    distances, indices = knn.kneighbors(similarity_matrix)
    # print(indices)
    
    # 构建邻接矩阵
    adjacency_matrix = np.zeros((features.shape[0], features.shape[0]))
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = similarity_matrix[i, neighbors]

    # 基于度矩阵进行标准化
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1) - np.diag(adjacency_matrix))
    # 计算 D^(-1/2) （处理度为 0 的节点）
    degree_inv_sqrt = np.where(degree_matrix > 0, np.linalg.inv(np.sqrt(degree_matrix)), 0)
    # 构建标准化邻接矩阵 A_norm = D^(-1/2) * A * D^(-1/2)
    adjacency_matrix_norm = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
 
    # 恢复对角线为1的自环
    np.fill_diagonal(adjacency_matrix_norm, 1)
    adjacency_matrix_norm = np.around(adjacency_matrix_norm, 2)
    pd.DataFrame(adjacency_matrix_norm).to_csv("adjacency_matrix_norm.csv")

    print(adjacency_matrix_norm)

    return adjacency_matrix_norm

model_graph('./Graph_Modeling/1.csv', 5, 42, 3)