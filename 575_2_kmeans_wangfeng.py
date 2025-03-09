import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------
# 1. 数据加载与展示模块
# ---------------------------
# 加载.mat文件
loaded_data = loadmat('AllSamples.mat')
sample_points = loaded_data['AllSamples']  # 原始NumPy数组格式

# 转换为Pandas DataFrame并展示前10行
df = pd.DataFrame(sample_points, columns=['Feature1', 'Feature2'])
print("\n=== 数据集前10行样本（Pandas展示）===")
print(df.head(10))

# ---------------------------
# 2. 初始化策略实现模块
# ---------------------------
def initialize_centers_strategy2(data_points, num_clusters, random_seed=None):
    """策略2初始化：迭代选择最远样本作为初始簇中心"""
    np.random.seed(random_seed)
    num_samples = data_points.shape[0]
    cluster_centers = np.zeros((num_clusters, data_points.shape[1]))
    
    # 第一步随机选择初始中心
    cluster_centers[0] = data_points[np.random.choice(num_samples, 1), :]
    
    # 后续选择到已有中心平均距离最大的样本
    for center_index in range(1, num_clusters):
        average_distances = np.zeros(num_samples)
        for point_index in range(num_samples):
            diffs = cluster_centers[:center_index] - data_points[point_index]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            average_distances[point_index] = np.mean(dists)
        cluster_centers[center_index] = data_points[np.argmax(average_distances)]
    return cluster_centers

# ---------------------------
# 3. 核心聚类算法模块
# ---------------------------
def calculate_distances(X, centers):
    """手动计算所有数据点到各簇中心的欧氏距离"""
    n = X.shape[0]
    k = centers.shape[0]
    X_expanded = X[:, np.newaxis, :]  # 形状 (n, 1, 2)
    centers_expanded = centers[np.newaxis, :, :]  # 形状 (1, k, 2)
    diffs = X_expanded - centers_expanded
    distances = np.sqrt(np.sum(diffs**2, axis=2))
    return distances

def perform_kmeans_clustering(data_points, num_clusters, initialization_method, 
                             max_iter=100, tol=1e-5, random_seed=None):
    """支持两种初始化策略的k均值聚类实现"""
    np.random.seed(random_seed)
    
    # 初始化簇中心
    if initialization_method == 1:
        indices = np.random.permutation(data_points.shape[0])[:num_clusters]
        centers = data_points[indices]
    else:
        centers = initialize_centers_strategy2(data_points, num_clusters, random_seed)
    
    prev_centers = centers.copy()
    for _ in range(max_iter):
        # 计算距离矩阵
        distances = calculate_distances(data_points, centers)
        labels = np.argmin(distances, axis=1)
        
        # 更新中心
        new_centers = np.zeros_like(centers)
        for i in range(num_clusters):
            members = data_points[labels == i]
            new_centers[i] = members.mean(axis=0) if len(members) > 0 else data_points[np.random.choice(data_points.shape[0], 1)]
        
        # 收敛检查
        if np.max(np.linalg.norm(new_centers - prev_centers, axis=1)) < tol:
            break
        prev_centers = new_centers
    
    # 计算目标函数
    obj = 0
    for i in range(num_clusters):
        members = data_points[labels == i]
        if len(members) > 0:
            obj += np.sum((members - new_centers[i])**2)
    return obj

# ---------------------------
# 4. 实验执行与可视化模块
# ---------------------------
# 实验参数设置
k_values = np.arange(2, 11)
strategies = {
    "Strategy1": {"method": 1, "seeds": [42, 99]},
    "Strategy2": {"method": 2, "seeds": [123, 456]}
}

# 执行实验
results = {}
for strategy in strategies:
    method = strategies[strategy]["method"]
    results[strategy] = []
    for seed in strategies[strategy]["seeds"]:
        obj_values = []
        for k in k_values:
            obj = perform_kmeans_clustering(sample_points, k, method, random_seed=seed)
            obj_values.append(obj)
        results[strategy].append(obj_values)

# 可视化结果
plt.figure(figsize=(10, 6))
colors = {'Strategy1': ['#1f77b4', '#aec7e8'], 'Strategy2': ['#ff7f0e', '#ffbb78']}
for strategy in results:
    for i, run in enumerate(results[strategy]):
        linestyle = '-' if i == 0 else '--'
        plt.plot(k_values, run, marker='o' if i == 0 else 's', 
                 linestyle=linestyle, color=colors[strategy][i],
                 label=f'{strategy} Run{i+1}')

plt.title('K-means Objective Function Comparison')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors')
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()