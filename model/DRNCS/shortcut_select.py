import pickle
from collections import defaultdict

# 加载数据
with open('data/chengdu_data/shortcut_data_hierarchy.pkl', 'rb') as pickle_file:
    partial_true_paths = pickle.load(pickle_file)

# 创建字典存储路径
path_dict = defaultdict(list)

# 假设partial_true_paths是一个包含多个路径的列表，每个路径是一个 (start_node, end_node, path) 元组
from tqdm import tqdm

# 遍历 partial_true_paths，更新 path_dict
for path in tqdm(partial_true_paths, desc="Updating Path Dictionary", total=len(partial_true_paths)):
    path_dict[(path[0], path[-1])].append(path)

import numpy as np

def calculate_depth_accuracy(true_path, predicted_path):
    """
    计算路径的深度准确率。
    深度准确率 = 交集的长度 / 真实路径的长度。
    """
    intersection_length = len(set(true_path).intersection(predicted_path))
    true_length = len(true_path)
    return intersection_length / true_length if true_length > 0 else 0

def calculate_reachability(true_path, predicted_path, end_node):
    """
    计算路径的可达性。可达性为1表示必须到达终点。
    """
    if predicted_path[-1] == end_node:
        return 1
    return 0

def select_best_path(paths, true_path, end_node):
    """
    从多条路径中选择最佳路径，优化深度准确率，并保证可达性为1。
    """
    best_path = None
    best_average_accuracy = -1
    best_reachability = 0

    # 计算每条路径的深度准确率
    depth_accuracies = [calculate_depth_accuracy(true_path, path) for path in paths]

    for i, path in enumerate(paths):
        # 计算当前路径的深度准确率
        depth_accuracy = depth_accuracies[i]
        reachability = calculate_reachability(true_path, path, end_node)

        # 计算当前路径的平均深度准确率（不包括自身）
        average_accuracy = np.mean([acc for j, acc in enumerate(depth_accuracies) if j != i])

        # 选择满足可达性为1且平均深度准确率最高的路径
        if reachability == 1 and average_accuracy > best_average_accuracy:
            best_path = path
            best_average_accuracy = average_accuracy
            best_reachability = reachability

    return best_path

def dynamic_generate(start_node, end_node, path_dict, max_iterations=10):
    """
    动态生成最佳路径：通过路径集合进行优化，选择深度准确率最大且可达性为1的路径。
    """
    if (start_node, end_node) not in path_dict:
        return []  # 如果路径集合为空，返回空路径

    paths = path_dict[(start_node, end_node)]  # 获取该节点对的所有路径
    # 假设我们已经有一条真实路径true_path作为参考
    # 你可以在此通过某种方式选择或指定true_path

    best_path = select_best_path(paths, true_path=[], end_node=end_node)

    # 返回最佳路径
    return best_path

best_paths_dict = {}



# 遍历所有节点对，生成最佳路径
for (start_node, end_node), paths in tqdm(path_dict.items(), desc="Generating Best Paths", total=len(path_dict)):
    best_path = dynamic_generate(start_node, end_node, path_dict)
    best_paths_dict[(start_node, end_node)] = best_path


# with open('data/chengdu_data/predicted_paths_dict.pkl', 'rb') as pickle_file:
#     predicted_paths_dict = pickle.load(pickle_file)
#
# for index, values in predicted_paths_dict.items():
#     # 如果 index 不在 best_paths_dict 中
#     if index not in best_paths_dict:
#         # 将该 index 和对应的值添加到 best_paths_dict
#         best_paths_dict[index] = values
# 保存最佳路径字典
with open('data/chengdu_data/best_paths_hierarchy.pkl', 'wb') as f:
    pickle.dump(best_paths_dict, f)

# #%%重新再修改一下G
# import networkx as nx
# import geopandas as gpd
#
#
# # 读取边数据
# edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')
#
#
#
# G = nx.DiGraph()
#
# for i in range(len(edge_df)):
#     G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'])
#
# for (start, end) in best_paths_dict.keys():
#     # 添加边从起点到终点，不设置权重
#     if not G.has_edge(start, end):
#         # 如果边不存在，则添加边
#         G.add_edge(start, end)
#
#
# #%% #%% 计算node_df每个node的后继节点
# from collections import defaultdict
#
# node_nbrs = defaultdict(set)
#
# for i in range(len(G.nodes())):
#     node = list(G.nodes())[i]
#     node_nbrs[node] = set(G.successors(node))
#
# node_nbrs = dict(node_nbrs)
# with open('data/chengdu_data/node_nbrs_hierarchy.pkl', 'wb') as f:
#     pickle.dump(node_nbrs, f)
#     f.close()