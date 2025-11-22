import pickle
import geopandas as gpd
import networkx as nx

node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
with open('data/chengdu_data/best_paths_hierarchy.pkl', 'rb') as pickle_file:
    partial_true_best_paths_dict = pickle.load(pickle_file)
# 加载数据
with open('data/chengdu_data/shortcut_pair_hierarchy.pkl', 'rb') as f:
    origin_shortcut_data = pickle.load(f)

existing_pairs = set(partial_true_best_paths_dict.keys())

# 筛选出不在 partial_true_best_paths_dict 中的节点对
shortcut_data = [pair for pair in origin_shortcut_data if pair not in existing_pairs]
# 读取边数据
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')
#生成图
G = nx.DiGraph()
for i in range(len(edge_df)):
    G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'])
start_nodes = []
end_nodes = []

for pair in shortcut_data :
    start_nodes.append(pair[0])  # 添加 source 节点
    end_nodes.append(pair[1])

#%% 读取节点嵌入
with open('data/chengdu_data/embeddings.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

#%% 添加key为-1的embedding，指定dtype为float32
import numpy as np
node_embeddings[-1] = np.array([0] * len(node_embeddings[288416374])).astype(np.float32)

#%% 读取node_nbrs
with open('data/chengdu_data/node_nbrs.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

#%% 确认node_nbrs的最大尺寸
max_nbrs = 0
for node in node_nbrs:
    if len(node_nbrs[node]) > max_nbrs:
        max_nbrs = len(node_nbrs[node])

#%% 将node_nbrs长度不到max_nbrs的补充到max_nbrs长度
for node in node_nbrs:
    node_nbrs[node] = list(node_nbrs[node])
    if len(node_nbrs[node]) < max_nbrs:
        node_nbrs[node] += [-1] * (max_nbrs - len(node_nbrs[node]))

#%% 训练
num_epoches = 10
batch_size = 1

from tqdm import tqdm
import torch
from model import Model
import random

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = Model(embedding=node_embeddings).to(device)


def dijkstra(true_trip, max_nbrs, model, G):
    src = true_trip[0]
    dest = true_trip[1]
    g = G

    with torch.no_grad():
        current_temp = [c for c in g.nodes()]

        current = [
            c for c in current_temp
            for _ in (node_nbrs[c] if c in node_nbrs else [])
        ]

        pot_next = [
            nbr for c in current_temp
            for nbr in (node_nbrs[c] if c in node_nbrs else [])
        ]

        dests = [
            dest for c in current_temp
            for _ in (node_nbrs[c] if c in node_nbrs else [])
        ]

        source_array = np.array([node_embeddings[node] for node in current])
        source_embed = torch.from_numpy(source_array).to(device)
        dest_array = np.array([node_embeddings[node] for node in dests])
        dest_embed = torch.from_numpy(dest_array).to(device)
        nbr_embed = torch.tensor([node_embeddings[node] for node in pot_next]).to(device)
        input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)

        traffic = None
        unnormalized_confidence = model(input_embed)
        unnormalized_confidence = -1 * torch.nn.functional.log_softmax(
            unnormalized_confidence.reshape(-1, max_nbrs),
            dim=1
        )

        transition_nll = unnormalized_confidence.detach().cpu().tolist()

    torch.cuda.empty_cache()

    count = 0
    for u in g.nodes():
        for i, nbr in enumerate(node_nbrs[u]):
            if nbr == -1:
                break
            g[u][nbr]["nll"] = transition_nll[count][i]
        count += 1

    path = nx.dijkstra_path(g, src, dest, weight="nll")
    path = [x for x in path]

    return path


# %% 使用模型进行测试
# 加载模型参数
model.load_state_dict(torch.load('param/model.pth'))
model.eval()  # 设置模型为评估模式

# 准备测试数据
predictions = []
targets = []
import numpy as np
import torch


from tqdm import tqdm

# 假设 all_predicted_paths 已经定义为一个空列表
all_predicted_paths = []

# 为计算预测路径的循环添加进度条
for pair in tqdm(shortcut_data, desc="Calculating Predicted Paths", total=len(shortcut_data)):
    all_predicted_paths.append(dijkstra(pair, max_nbrs, model, G))

predicted_paths_dict = {}

# 为处理起始节点的循环添加进度条
for i in tqdm(range(len(start_nodes)), desc="Creating Predicted Paths Dictionary", total=len(start_nodes)):
    start_node = start_nodes[i]
    end_node = end_nodes[i]
    predict_path = all_predicted_paths[i]
        # 检查路径的起点与终点
    if predict_path[0] == start_node and predict_path[-1] == end_node:
        # 如果匹配，将其添加到字典中，键为 (start_node, end_node)，值为 predicted_path
        predicted_paths_dict[(start_node, end_node)] = predict_path


# 输出结果
print("预测路径字典创建完成")


# #%% Haversine 函数
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371000  # 地球半径，单位是米
#     phi1 = np.radians(lat1)
#     phi2 = np.radians(lat2)
#     delta_phi = np.radians(lat2 - lat1)
#     delta_lambda = np.radians(lon2 - lon1)
#
#     a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#
#     distance = R * c  # 计算距离
#     return distance
#
#
# # 动态预测函数
# def dynamic_predict(start_node, end_node):
#     predicted_path = [start_node]
#     current_node = start_node
#     max_iterations = 10  # 设置最大迭代次数
#     iteration_count = 0  # 初始化计数器
#
#     while current_node != end_node and iteration_count < max_iterations:
#         if current_node in node_nbrs:
#             nbr = []
#             for node in predicted_path:  # 遍历路径中的每个节点
#                 if node in node_nbrs:  # 确保 current_node 在 node_nbrs 中
#                     nbr.extend(node_nbrs[node])  # 将邻居添加到 all_nbrs
#
#             source = [item for item in predicted_path for _ in range(max_nbrs)]
#             dest = [end_node] * (max_nbrs * len(predicted_path))  # 每个 dest 重复最大邻居节点数次
#
#             source_array = np.array([node_embeddings[node] for node in source])
#             source_embed = torch.from_numpy(source_array).to(device)
#             dest_array = np.array([node_embeddings[node] for node in dest])
#             dest_embed = torch.from_numpy(dest_array).to(device)
#             nbr_embed = torch.tensor([node_embeddings[node] for node in nbr]).to(device)
#
#             input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
#
#             pred = model(input_embed)
#             # 构造mask矩阵
#             mask = torch.tensor([1 if nbr[i] != -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
#             # 将pred中对应nbr==-1的部分置为0
#             pred = pred * mask
#             predictions = pred.view(-1, max_nbrs).argmax(dim=1).tolist()
#             predicted_path_tem = [start_node]
#             index_offset = 0
#             for j in range(len(predicted_path) -1):
#                 # 寻找对应的预测结果
#                 next_node = node_nbrs[predicted_path[j]][predictions[index_offset]]
#                 predicted_path_tem.append(next_node)
#                 index_offset += 1
#
#
#             next_node = node_nbrs[predicted_path[-1]][predictions[-1]]
#
#             predicted_path = predicted_path_tem
#             predicted_path.append(next_node)
#             # 计算 next_node 和 end_node 之间的距离
#             if next_node != end_node and not node_df[node_df['osmid'] == next_node].empty:
#                 lat1, lon1 = node_df[node_df['osmid'] == next_node].iloc[0, 0], \
#                     node_df[node_df['osmid'] == next_node].iloc[0, 1]
#
#                 lat2, lon2 = node_df[node_df['osmid'] == end_node].iloc[0, 0], \
#                     node_df[node_df['osmid'] == end_node].iloc[0, 1]
#                 distance = haversine(lon1, lat1, lon2, lat2)
#
#                 # 如果距离小于 50 米，则直接添加 end_node
#                 if distance < 50:
#                     predicted_path.append(end_node)
#                     break
#
#
#             current_node = next_node
#
#         else:
#             break
#
#         iteration_count += 1  # 增加计数器
#
#     return predicted_path
#
#
# all_predicted_paths = []
# # 使用函数进行路径预测
# for start_node, end_node in zip(start_nodes, end_nodes):
#     predicted_path = dynamic_predict(start_node, end_node)
#     all_predicted_paths.append(predicted_path)
#
# import json
# # print(all_predicted_paths[:20])  # 输出预测路径
# # 创建一个空字典用于存储结果
# predicted_paths_dict = {}
# for i in range(len(start_nodes)):
#     start_node = start_nodes[i]
#     end_node = end_nodes[i]
#     predict_path = all_predicted_paths[i]
#
#     # 检查路径的起点与终点
#     if predict_path[0] == start_node and predict_path[-1] == end_node:
#         # 如果匹配，将其添加到字典中，键为 (start_node, end_node)，值为 predicted_path
#         predicted_paths_dict[(start_node, end_node)] = predict_path

# 现在 predicted_paths_dict 中存储了符合条件的路径
print(len(predicted_paths_dict))

file_name = 'data/chengdu_data/predicted_paths_dict_hierarchy.pkl'
with open(file_name, 'wb') as pickle_file:
    pickle.dump(predicted_paths_dict, pickle_file)



