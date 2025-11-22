# from shortcut_dynamic_merge import dijkstra
import pandas as pd
import geopandas as gpd
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
# from sklearn.cluster import KMeans
import time
from haversine import haversine
import networkx as nx
import multiprocessing as mp
from scipy import stats
import statistics
from termcolor import cprint, colored
import sys
from collections import OrderedDict
import datetime
MAX_ITERS = 300
JUMP = 1000
node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')


# def haversine(lat1, lon1, lat2, lon2):
#     # 标准哈弗辛公式的实现
#     from math import radians, sin, cos, sqrt, atan2
#
#     # 将经纬度从度转换为弧度
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#
#     # 哈弗辛公式
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))
#
#     # 地球半径 (km)
#     radius = 6371.0
#     distance = radius * c
#     return distance
#%%双向最短路
import networkx as nx
import heapq

def bidirectional_dijkstra(graph, start, goal):
    if start == goal:
        return [start]

    # 前向和后向堆
    forward_heap = [(0, start)]  # (距离, 节点)
    backward_heap = [(0, goal)]

    # 前向和后向的距离
    forward_distances = {start: 0}
    backward_distances = {goal: 0}

    # 前向和后向的前驱节点
    forward_predecessors = {start: None}
    backward_predecessors = {goal: None}

    while forward_heap and backward_heap:
        # 扩展前向方向
        forward_distance, forward_node = heapq.heappop(forward_heap)

        if forward_node in backward_distances:
            return reconstruct_path(forward_predecessors, backward_predecessors, forward_node)

        for neighbor in graph.neighbors(forward_node):
            weight = graph[forward_node][neighbor]['nll']
            distance = forward_distance + weight
            if neighbor not in forward_distances or distance < forward_distances[neighbor]:
                forward_distances[neighbor] = distance
                forward_predecessors[neighbor] = forward_node
                heapq.heappush(forward_heap, (distance, neighbor))

        # 扩展后向方向
        backward_distance, backward_node = heapq.heappop(backward_heap)

        if backward_node in forward_distances:
            return reconstruct_path(forward_predecessors, backward_predecessors, backward_node)

        for neighbor in graph.neighbors(backward_node):
            weight = graph[backward_node][neighbor]['nll']
            distance = backward_distance + weight
            if neighbor not in backward_distances or distance < backward_distances[neighbor]:
                backward_distances[neighbor] = distance
                backward_predecessors[neighbor] = backward_node
                heapq.heappush(backward_heap, (distance, neighbor))

    return None  # 如果没有找到路径

def reconstruct_path(forward_predecessors, backward_predecessors, meeting_node):
    # 从前向和后向路径重构最短路径
    path = []

    # 从起点到 meeting_node
    current = meeting_node
    while current is not None:
        path.append(current)
        current = forward_predecessors[current]
    path.reverse()

    # 从 meeting_node 到终点
    current = backward_predecessors[meeting_node]
    while current is not None:
        path.append(current)
        current = backward_predecessors[current]

    return path
def dijkstra(true_trip, max_nbrs, model, G):
    src = true_trip[1][0]
    dest = true_trip[1][-1]
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

    path =nx.dijkstra_path (g, src, dest)
    path = [x for x in path]
    new_sequence = []
    j = 0
    while j < len(path):
        # 取出相邻节点对
        if j < len(path) - 1:  # 确保有相邻节点对可检查
            node_pair = (path[j], path[j + 1])

            # 检查这个节点对是否在字典中
            if node_pair in dict:
                # 替换这个节点对为字典中的值（更长的节点序列）
                new_sequence.extend(dict[node_pair])  # 将替换的序列扩展到新序列中
                j += 1  # 跳过已替换的两个节点
                continue

        # 如果没有找到替换，保留当前节点
        if not new_sequence or new_sequence[-1] != path[j]:
            new_sequence.append(path[j])
        j += 1

    return new_sequence


def trip_length(path):



    return len(path)

def intersections_and_unions(path1, path2):
    intersection_nodes = set()  # 用于存储所有交集节点

    # 计算交集：检查路径中相邻的节点是否匹配
    for i in range(len(path1) - 1):
        if path1[i] in path2 and path1[i + 1] in path2:
            intersection_nodes.add(path1[i])     # 添加当前节点
            intersection_nodes.add(path1[i + 1]) # 添加下一个节点

    # 求并集：不同节点总数
    union_count = len(set(path1).union(set(path2)))

    return len(intersection_nodes), union_count

def shorten_path(path, true_dest):
    global node_df  # 确保 node_df 是全局变量

    # 获取目标节点的第一个连接节点
    dest_node = true_dest

    # 从 node_df 获取目标节点的经纬度
    lat_dest, lon_dest = node_df[node_df['osmid'] == dest_node].iloc[0, 0], \
                          node_df[node_df['osmid'] == dest_node].iloc[0, 1]

    # 计算与目标节点的最小哈弗辛距离
    _, index = min(
        [
            (
                haversine(
                    (node_df[node_df['osmid'] == path[i]].iloc[0, 0],
                     node_df[node_df['osmid'] == path[i]].iloc[0, 1]),
                    (lat_dest, lon_dest)
                ),
                i
            )
            for i, edge in enumerate(path)
        ]
    )

    return path[:index + 1]

def remove_duplicates(new_sequence):
    seen = set()
    return [seen.add(node) or node for node in new_sequence if node not in seen]

def gen_paths_no_hierarchy(all_paths):
    global JUMP
    ans = []

    # 分批处理路径
    for i in tqdm(list(range(0, len(all_paths), JUMP)), desc="batch_eval", dynamic_ncols=True):
        temp = all_paths[i:i + JUMP]
        ans.append(gen_paths_no_hierarchy_helper(temp))

    return [t for sublist in ans for t in sublist]


def gen_paths_no_hierarchy_helper(all_paths):
    global model, node_nbrs, max_nbrs, dict, node_embeddings
    true_paths = [p for _, p in all_paths]
    total_time = 0  # 初始化总时间
    iterations = 0  # 初始化迭代计数器
    model.eval()
    gens = [[t[0]] for t in true_paths]
    pending = OrderedDict({i: None for i in range(len(all_paths))})
    with torch.no_grad():
        for _ in tqdm(range(MAX_ITERS), desc="generating trips in lockstep", dynamic_ncols=True):
            start_time = time.time()
            true_paths = [all_paths[i][1] for i in pending]
            current_temp = [gens[i][-1] for i in pending]
            current = [c for c in current_temp for _ in node_nbrs[c]]
            pot_next = [nbr for c in current_temp for nbr in node_nbrs[c]]
            dests = [t[-1] for c, t in zip(current_temp, true_paths) for _ in (node_nbrs[c] if c in node_nbrs else [])]
            source_array = np.array([node_embeddings[node] for node in current])
            source_embed = torch.from_numpy(source_array).to(device)
            dest_array = np.array([node_embeddings[node] for node in dests])
            dest_embed = torch.from_numpy(dest_array).to(device)
            nbr_array = np.array([node_embeddings[node] for node in pot_next])
            nbr_embed = torch.from_numpy(nbr_array).to(device)
            input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)

            traffic = None
            unnormalized_confidence = model(input_embed)
            mask = torch.tensor([1 if pot_next[i] != -1 else 0 for i in range(len(pot_next))]).to(device).unsqueeze(1)
            unnormalized_confidence = unnormalized_confidence*mask
            chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_nbrs), dim=1)
            chosen = chosen.detach().cpu().tolist()
            pending_trip_ids = list(pending.keys())
            for identity, choice in zip(pending_trip_ids, chosen):
                choice = node_nbrs[gens[identity][-1]][choice]
                last = gens[identity][-1]
                if choice in gens[identity] or choice == -1:
                    del pending[identity]
                    continue
                gens[identity].append(choice)
                if choice == all_paths[identity][1][-1]:
                    del pending[identity]

            end_time = time.time()  # 记录结束时间
            iteration_time = end_time - start_time  # 计算当前迭代耗时
            total_time += iteration_time  # 累加总时间

            if len(pending) == 0:
                break
    gens = [shorten_path(gen, true[1][-1]) if gen[-1] != true[1][-1] else gen for gen, true in (zip(gens, all_paths))]
    new_gens = []
    # 遍历 gens 中的每条节点序列
    for sequence in tqdm(gens, desc="Processing sequences", unit="sequence"):
        # 用于存储新序列
        new_sequence = []
        j = 0
        while j < len(sequence):
            # 取出相邻节点对
            if j < len(sequence) - 1:  # 确保有相邻节点对可检查
                node_pair = (sequence[j], sequence[j + 1])

                # 检查这个节点对是否在字典中
                if node_pair in dict:
                    # 替换这个节点对为字典中的值（更长的节点序列）
                    new_sequence.extend(dict[node_pair])  # 将替换的序列扩展到新序列中
                    j += 1  # 跳过已替换的两个节点
                    continue

            # 如果没有找到替换，保留当前节点
            if not new_sequence or new_sequence[-1] != sequence[j]:
                new_sequence.append(sequence[j])
            j += 1

        # 将新的节点序列添加到 new_gens 中
        new_gens.append(remove_duplicates(new_sequence))
    # new_gens = []

    # # 遍历 gens 中的每条节点序列
    # for sequence in tqdm(gens, desc="Processing sequences", unit="sequence"):
    #     # 用于存储新序列
    #     new_sequence = []
    #     j = 0
    #
    #     while j < len(sequence):
    #         # 标记是否进行了替换
    #         replaced = False
    #
    #         # 检查顺序节点对，从当前节点开始逐个检查
    #         for k in range(j + 1, len(sequence)):
    #             node_pair = (sequence[j], sequence[k])
    #
    #             # 检查这个节点对是否在字典中
    #             if node_pair in dict:
    #                 # 替换这个节点对及之间的部分为字典中的值（更长的节点序列）
    #                 new_sequence.extend(dict[node_pair])  # 将替换的序列扩展到新序列中
    #
    #                 # 更新 j 到替换部分的结束位置
    #                 j = k  # k 是当前检查的节点，继续下一轮循环
    #                 replaced = True
    #                 break  # 找到一对后跳出循环，避免重复替换
    #
    #         # 如果没有找到替换，保留当前节点
    #         if not replaced:
    #             # 确保不添加重复的节点
    #             if not new_sequence or new_sequence[-1] != sequence[j]:
    #                 new_sequence.append(sequence[j])
    #             j += 1  # 继续下一个节点
    #
    #     # 将新的节点序列添加到 new_gens 中
    #     new_gens.append(remove_duplicates(new_sequence))
    average_time = total_time / len(all_paths)
    print(f"Average time to generate a path: {average_time:.4f} seconds")
    model.train()
    return new_gens


def evaluate_no_hierarchy(data, num=1000, with_dijkstra=False):
    global contracted_G
    to_do = ["precision", "recall", "reachability", "avg_reachability", "acc", "nll", "generated"]
    results = {s: None for s in to_do}
    cprint("Evaluating {} number of trips".format(num), "magenta")
    partial = random.sample(data, num)
    t1 = time.time()
    if with_dijkstra:
        gens = gens = [dijkstra(t,max_nbrs,model,G) for t in tqdm(partial, desc = "Dijkstra for generation", unit = "trip", dynamic_ncols=True)]
    else:
        gens = gen_paths_no_hierarchy(partial)
    elapsed = time.time() - t1
    results["time"] = elapsed
    preserved_with_stamps = partial.copy()
    partial = [p for _, p in partial]
    print("Without correction (everything is weighed according to the edge lengths)")
    generated = list(zip(partial, gens))
    generated = [(t, g) for t, g in generated if len(t) > 1]
    lengths = [(trip_length(t), trip_length(g)) for (t, g) in generated]
    inter_union = [intersections_and_unions(t, g) for (t, g) in generated]
    inters = [inter for inter, union in inter_union]
    lengths_gen = [l_g for l_t, l_g in lengths]
    lengths_true = [l_t for l_t, l_g in lengths]
    precs = [i / l if l > 0 else 0 for i, l in zip(inters, lengths_gen)]
    precision1 = round(100 * sum(precs) / len(precs), 2)
    recs = [i / l if l > 0 else 0 for i, l in zip(inters, lengths_true)]
    recall1 = round(100 * sum(recs) / len(recs), 2)
    deepst_accs = [i / max(l1, l2) for i, l1, l2 in zip(inters, lengths_true, lengths_gen) if max(l1, l2) > 0]
    deepst = round(100 * sum(deepst_accs) / len(deepst_accs), 2)
    num_reached = 0
    lefts = [haversine((node_df[node_df['osmid'] == g[0]].iloc[0, 0],
                     node_df[node_df['osmid'] == g[0]].iloc[0, 1]),
                       (node_df[node_df['osmid'] == t[0]].iloc[0, 0],
                        node_df[node_df['osmid'] == t[0]].iloc[0, 1])) for t, g in generated]
    rights = [haversine((node_df[node_df['osmid'] == g[-1]].iloc[0, 0],
                        node_df[node_df['osmid'] == g[-1]].iloc[0, 1]),
                       (node_df[node_df['osmid'] == t[-1]].iloc[0, 0],
                        node_df[node_df['osmid'] == t[-1]].iloc[0, 1])) for t, g in generated]
    for r in rights:
        if r*1000<=50:
            num_reached+=1
    reachability = [(l + r) / 2 for (l, r) in zip(lefts, rights)]

    all_reach = np.mean(reachability)
    all_reach = round(1000 * all_reach, 2)

    if len(reachability) != num_reached:
        reach = sum(reachability) / (len(reachability) - num_reached)
    else:
        reach = 0
    reach = round(1000 * reach, 2)
    percent_reached = round(100 * (num_reached / len(reachability)), 2)
    print()
    cprint("Precision is                            {}%".format(precision1), "green")
    cprint("Recall is                               {}%".format(recall1), "green")
    print()
    cprint("%age of trips reached is                {}%".format(percent_reached), "green")
    cprint("Avg Reachability(across all trips) is   {}m".format(all_reach), "green")
    print()
    results["precision"] = precision1
    results["reachability"] = percent_reached
    results["avg_reachability"] = (all_reach, reach)
    results["recall"] = recall1
    results["generated"] = list(zip(preserved_with_stamps, gens))
    return results

#%% 读取轨迹数据
import pickle
import random

with open('data/chengdu_data/contracted_graph.pkl', 'rb') as f:
    contracted_G = pickle.load(f)

with open('data/chengdu_data/predicted_paths_dict.pkl', 'rb') as pickle_file:
    predicted_paths_dict = pickle.load(pickle_file)

with open('data/chengdu_data/best_paths.pkl', 'rb') as pickle_file:
    best_paths_dict = pickle.load(pickle_file)

dict = {**predicted_paths_dict, **best_paths_dict}

filtered_dict = {key: value for key, value in dict.items() if value is not None and (not isinstance(value, (list, str, dict)) or len(value) > 1)}

dict = filtered_dict

with open('data/chengdu_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

with open('data/chengdu_data/preprocessed_validation_trips_small_osmid.pkl', 'rb') as f:
    val_data = pickle.load(f)
    f.close()

with open('data/chengdu_data/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

# # 构建数据集
# for i in range(len(train_data)):
#     train_data[i] = ([train_data[i][1][0], train_data[i][1][-1], train_data[i][2][0]], train_data[i][1])
#
# for i in range(len(val_data)):
#     val_data[i] = ([val_data[i][1][0], val_data[i][1][-1], val_data[i][2][0]], val_data[i][1])
#
# for i in range(len(test_data)):
#     test_data[i] = ([test_data[i][1][0], test_data[i][1][-1], test_data[i][2][0]], test_data[i][1])

#%% map数据
import geopandas as gpd

node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%%
# print(node_df.head(5))
# print(edge_df.iloc[0])


#%% 构造图G
import networkx as nx

G = nx.DiGraph()

for i in range(len(edge_df)):
    G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'], weight=edge_df.iloc[i]['length'])


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
#%%训练数据准备
train_data_filtered = []  # 新建一个列表用于存储过滤后的数据
for i in range(len(train_data)):
    # 检查 train_data[i][1][0] 和 train_data[i][1][-1] 是否在 node_nbrs[0] 中
    if train_data[i][1][0] in node_nbrs and train_data[i][1][-1] in node_nbrs:
        train_data_filtered.append(([train_data[i][1][0], train_data[i][1][-1], train_data[i][2][0]], train_data[i][1]))

train_data = train_data_filtered  # 用过滤后的数据替换原数据

# 处理 val_data
val_data_filtered = []
for i in range(len(val_data)):
    if val_data[i][1][0] in node_nbrs and val_data[i][1][-1] in node_nbrs:
        val_data_filtered.append(([val_data[i][1][0], val_data[i][1][-1], val_data[i][2][0]], val_data[i][1]))

val_data = val_data_filtered  # 用过滤后的数据替换原数据

# 处理 test_data
test_data_filtered = []
for i in range(len(test_data)):
    if test_data[i][1][0] in node_nbrs and test_data[i][1][-1] in node_nbrs:
        test_data_filtered.append(([test_data[i][1][0], test_data[i][1][-1], test_data[i][2][0]], test_data[i][1]))

test_data = test_data_filtered  # 用过滤后的数据替换原数据


batch_size = 64
from tqdm import tqdm
import torch
from model import Model
import random

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = Model(embedding=node_embeddings).to(device)


# %% 使用模型进行测试
# 加载模型参数
model.load_state_dict(torch.load('param/model_CH.pth'))
model.eval()  # 设置模型为评估模式

# 准备测试数据
predictions = []
targets = []

evaluate_no_hierarchy(val_data,JUMP,0)

# with torch.no_grad():  # 不需要梯度计算，提高速度并减少内存消耗
#     for i in range(0, len(test_data), batch_size):
#         batch = [item[1] for item in test_data[i:i + batch_size]]
#         source = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
#         dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
#         nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
#
#         source_array = np.array([node_embeddings[node] for node in source])
#         source_embed = torch.from_numpy(source_array).to(device)
#         dest_array = np.array([node_embeddings[node] for node in dest])
#         dest_embed = torch.from_numpy(dest_array).to(device)
#         nbr_embed = torch.tensor([node_embeddings[node] for node in nbr]).to(device)
#
#         input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
#
#         # 进行预测
#         pred = model(input_embed)
#
#         # 构造mask矩阵
#         mask = torch.tensor([1 if nbr[i] != -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
#         # 将pred中对应nbr==-1的部分置为0
#         pred = pred * mask
#
#         # 获取真实目标
#         true_target = torch.tensor(
#             [node_nbrs[item[j]].index(item[j + 1]) for item in batch for j in range(len(item) - 1)]).to(device)
#
#         predictions.extend(pred.view(-1, max_nbrs).argmax(dim=1).tolist())  # 保存预测结果
#         targets.extend(true_target.tolist())  # 保存真实标签
