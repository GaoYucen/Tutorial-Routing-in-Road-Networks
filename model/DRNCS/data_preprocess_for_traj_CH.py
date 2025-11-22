import networkx as nx
import geopandas as gpd
import pickle
from tqdm import tqdm
from collections import Counter

# 读取边数据
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')



G = nx.DiGraph()

for i in range(len(edge_df)):
    G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'])
def edge2osmid(edges_seq):
    osmids = []
    for edge_idx in edges_seq:
        osmids.append(edge_df.iloc[edge_idx]['u'])
    osmids.append(edge_df.iloc[edges_seq[-1]]['v'])
    return osmids


import networkx as nx


def calculate_shortcuts(graph, node):
    # 计算收缩节点node后新增的shortcut数量
    predecessors = list(graph.predecessors(node))
    successors = list(graph.successors(node))

    shortcuts = 0
    # 遍历前驱节点和后继节点，检查是否可以通过node形成shortcut
    for pred in predecessors:
        for succ in successors:
            if not graph.has_edge(pred, succ):
                shortcuts += 1
    return shortcuts


def contraction_hierarchy(graph,shortcut_pair, contraction_ratio=0.3):
    # 初始化收缩后的图
    contracted_graph = graph.copy()

    # 确定需要收缩的节点数量
    num_to_contract = int(len(graph) * contraction_ratio)

    # 收缩节点的列表
    nodes_to_contract = []

    for _ in tqdm(range(num_to_contract), desc="收缩节点进度", unit="节点"):
        # 计算当前所有节点的shortcut数量
        shortcut_count = {node: calculate_shortcuts(contracted_graph, node) for node in contracted_graph.nodes()}

        # 按照shortcut数量从小到大排序节点
        sorted_nodes = sorted(shortcut_count.items(), key=lambda x: x[1])

        # 选择shortcut数目最少的节点进行收缩
        node_to_contract = sorted_nodes[0][0]
        nodes_to_contract.append(node_to_contract)

        predecessors = list(contracted_graph.predecessors(node_to_contract))
        successors = list(contracted_graph.successors(node_to_contract))

        # 为每个前驱和后继节点建立直接边
        for pred in predecessors:
            for succ in successors:
                if not contracted_graph.has_edge(pred, succ):
                    contracted_graph.add_edge(pred, succ)
                    shortcut_pair.add((pred, succ))

        # 删除当前节点
        contracted_graph.remove_node(node_to_contract)

    return contracted_graph, nodes_to_contract

contracted_nodes = set()
shortcut_pair = set()

G, contracted_nodes = contraction_hierarchy(G,shortcut_pair, contraction_ratio=0.3)

shortcut_pair = [(u, v) for u, v in shortcut_pair if u not in contracted_nodes and v not in contracted_nodes]

with open('data/chengdu_data/shortcut_pair.pkl', 'wb') as f:
    pickle.dump(shortcut_pair, f)  # 保存数据到文件



with open('data/chengdu_data/contracted_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

print("end")
#%% #%% 计算node_df每个node的后继节点
from collections import defaultdict

node_nbrs = defaultdict(set)

for i in range(len(G.nodes())):
    node = list(G.nodes())[i]
    node_nbrs[node] = set(G.successors(node))

node_nbrs = dict(node_nbrs)
with open('data/chengdu_data/node_nbrs_CH.pkl', 'wb') as f:
    pickle.dump(node_nbrs, f)
    f.close()

print("end")
# %% 处理轨迹数据并保存
name_list = ['train', 'test', 'validation']
for name in name_list:
    with open(f'data/chengdu_data/preprocessed_{name}_trips_small.pkl', 'rb') as f:
        train_data_origin = pickle.load(f)

    train_data = []
    shortcut_sequence = []
    for idx, edges_seq, timestamps in tqdm(train_data_origin, desc="Processing Training Data", unit="batch"):
        osmids = edge2osmid(edges_seq)
        for node1, node2 in shortcut_pair:
            if node1 in osmids and node2 in osmids:
                start_index = osmids.index(node1)
                end_index = osmids.index(node2)
                # 取出这两个节点之间（包括这两个节点）的部分
                shortcut_segment = osmids[start_index:end_index + 1]
                # 仅在 shortcut_segment 不为空时才 append
                if len(shortcut_segment) > 1:  # 确保 segment 不为空
                    shortcut_sequence.append(shortcut_segment)

        # 过滤掉收缩的节点
        osmids = [osm_id for osm_id in osmids if osm_id not in contracted_nodes]
        train_data.append((idx, osmids, timestamps))

    # 保存处理后的数据
    with open(f'data/chengdu_data/preprocessed_{name}_trips_small_osmid_CH.pkl', 'wb') as f:
        pickle.dump(train_data, f)

#%%存储shortcut序列准备用于训练

# shortcut_data = []
#
# # 对 contracted_nodes 中的每一对节点生成序列
# contracted_nodes_list = list(contracted_nodes)  # 将集合转换为列表以便索引
#
# for i in range(len(contracted_nodes_list)):
#     for j in range(i + 1, len(contracted_nodes_list)):
#         node_a = contracted_nodes_list[i]
#         node_b = contracted_nodes_list[j]
#
#         # 根据需要定义生成的节点序列
#         # 假设我们将节点对（node_a, node_b）作为序列存入 shortcut_data
#         sequence = (node_a, node_b)
#         shortcut_data.append(sequence)

# 输出生成的节点序列
print(shortcut_sequence[:10])
print(len(shortcut_sequence))
# 接下来，可以将 shortcut_data 保存到文件中
with open('data/chengdu_data/shortcut_data.pkl', 'wb') as f:
    pickle.dump(shortcut_sequence, f)
