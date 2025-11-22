#%% map数据
import geopandas as gpd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
node_df = gpd.read_file('data/beijing_data/map/nodes.shp')
edge_df = gpd.read_file('data/beijing_data/map/edges.shp')
from tqdm import tqdm
#%%
print(node_df.head(5))
print(edge_df.head(5))
print(len(node_df))
print(edge_df.iloc[0])
#%% 构造图G
import networkx as nx

G = nx.DiGraph()


with open('data/beijing_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

for i in range(len(train_data)):
    train_data[i] = ([train_data[i][1][0], train_data[i][1][-1]], train_data[i][1])



# 创建图
G = nx.DiGraph()

# 添加边和权重
for i in range(len(edge_df)):
    G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'], weight=edge_df.iloc[i]['length'])

# 存储最短路径
shortest_paths_dict = {}
true_paths_dict = {}


# 计算最短路径
for i in range(len(train_data)):
    # 计算最短路径
    shortest_path = nx.dijkstra_path(G, train_data[i][0][0], train_data[i][0][1])
    shortest_paths_dict[(train_data[i][0][0], train_data[i][0][1])] = shortest_path
    true_paths_dict[(train_data[i][0][0], train_data[i][0][1])] = train_data[i][1]

# 计算路径重合度
overlap_results = {}


def trip_length(path):
    return len(path)


def intersections_and_unions(path1, path2):
    intersection_nodes = set()
    for i in range(len(path1) - 1):
        if path1[i] in path2 and path1[i + 1] in path2:
            intersection_nodes.add(path1[i])
            intersection_nodes.add(path1[i + 1])

    union_count = len(set(path1).union(set(path2)))
    return len(intersection_nodes), union_count


# 遍历路径字典计算重合度
for (start, end), true_path in tqdm(true_paths_dict.items(), desc="Processing paths", total=len(true_paths_dict)):
    if (start, end) in shortest_paths_dict and true_path is not None:
        shortest_path = shortest_paths_dict[(start, end)]

        intersection_count, union_count = intersections_and_unions(true_path, shortest_path)

        # 计算重合度（Jaccard指数）
        overlap_ratio = intersection_count / len(true_path) if len(true_path) else 0

        overlap_results[(start, end)] = {
            'true_path': true_path,
            'shortest_path': shortest_path,
            'intersection_count': intersection_count,
            'union_count': union_count,
            'overlap_ratio': overlap_ratio
        }
# 输出结果
count = 0  # 初始化计数器
average_overlap=0

for (start, end), result in overlap_results.items():


    average_overlap+=result['overlap_ratio']

    count += 1  # 满足条件时增加计数

print(average_overlap/count)

# #%%draw
# G_draw = nx.Graph()
# # 添加边到图中
# for (start, end), path in list(true_paths_dict.items())[:20]:
#     for i in range(len(path) - 1):
#         G_draw.add_edge(path[i], path[i + 1])  # 逐个添加边
#
# # 添加 shortest_paths_dict 中的边
# for (start, end), path in list(shortest_paths_dict.items())[:20]:
#     for i in range(len(path) - 1):
#         G_draw.add_edge(path[i], path[i + 1])
#
# # 绘制图形
# pos = nx.spring_layout(G_draw)  # 图的布局
#
# # 绘制 true_paths
# # 绘制 true_paths
# for (start, end), path in list(true_paths_dict.items())[:5]:
#     # 绘制节点
#     for node in path:
#         nx.draw_networkx_nodes(G_draw, pos, nodelist=[node], node_color='blue', node_size=1)
#
#     # 绘制边
#     for i in range(len(path) - 1):
#         nx.draw_networkx_edges(G_draw, pos, edgelist=[(path[i], path[i + 1])], edge_color='blue', width=2)
#
# # 绘制 shortest_paths
# for (start, end), path in list(shortest_paths_dict.items())[:5]:
#     # 绘制节点
#     for node in path:
#         nx.draw_networkx_nodes(G_draw, pos, nodelist=[node], node_color='yellow', node_size=1)
#
#     # 绘制边
#     for i in range(len(path) - 1):
#         nx.draw_networkx_edges(G_draw, pos, edgelist=[(path[i], path[i + 1])], edge_color='yellow', width=2)
#
# # # 添加标签
# # nx.draw_networkx_labels(G_draw, pos)
#
# # 保存图形为 PDF
# plt.title("Directed Graph with True Paths and Shortest Paths")
# plt.axis('off')  # 关闭坐标轴
# plt.savefig("data/chengdu_data/directed_graph.pdf")  # 保存为PDF
# plt.show()