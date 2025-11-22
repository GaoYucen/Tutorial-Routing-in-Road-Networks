#%% 读取chengdu数据集
import geopandas as gpd
import numpy as np

# 读取node和edge数据
node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%% 使用node_df和edge_df构建图
import networkx as nx

G = nx.DiGraph()

for i in range(len(edge_df)):
    u = edge_df.iloc[i]['u']
    v = edge_df.iloc[i]['v']
    G.add_edge(u, v)

#%% 用Node2Vec算法生成节点embedding
from node2vec import Node2Vec
from gensim.models import KeyedVectors

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 保存模型
model.wv.save_word2vec_format('data/chengdu_data/node2vec.emb')

# 加载模型
model = KeyedVectors.load_word2vec_format('data/chengdu_data/node2vec.emb')


#%%
# 得到所有点的嵌入
embeddings = {}
for node in G.nodes():
    embeddings[node] = model[str(node)]

# # 将嵌入转换为numpy数组
# embeddings_array = np.array([embeddings[node] for node in G.nodes()])

# # 打印嵌入结果
# print(embeddings_array)

#%% 保存嵌入结果
# np.save('data/chengdu_data/embeddings.npy', embeddings_array)

# 保存嵌入结果
import pickle

with open('data/chengdu_data/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    f.close()