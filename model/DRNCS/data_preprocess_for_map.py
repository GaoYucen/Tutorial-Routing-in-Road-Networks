#%% map数据
import geopandas as gpd

node_df = gpd.read_file('data/porto_data/map/nodes.shp')
edge_df = gpd.read_file('data/porto_data/map/edges.shp')

#%%
print(node_df.head(5))
print(edge_df.head(5))
print(len(node_df))
print(len(edge_df))
#%% 构造图G
import networkx as nx

G = nx.DiGraph()

for i in range(len(edge_df)):
    G.add_edge(edge_df.iloc[i]['u'], edge_df.iloc[i]['v'])

#%% #%% 计算node_df每个node的后继节点
from collections import defaultdict

node_nbrs = defaultdict(set)

for i in range(len(G.nodes())):
    node = list(G.nodes())[i]
    node_nbrs[node] = set(G.successors(node))

node_nbrs = dict(node_nbrs)

# #%%
# print(G.nodes)
#
# #%% 计算node_df每个node的后继节点
# from collections import defaultdict
#
# node_nbrs = defaultdict(set)
#
# for i in range(len(edge_df)):
#     u = edge_df.iloc[i]['u']
#     v = edge_df.iloc[i]['v']
#     node_nbrs[u].add(v)
#
# node_nbrs = dict(node_nbrs)

#%% 存储node_nbrs
import pickle

with open('data/porto_data/node_nbrs.pkl', 'wb') as f:
    pickle.dump(node_nbrs, f)
    f.close()



