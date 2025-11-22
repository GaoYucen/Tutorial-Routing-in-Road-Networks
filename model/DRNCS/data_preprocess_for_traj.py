# %% 读取轨迹数据
import pickle
import tqdm
# %% 边序列转化为点osmid序列
import geopandas as gpd

# node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/harbin_data/map/edges.shp')

def edge2osmid(edges_seq):
    osmids = []
    for edge_idx in edges_seq:
        osmids.append(edge_df.iloc[edge_idx]['u'])
    osmids.append(edge_df.iloc[edges_seq[-1]]['v'])
    return osmids

# name_list = ['train', 'test', 'validation']
name_list = ['train']
for name in name_list:

    with open('data/harbin_data/preprocessed_'+name+'_trips_all_partial_5.0.pkl', 'rb') as f:
        train_data_origin = pickle.load(f)
        f.close()

    train_data = []
    for idx, edges_seq, timestamps in tqdm.tqdm(train_data_origin, desc="Processing"):
        train_data.append((idx, edge2osmid(edges_seq), timestamps))

    # %% 保存处理后的数据
    with open('data/harbin_data/preprocessed_'+name+'_trips_small_osmid.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        f.close()