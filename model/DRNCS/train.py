#%% 读取轨迹数据
import pickle
import random

with open('data/chengdu_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

# with open('data/chengdu_data/preprocessed_validation_trips_small_osmid.pkl', 'rb') as f:
#     val_data = pickle.load(f)
#     f.close()
#
# with open('data/chengdu_data/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
#     test_data = pickle.load(f)
#     f.close()

# 构建数据集
for i in range(len(train_data)):
    train_data[i] = ([train_data[i][1][0], train_data[i][1][-1], train_data[i][2][0]], train_data[i][1])

# for i in range(len(val_data)):
#     val_data[i] = ([val_data[i][1][0], val_data[i][1][-1], val_data[i][2][0]], val_data[i][1])
#
# for i in range(len(test_data)):
#     test_data[i] = ([test_data[i][1][0], test_data[i][1][-1], test_data[i][2][0]], test_data[i][1])

# #%% 构造dataloader → 需要size相同
# import torch
# from torch.utils.data import DataLoader
#
# train_dataset = DataLoader(train_data, batch_size=32, shuffle=True)
# val_dataset = DataLoader(val_data, batch_size=32, shuffle=True)
# test_dataset = DataLoader(test_data, batch_size=32, shuffle=True)

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

# #%% 读取config中定义的参数
# # 将当前目录加上/code添加到目录中
# import os
# import sys
# sys.path.append(os.getcwd() + '/code')
# import config
#
# params, _ = config.get_config()

#%% 训练
num_epoches =50
batch_size = 512

from tqdm import tqdm
import torch
from model import Model
import random

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = Model(embedding=node_embeddings).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(num_epoches)):
    random.shuffle(train_data)
    for i in range(0, len(train_data), batch_size):
        optimizer.zero_grad()
        batch = [item[1] for item in train_data[i:i + batch_size]]
        source = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        source_array = np.array([node_embeddings[node] for node in source])
        source_embed = torch.from_numpy(source_array).to(device)
        dest_array = np.array([node_embeddings[node] for node in dest])
        dest_embed = torch.from_numpy(dest_array).to(device)
        nbr_array = np.array([node_embeddings[node] for node in nbr])
        nbr_embed = torch.from_numpy(nbr_array).to(device)
        input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
        pred = model(input_embed)
        # 构造mask矩阵
        mask = torch.tensor([1 if nbr[i] != -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
        # 将pred中对应nbr==-1的部分置为0
        pred = pred * mask
        target = torch.tensor([node_nbrs[item[j]].index(item[j + 1]) for item in batch for j in range(len(item) - 1)]).to(device)
        loss = torch.nn.functional.cross_entropy(pred.view(-1, max_nbrs), target)
        loss.backward()
        optimizer.step()
    print('loss:', loss)
#存储模型数据
torch.save(model.state_dict(), 'param/model.pth')









