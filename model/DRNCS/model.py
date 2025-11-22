import torch
import torch.nn as nn

#%% 定义一个MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = []
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self, embedding=None):
        super(Model, self).__init__()
        self.embedding = embedding
        self.embedding_len = len(self.embedding[list(self.embedding.keys())[0]])
        self.mlp_model = MLP(input_dim = 3 * self.embedding_len, output_dim=1, num_layers=3, hidden_dim=128)

    def forward(self, input_embed):
        output_embed = self.mlp_model(input_embed)
        return output_embed

class Model_WD(nn.Module):
    def __init__(self, node_list_len=None, embedding_len=None, param_path=None):
        super(Model_WD, self).__init__()
        self.mlp_s = MLP(input_dim = node_list_len, output_dim=embedding_len, num_layers=1, hidden_dim=128)
        self.mlp_d = MLP(input_dim = node_list_len, output_dim=embedding_len, num_layers=1, hidden_dim=128)

        self.mlp_encoder = MLP(input_dim = 2 * embedding_len, output_dim=embedding_len, num_layers=2, hidden_dim=128)
        self.mlp_c = MLP(input_dim = node_list_len, output_dim=embedding_len, num_layers=1, hidden_dim=128)
        if param_path is not None:
            try:
                self.mlp_s.load_state_dict(torch.load(param_path))
                self.mlp_d.load_state_dict(torch.load(param_path))
                self.mlp_c.load_state_dict(torch.load(param_path))
            except Exception as e:
                print(f"Error loading model parameters: {e}")

    def forward(self, s_embed, d_embed, c_list_embed, label):
        s_embed_l1 = self.mlp_s(s_embed)
        d_embed_l1 = self.mlp_d(d_embed)
        sd_embed_l1 = torch.cat((s_embed_l1, d_embed_l1), dim=1)
        sd_embed_l2 = self.mlp_encoder(sd_embed_l1)
        c_list_embed_l1 = self.mlp_c(c_list_embed)
        # 对sd_embed_l2的shape是[embedding_len], c_list_embed_l1的shape是[num, embedding_len]，如何对sd_embed_l2及c_list_embed_l1中的每个元素进行点积并sigmoid后得到shape为[num, 1]的tensor
        output_embed = torch.sigmoid(torch.matmul(c_list_embed_l1, sd_embed_l2.t()))
        return output_embed