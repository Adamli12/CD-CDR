# -*- coding: utf-8 -*-
# Ref: Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion (NIPS 23' DreamRec)

r"""
CDCDR
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math

from recbole.model.init import xavier_normal_initialization
from torch.nn.init import xavier_normal_, constant_, normal_
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from CDCDR.model.crossdomain_recommender import CrossDomainRecommender

def diffusion_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Linear layers. For bias in nn.Linear layers, using constant 0 to initialize.
    Use normal_ for embeddings.
    """
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data, 0, 1)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

## Edit from DreamRec
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

## Edit from DreamRec
def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

## Edit from DreamRec
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

## Edit from DreamRec
def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

## Edit from DreamRec
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def build_history_matrix(inter_feat, max_history_len=None):
    """
    构建用户历史交互矩阵，固定使用 'timestamp' 作为值。

    Args:
        inter_feat (pd.DataFrame): 包含 'user_id', 'item_id', 'timestamp' 的交互数据。
        user_num (int): 用户总数（用户 ID 范围为 [0, user_num)）。
        max_history_len (int, optional): 历史序列最大长度。若为 None，则用实际最大长度。

    Returns:
        tuple: (history_matrix, history_values, history_lengths)
            - history_matrix: [user_num, L] 的 item_id 序列（LongTensor）
            - history_values: [user_num, L] 的 timestamp 序列（FloatTensor）
            - history_lengths: [user_num] 每个用户的实际历史长度（LongTensor）
    """
    user_num = 0
    inter_df = []
    for domain in ["source","target"]:#source, target
        uid = inter_feat[domain + "_user_id"]      # int64
        iid = inter_feat[domain + "_item_id"]      # int64
        ts  = inter_feat[domain + "_timestamp"]    # float32 (or can be cast to)

        df = pd.DataFrame({
            'user_id': uid.cpu().numpy(),
            'item_id': iid.cpu().numpy(),
            'timestamp': ts.cpu().numpy().astype('float32')  # 显式 float32
        })

        # 排序并保留最近的 max_history_len 条
        df = df.sort_values(['user_id', 'timestamp'])
        if max_history_len is not None:
            df = df.groupby('user_id').tail(max_history_len)

        temp_user_num = int(df['user_id'].max()) + 1 if len(df) > 0 else 1
        user_num = max(user_num, temp_user_num)
        inter_df.append(df)

    Hs = []
    Vs = []
    lens_list = []
    for df in inter_df:
        L = max_history_len
        L = max(1, L)

        # 初始化输出张量
        H = torch.zeros(user_num, L, dtype=torch.int64)
        V = torch.zeros(user_num, L, dtype=torch.float32)  # float32!
        lens = torch.zeros(user_num, dtype=torch.int64)

        # 批量填充：按用户分组，整片赋值（避免逐元素）
        for user, group in df.groupby('user_id'):
            k = min(len(group), L)
            # 关键：用 torch.from_numpy + .to(torch.float32) 确保类型匹配
            H[user, :k] = torch.from_numpy(group['item_id'].values[:k])
            V[user, :k] = torch.from_numpy(group['timestamp'].values[:k]).to(torch.float32)
            lens[user] = k
        Hs.append(H)
        Vs.append(V)
        lens_list.append(lens)

    return torch.stack(Hs, dim=0), torch.stack(Vs, dim=0), torch.stack(lens_list, dim=0)

def temporal_mask_history(history_item, history_timestamp, history_len, current_timestamp):
    """
    Args:
        history_item:       [B, L] int64
        history_timestamp:  [B, L] float32
        history_len:        [B]    int64
        current_timestamp:  [B]    float32  (当前交互的时间戳)

    Returns:
        masked_item: [B, L] int64  (未来部分置0)
        new_len:     [B]    int64  (裁剪后的真实长度)
    """
    B, L = history_item.shape

    valid_mask = torch.arange(L, device=history_item.device).unsqueeze(0) < history_len.unsqueeze(1)  # [B, L]
    time_mask = history_timestamp < current_timestamp.unsqueeze(1)  # [B, L]
    final_mask = valid_mask & time_mask  # [B, L]
    new_len = final_mask.sum(dim=1)  # [B]
    masked_item = history_item * final_mask  # [B, L]

    return masked_item, new_len

def remove_pos_from_history(pos_item, history_item, history_len):
    """
    从历史记录中移除正样本项（pos_item），并更新历史长度。
    
    Args:
        pos_item: [B]
        history_item: [B, L]
        history_len: [B]
    
    Returns:
        filtered_history_item: [B, L]（移除 pos_item 后，用 0 填充）
        updated_history_len: [B]
    """
    pos_item_expanded = pos_item.unsqueeze(1)  # [B, 1]
    mask = (history_item != pos_item_expanded)  # [B, L]
    remove_counts = (~mask).sum(dim=1)  # [B]
    updated_history_len = history_len - remove_counts
    filtered_history_item = history_item * mask.long()
    return filtered_history_item, updated_history_len

def gather_indexes(output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)

## Edit from DreamRec
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CDCDR(CrossDomainRecommender):
    r"""CDCDR, implemented based on the aggregation mechanism of simplex and diffusion mechanism of DreamRec
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CDCDR, self).__init__(config, dataset)

        # Get user history interacted items
        #self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(max_history_len=config["history_len"])
        both_domain_inter_feat = {
            "source_user_id": dataset.source_domain_dataset.inter_feat["source_user_id"],
            "source_item_id": dataset.source_domain_dataset.inter_feat["source_item_id"],
            "source_timestamp": dataset.source_domain_dataset.inter_feat["source_timestamp"],
            "target_user_id": dataset.target_domain_dataset.inter_feat["target_user_id"],
            "target_item_id": dataset.target_domain_dataset.inter_feat["target_item_id"],
            "target_timestamp": dataset.target_domain_dataset.inter_feat["target_timestamp"]}
        history_item_id, history_timestamp, history_item_len = build_history_matrix(
            both_domain_inter_feat,
            max_history_len=config["history_len"]
        )

        # 注册为 buffer，确保随模型移动到 GPU
        self.register_buffer("history_item_id", history_item_id)
        self.register_buffer("history_timestamp", history_timestamp)
        self.register_buffer("history_item_len", history_item_len)
        self.history_len = [torch.max(self.history_item_len[0], dim=0), torch.max(self.history_item_len[1], dim=0)]  # source, target

        # load parameters info
        self.embedding_size = config["embedding_size"]
        #self.margin = config["margin"]
        #self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        #self.reg_weight = config["reg_weight"]
        self.aggregator = config["aggregator"]
        self.loss_n = config["loss_n"]
        if self.aggregator not in ["mean", "user_attention", "self_attention", "transformer"]:
            raise ValueError(
                "aggregator must be mean, user_attention, self_attention or transformer"
            )
        # Transformer聚合器特有参数
        if self.aggregator == "transformer":
            self.n_heads = config["n_heads"]
            self.dropout_prob = config["dropout"]
            # 位置编码
            #self.position_embedding = nn.Embedding(config["history_len"], self.embedding_size)
            # Transformer编码层
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=self.n_heads,
                #dim_feedforward=self.embedding_size * 4,
                dim_feedforward=self.embedding_size,
                dropout=self.dropout_prob,
                activation='gelu',
                batch_first=True
            )

        # user embedding matrix
        self.user_emb = nn.Embedding(self.total_num_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.total_num_items, self.embedding_size, padding_idx=0)
        ## Edit from DreamRec
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.embedding_size,
        )

        # Domain condition generator
        self.domain_emb = nn.Embedding(2, embedding_dim=self.embedding_size)
        self.domain_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=self.n_heads,
            batch_first=True
        )
        self.gate_proj = nn.Linear(2 * self.embedding_size, self.embedding_size)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False, )
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_size, 1, bias=False)
        elif self.aggregator == "transformer":
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=self.transformer_layer,
                num_layers=config["n_layers"]
            )
            self.UI_map = nn.Identity()
            self.mean_pooling = config["mean_pooling"]

        self.simple = config['simple']
        self.sigmoid = nn.Sigmoid()
        self.bprloss = BPRLoss()
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()

        ## Edit from DreamRec
        self.diffuser_type = config["diffuser_type"]
        self.loss_type = config["loss_type"]
        self.timesteps = config['timestep'] # 200, diffusion steps
        self.linespace = 100
        self.w = config['uncon_w'] # 2, the weight of conditioned diffusion in inference phase
        self.p = config['uncon_p'] # 0.1, how much prob does train phase use unconditioned diffusion
        self.beta_sche = config['beta_sche'] # exp, the schedule of beta sequence
        self.dropout = config['dropout']
        self.emb_dropout = nn.Dropout(self.dropout)
        layer_norm = config['layer_norm']
        if layer_norm:
            self.ln_1 = nn.LayerNorm(self.embedding_size)
            self.ln_2 = nn.LayerNorm(self.embedding_size)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
        #TimestepEmbedder slightly difference
        self.step_mlp = nn.Sequential( # time vector mlp
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, self.embedding_size),
            nn.GELU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        if self.diffuser_type =='mlp1':
            self.diffu_mlp = nn.Sequential( # diffusion mlp, in: x,c,t; out: x
                nn.Linear(self.embedding_size*3, self.embedding_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffu_mlp = nn.Sequential(
            nn.Linear(self.embedding_size * 3, self.embedding_size*2),
            nn.GELU(),
            nn.Linear(self.embedding_size*2, self.embedding_size)
        )
        self.beta_start = 0.0001
        self.beta_end = 0.02
        if self.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif self.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()
        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # DDIM Reverse Process
        indices = list(range(0, self.timesteps+1, self.linespace)) # [0,100,...,2000]
        self.sub_timesteps = len(indices)
        indices_now = [indices[i]-1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)
        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(1.-self.alphas_cumprod_ddim_prev)/ self.sqrt_recipm1_alphas_cumprod_ddim
        self.posterior_ddim_coef2 = torch.sqrt(1.-self.alphas_cumprod_ddim_prev) / torch.sqrt(1. - self.alphas_cumprod_ddim)

        # parameters initialization
        if self.simple:
            self.apply(xavier_normal_initialization)
        else:
            self.apply(diffusion_initialization)
        # get the mask
        self.item_emb.weight.data[0, :] = 0

    ## Edit from DreamRec
    # pertube_item, pertube
    def q_sample(self, x_start, t, noise=None): # add noise to x_start according to a series of timestamp t
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            #noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    ## Edit from DreamRec
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index): # inference one step: denoising from x generating x_start
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # DreamRec_backbone i_sample
    @torch.no_grad()
    def i_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        # cf guidance

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t) 
        
        x_t = x 
        model_mean = (
            self.posterior_ddim_coef1[t_index] * x_start +
            self.posterior_ddim_coef2[t_index] * x_t
        )

        return model_mean 
        
    # DreamRec_backbone sample_from_noise
    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_uncon, h):

        x = torch.randn_like(h)

        #for n in reversed(range(0, self.timesteps, self.linespace)):
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0], ), n*self.linespace, device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_uncon, x, h, step, n)

        return x

    ## Edit from DreamRec
    # DreamRec_backbone forward
    def denoise_step(self, x, h, step):
        t = self.step_mlp(step)
        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        return res
    # DreamRec_backbone forward_uncon
    def denoise_uncon(self, x, step): # with out condition
        h = self.none_embedding(torch.tensor([0], device = self.device))
        h = torch.cat([h.view(1, self.embedding_size)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))   
        return res

    def get_user_representation(self, user, history_item, history_len):
        """
        计算用户聚合表征 UI_aggregation_e。
        
        Args:
            user: [B]
            history_item: [B, L]
            history_len: [B]
        
        Returns:
            UI_aggregation_e: [B, D]
        """
        user_e = self.user_emb(user)  # [B, D]
        history_item_e = self.item_emb(history_item)  # [B, L, D]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)  # [B, D]
        return UI_aggregation_e

    def get_UI_aggregation(self, user_e, history_item_e, history_len):
        r"""Get the combined vector of user and historically interacted items

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            history_item_e (torch.Tensor): History item's feature vector,
                shape: [user_num, max_history_len, embedding_size]
            history_len (torch.Tensor): User's history length, shape: [user_num]

        Returns:
            torch.Tensor: Combined vector of user and item sequences, shape: [user_num, embedding_size]
        """
        if self.aggregator == "mean":
            pos_item_sum = history_item_e.sum(dim=1)
            # [user_num, embedding_size]
            out = pos_item_sum / (history_len + 1.e-10).unsqueeze(1)
        elif self.aggregator in ["user_attention", "self_attention"]: #residual？
            mask = (torch.abs(history_item_e).sum(dim=-1) > 1e-8).int()
            history_item_e = self.ln_1(history_item_e) # [user_num, max_history_len, embedding_size]
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2) # [user_num, max_history_len]
            elif self.aggregator == "self_attention":
                attention = self.W_q(key).squeeze(2)
            attention_stable = attention - attention.max(dim=1, keepdim=True).values
            e_attention = torch.exp(attention_stable)
            e_attention = e_attention * mask

            attention_weight = e_attention / (e_attention.sum(dim=1, keepdim=True) + 1.0e-10)
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
            #out = self.ln_2(out) #TODO:在后面加LN
        elif self.aggregator == "transformer":
            mask = torch.abs(history_item_e).sum(dim=-1) < 1e-8  # 填充位置为True
            fully_padded = mask.all(dim=1)
            mask[fully_padded, 0] = False
            transformer_input = history_item_e
            """
            # 生成位置编码
            positions = torch.arange(history_item_e.size(1), device=self.device).expand(history_item_e.size(0), -1)
            pos_embedding = self.position_embedding(positions)
            
            # 添加位置编码并应用Transformer
            transformer_input = transformer_input + pos_embedding
            """
            transformer_input = self.ln_1(transformer_input)
            transformer_input = self.emb_dropout(transformer_input)
            transformer_output = self.transformer_encoder(
                transformer_input,
                src_key_padding_mask=mask
            )
            if self.mean_pooling:
                # 生成有效位置mask并计算均值
                mask = ~mask.unsqueeze(-1)  # 有效位置为True [B, L, 1]
                masked_output = transformer_output * mask.float()       # 将padding位置置零
                sum_output = masked_output.sum(dim=1)                   # 按序列维度求和 [B, D]
                valid_length = history_len.unsqueeze(1).float()         # 有效序列长度 [B, 1]
                out = sum_output / (valid_length + 1e-10)               # 平均池化
            else:
                # 找到每个序列中最后一个True的位置（即最后一个有效项的位置）
                last_valid_pos = (mask.size(1) - 1) - mask.flip(dims=[1]).int().argmin(dim=1)  # [batch_size]
                # 处理全为True的情况（无有效项）
                last_valid_pos[mask.all(dim=1)] = 0
                out = gather_indexes(transformer_output, last_valid_pos)
        # Combined vector of user and item sequences
        out = self.UI_map(out)
        UI_aggregation_e = self.gamma * user_e + (1 - self.gamma) * out #simplex也是out的绝对值0.5，user的绝对值0.0294，试试g=0试试
        #UI_aggregation_e = self.ln_2(UI_aggregation_e)
        return UI_aggregation_e

    def get_cos(self, user_e, item_e):
        r"""Get the cosine similarity between user and item

        Args:
            user_e (torch.Tensor): User's feature vector, shape: [user_num, embedding_size]
            item_e (torch.Tensor): Item's feature vector,
                shape: [user_num, item_num, embedding_size]

        Returns:
            torch.Tensor: Cosine similarity between user and item, shape: [user_num, item_num]
        """
        user_e = F.normalize(user_e, dim=1)
        # [user_num, embedding_size, 1]
        user_e = user_e.unsqueeze(2)
        item_e = F.normalize(item_e, dim=2)
        UI_cos = torch.matmul(item_e, user_e)
        return UI_cos.squeeze(2)

    def run_diffusion_process(self, target_item_emb, condition_emb):
        """
        执行扩散模型的前向加噪与去噪预测。
        
        Args:
            target_item_emb: x_start, [B, D]
            condition_emb: h (condition), [B, D]
        
        Returns:
            x_start: 原始目标嵌入 [B, D]
            predicted_x: 模型预测的去噪结果 [B, D]
        """
        B = target_item_emb.shape[0]
        n = torch.randint(0, self.timesteps, (B,), device=self.device).long()  # [B]
        noise = torch.randn_like(target_item_emb)  # [B, D]
        
        # 加噪
        x_noisy = self.q_sample(x_start=target_item_emb, t=n, noise=noise)  # [B, D]
        
        # 条件注入（如 DreamRec 中的 unconditional mixing）
        h = self.add_uncon(condition_emb)  # [B, D]
        
        # 去噪预测
        predicted_x = self.denoise_step(x=x_noisy, h=h, step=n)  # [B, D]
        
        return target_item_emb, predicted_x

    def add_uncon(self, h):
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - self.p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0], device = self.device)) * (1-mask)
        return h

    def domain_condition_generator(self, UI_aggregation_e, domain):
        """
        使用可学习的 domain embedding 作为 Q，UI_aggregation_e 作为 K=V，
        执行 batch-first MultiheadAttention。
        
        Args:
            UI_aggregation_e: [B, D]
            domain: [B,]  (e.g., 0 or 1)
        
        Returns:
            [B, D] — 注意力输出
        """
        """if 1:
            output = UI_aggregation_e.unsqueeze(1) * domain  # [B, 1, D]
        else:"""
        if 1:
            Q = self.domain_emb(domain).unsqueeze(1)      # [B, 1, D]
            K = V = UI_aggregation_e.unsqueeze(1)         # [B, 1, D]
            output, _ = self.domain_attn(Q, K, V)         # [B, 1, D]
        return output.squeeze(1)                      # [B, D]

    def calculate_loss(self, interaction):
        r"""Data processing and call function forward(), return loss

        To use CDCDR, a user must have a historical transaction record,
        a pos item and a sequence of neg items. Based on the RecBole
        framework, the data in the interaction object is ordered, so
        we can get the data quickly.
        """

        user_id = [interaction[self.SOURCE_USER_ID],interaction[self.TARGET_USER_ID]]#source, target
        item_id = [interaction[self.SOURCE_ITEM_ID],interaction[self.TARGET_ITEM_ID]]#source, target
        #label = [interaction[self.SOURCE_LABEL],interaction[self.TARGET_LABEL]]#source, target
        predict = [0,0]
        loss_all = 0
        neg_item_id = [interaction[self.SOURCE_NEG_ITEM_ID],interaction[self.TARGET_NEG_ITEM_ID]]
        neg_predict = [0,0]
        losses = []
        for domain in [0,1]:#source, target
            user = user_id[domain]
            pos_item = item_id[domain]
            neg_item = neg_item_id[domain]
            neg_item_seq = neg_item.reshape((self.neg_seq_len, -1)).T # [B, neg_seq_len]
            user_number = int(len(user) / self.neg_seq_len)
            user = user[0:user_number] # [B]
            pos_item = pos_item[0:user_number] # [B]

            s_history_item = self.history_item_id[0][user]      # [B, L]
            s_history_len = self.history_item_len[0][user]      # [B]
            # Remove pos_item from history
            s_history_item, s_history_len = remove_pos_from_history(pos_item, s_history_item, s_history_len)

            t_history_item = self.history_item_id[1][user]      # [B, L]
            t_history_len = self.history_item_len[1][user]      # [B]
            # Remove pos_item from other history
            t_history_item, t_history_len = remove_pos_from_history(pos_item, t_history_item, t_history_len)

            """# 动态裁剪：移除 timestamp >= 当前时间的历史
            t_timestamp = interaction['target_timestamp'] # [B * neg_seq_len]
            t_timestamp = t_timestamp[:user_number]  # [B] ← 关键：取每个用户的当前时间戳
            t_history_ts = self.history_timestamp[1][user]      # [B, L]
            t_history_item, t_history_len = temporal_mask_history(
                t_history_item, t_history_ts, t_history_len, t_timestamp
            )"""

            # Aggregated Domain-Specific Interactions
            s_UI_aggregation_e = self.get_user_representation(user, s_history_item, s_history_len) # [B, D]
            t_UI_aggregation_e = self.get_user_representation(user, t_history_item, t_history_len) # [B, D]
            pos_item_e = self.item_emb(pos_item)  # [B, D]

            """# Domain condition generator
            gate = torch.sigmoid(self.gate_proj(torch.cat([s_UI_aggregation_e, t_UI_aggregation_e], dim=-1)))
            UI_aggregation_e = gate * s_UI_aggregation_e + (1 - gate) * t_UI_aggregation_e  # [B, D]
            domain_idx = torch.full_like(user, domain)  # [B], e.g., all 0 for source
            UI_aggregation_e = self.domain_condition_generator(UI_aggregation_e, domain_idx)  # [B, D]"""
            UI_aggregation_e = t_UI_aggregation_e if domain ==1 else s_UI_aggregation_e

            if self.simple:
                neg_item_seq_e = self.item_emb(neg_item_seq)  # [B, neg_seq_len, D]
                pos_item_score = torch.mul(UI_aggregation_e.unsqueeze(1), pos_item_e.unsqueeze(1)).sum(dim=2)
                neg_item_score = torch.mul(UI_aggregation_e.unsqueeze(1), neg_item_seq_e).sum(dim=2)
                
                if self.loss_n == 'bce':
                    pos_label = torch.ones_like(pos_item_score)
                    neg_label = torch.zeros_like(neg_item_score)
                    loss = self.bceloss(self.sigmoid(pos_item_score), pos_label) + \
                        self.bceloss(self.sigmoid(neg_item_score), neg_label)
                elif self.loss_n == 'bpr':
                    loss = self.bprloss(pos_item_score, neg_item_score)
                elif self.loss_n == 'mse':
                    pos_label = torch.ones_like(pos_item_score)
                    neg_label = torch.zeros_like(neg_item_score)
                    loss = self.mseloss(pos_item_score, pos_label) + \
                        self.mseloss(neg_item_score, neg_label)
                elif self.loss_n == 'ccl':
                    pos_cos = self.get_cos(UI_aggregation_e, pos_item_e.unsqueeze(1))
                    neg_cos = self.get_cos(UI_aggregation_e, neg_item_seq_e)
                    pos_loss = torch.relu(1 - pos_cos)
                    #margin: 0.5, negative_weight: 10
                    neg_loss = torch.relu(neg_cos - 0.5) * 10
                    loss = pos_loss.mean() + neg_loss.mean()
            else:
                x_start, predicted_x = self.run_diffusion_process(pos_item_e, UI_aggregation_e)
                if self.loss_type == 'l1':
                    loss = F.l1_loss(x_start, predicted_x)
                elif self.loss_type == 'l2':
                    loss = F.mse_loss(x_start, predicted_x)
                elif self.loss_type == "huber":
                    loss = F.smooth_l1_loss(x_start, predicted_x)
                else:
                    raise NotImplementedError()
            losses.append(loss)
        return_loss = 0.2 * losses[0] + 0.8 * losses[1]
        return return_loss

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        history_item = self.history_item_id[1][user]      # [B, L]
        history_len = self.history_item_len[1][user]      # [B]

        """# 动态裁剪：移除 timestamp >= 当前时间的历史
        timestamp = interaction['timestamp']  # [B] ← 关键：取每个用户的当前时间戳
        history_ts = self.history_timestamp[user]      # [B, L]
        history_item, history_len = temporal_mask_history(
            history_item, history_ts, history_len, timestamp
        )"""

        UI_aggregation_e = self.get_user_representation(user, history_item, history_len)
        h = UI_aggregation_e
        x = self.sample_from_noise(self.denoise_step, self.denoise_uncon, h)

        test_item = interaction[self.TARGET_ITEM_ID]
        UI_cos = self.get_cos(x, self.item_emb(test_item).unsqueeze(1))
        return UI_cos.squeeze(1)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        history_item = self.history_item_id[1][user]      # [B, L]
        history_len = self.history_item_len[1][user]      # [B]

        """# 动态裁剪：移除 timestamp >= 当前时间的历史
        timestamp = interaction['timestamp']  # [B] ← 关键：取每个用户的当前时间戳
        history_ts = self.history_timestamp[user]      # [B, L]
        history_item, history_len = temporal_mask_history(
            history_item, history_ts, history_len, timestamp
        )"""

        UI_aggregation_e = self.get_user_representation(user, history_item, history_len)
        if self.simple:
            x = UI_aggregation_e
        else:
            ## Edit from DreamRec
            h = UI_aggregation_e ## History as condition
            x = self.sample_from_noise(self.denoise_step, self.denoise_uncon, h)

        all_item_emb = self.item_emb.weight
        target_all_item_emb = all_item_emb[:self.target_num_items]
        II_cos = torch.matmul(x, target_all_item_emb.T)
        return II_cos
        
        #TODO: 要不要norm？
        x = F.normalize(x, dim=1)
        target_all_item_emb = F.normalize(target_all_item_emb, dim=1)
        II_cos = torch.matmul(x, target_all_item_emb.T)
        return II_cos