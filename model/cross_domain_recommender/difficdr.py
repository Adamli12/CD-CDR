# -*- coding: utf-8 -*-
# Ref: Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion (NIPS 23' DreamRec)

r"""
DiffICDR
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss, BPRLoss
from recbole.utils import InputType
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender

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

class DiffICDR(CrossDomainRecommender):
    r"""DiffICDR, implemented based on the aggregation mechanism of simplex and diffusion mechanism of DreamRec
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DiffICDR, self).__init__(config, dataset)

        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # Get user history interacted items
        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        #self.margin = config["margin"]
        #self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        #self.reg_weight = config["reg_weight"]
        self.aggregator = config["aggregator"]
        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention or self_attention"
            )
        self.history_len = torch.max(self.history_item_len, dim=0)

        # user embedding matrix
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        ## Edit from DreamRec
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.embedding_size,
        )

        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False, )
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            if self.aggregator == "self_attention":
                self.W_q = nn.Linear(self.embedding_size, 1, bias=False)
        #self.require_pow = config["require_pow"]
        # l2 regularization loss
        #self.reg_loss = EmbLoss()

        self.simple = config['simple']
        self.bprloss = BPRLoss()

        ## Edit from DreamRec
        self.diffuser_type = config["diffuser_type"]
        self.loss_type = config["loss_type"]
        self.timesteps = config['timestep'] # 200, diffusion steps
        self.w = config['uncon_w'] # 2, the weight of conditioned diffusion in inference phase
        self.p = config['uncon_p'] # 0.1, how much prob does train phase use unconditioned diffusion
        self.beta_sche = config['beta_sche'] # exp, the schedule of beta sequence
        self.dropout = config['dropout']
        self.emb_dropout = nn.Dropout(self.dropout)
        layer_norm = config['layer_norm']
        if layer_norm:
            self.ln_1 = nn.LayerNorm(self.embedding_size)
            self.ln_2 = nn.LayerNorm(self.embedding_size)
            self.ln_3 = nn.LayerNorm(self.embedding_size)
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
            self.ln_3 = nn.Identity()
        self.step_mlp = nn.Sequential( # time vector mlp
            SinusoidalPositionEmbeddings(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size*2),
            nn.GELU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
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
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # get the mask
        self.item_emb.weight.data[0, :] = 0

    ## Edit from DreamRec
    def q_sample(self, x_start, t, noise=None): # add noise to x_start according to a series of timestamp t
        # print(self.betas)
        if noise is None:
            #noise = torch.randn_like(x_start)
            noise = torch.randn_like(x_start) / 100
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

    ## Edit from DreamRec
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h): # an inference process, sample for timestep times
        #x = torch.randn_like(h)
        x = torch.randn_like(h) / 100

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=self.device, dtype=torch.long), n)

        return x

    ## Edit from DreamRec
    def denoise_step(self, x, h, step):
        t = self.step_mlp(step)
        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))
        return res
    def denoise_uncon(self, x, step): # with out condition
        h = self.none_embedding(torch.tensor([0], device = self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        res = self.diffu_mlp(torch.cat((x, h, t), dim=1))   
        return res

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
            out = pos_item_sum / (history_len + 1.0e-10).unsqueeze(1)
        elif self.aggregator in ["user_attention", "self_attention"]:
            history_item_e = self.ln_1(history_item_e)
            # [user_num, max_history_len, embedding_size]
            key = self.W_k(history_item_e)
            if self.aggregator == "user_attention":
                # [user_num, max_history_len]
                attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
            elif self.aggregator == "self_attention":
                # [user_num, max_history_len]
                attention = self.W_q(key).squeeze(2)
            e_attention = torch.exp(attention)
            mask = (history_item_e.sum(dim=-1) != 0).int()
            e_attention = e_attention * mask
            # [user_num, max_history_len]
            attention_weight = e_attention / (
                e_attention.sum(dim=1, keepdim=True) + 1.0e-10
            )
            # [user_num, embedding_size]
            out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
            out = self.ln_2(out)
        # Combined vector of user and item sequences
        out = self.UI_map(out)
        out = self.ln_3(out)
        g = self.gamma
        UI_aggregation_e = g * user_e + (1 - g) * out
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

    def add_uncon(self, h):
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - self.p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0], device = self.device)) * (1-mask)
        return h

    def forward(self, user, pos_item, history_item, history_len):
        r"""Get the loss

        Args:
            user (torch.Tensor): User's id, shape: [user_num]
            pos_item (torch.Tensor): Positive item's id, shape: [user_num]
            history_item (torch.Tensor): Id of historty item, shape: [user_num, max_history_len]
            history_len (torch.Tensor): History item's length, shape: [user_num]
            neg_item_seq (torch.Tensor): Negative item seq's id, shape: [user_num, neg_seq_len]

        Returns:
            torch.Tensor: Loss, shape: []
        """
        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        pos_item_e = self.item_emb(pos_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)
        history_item_e = self.emb_dropout(history_item_e)
        # [nuser_num, neg_seq_len, embedding_size]
        #neg_item_seq_e = self.item_emb(neg_item_seq)
        batch_size = user_e.shape[0]

        ## Edit from DreamRec
        x_start = pos_item_e ## Target item e^0
        n = torch.randint(0, self.timesteps, (batch_size, ), device=self.device).long() ## sample random timestemps (int)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        ## Edit from DreamRec
        h = UI_aggregation_e ## History as condition
        h = self.add_uncon(h)

        ## Edit from DreamRec
        noise = torch.randn_like(x_start) 
        x_noisy = self.q_sample(x_start = x_start, t = n, noise = noise)
        predicted_x = self.denoise_step(x = x_noisy, h = h, step = n)
        if self.loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()
        return loss

    def simple_foward(self, user, pos_item, neg_item, history_item, history_len):
        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        pos_item_e = self.item_emb(pos_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)
        #history_item_e = self.emb_dropout(history_item_e)
        # [nuser_num, neg_seq_len, embedding_size]
        neg_item_seq_e = self.item_emb(neg_item)
        batch_size = user_e.shape[0]
        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        pos_item_score, neg_item_score = torch.mul(UI_aggregation_e, pos_item_e).sum(dim=1), torch.mul(UI_aggregation_e, neg_item_seq_e).sum(dim=1)
        #pos_item_score, neg_item_score = torch.mul(user_e, pos_item_e).sum(dim=1), torch.mul(user_e, neg_item_seq_e).sum(dim=1)
        loss = self.bprloss(pos_item_score, neg_item_score)
        return loss

    def calculate_loss(self, interaction):
        r"""Data processing and call function forward(), return loss

        To use DiffICDR, a user must have a historical transaction record,
        a pos item and a sequence of neg items. Based on the RecBole
        framework, the data in the interaction object is ordered, so
        we can get the data quickly.
        """
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # get the sequence of neg items
        #neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        #neg_item_seq = neg_item_seq.T
        user_number = int(len(user) / self.neg_seq_len)
        # user's id
        user = user[0:user_number]
        # historical transaction record
        history_item = self.history_item_id[user]
        # positive item's id
        pos_item = pos_item[0:user_number]
        # history_len
        history_len = self.history_item_len[user]

        if self.simple:
            loss = self.simple_foward(user, pos_item, neg_item, history_item, history_len)
        else:
            loss = self.forward(user, pos_item, history_item, history_len)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]
        test_item = interaction[self.ITEM_ID]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        test_item_e = self.item_emb(test_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        ## Edit from DreamRec
        h = UI_aggregation_e ## History as condition
        x = self.sample(self.denoise_step, self.denoise_uncon, h)

        UI_cos = self.get_cos(x, test_item_e.unsqueeze(1))
        return UI_cos.squeeze(1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        if self.simple:
            x = UI_aggregation_e
            #x = user_e
            all_item_emb = self.item_emb.weight
            II_cos = torch.matmul(x, all_item_emb.T)
            return II_cos
        else:
            ## Edit from DreamRec
            h = UI_aggregation_e ## History as condition
            x = self.sample(self.denoise_step, self.denoise_uncon, h)
            all_item_emb = self.item_emb.weight
            II_cos = torch.matmul(x, all_item_emb.T)
            #return II_cos
        
        all_item_emb = self.item_emb.weight

        #TODO: norm？
        x = F.normalize(x, dim=1)
        all_item_emb = F.normalize(all_item_emb, dim=1)
        II_cos = torch.matmul(x, all_item_emb.T)
        return II_cos
