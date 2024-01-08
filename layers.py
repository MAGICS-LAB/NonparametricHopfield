import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math
from functools import partial

from math import sqrt
from utils.sparse_max import Sparsemax
from utils.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from utils.diagonaled import *
from utils.favor_utils import *

from functools import partial

import torch
from torch.nn import Module
from utils.feature_map import elu_feature_map
from utils.feature_map import *
from utils.entmax import Entmax15


class LinearHopfieldCore(nn.Module):
    def __init__(self, attention_dropout=0.0, eps=1e-6, norm=False):
        super(LinearHopfieldCore, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.eps = eps
        self.norm = norm

    def forward(self, queries, keys, values, mask=None):

        B, L, H, D = queries.shape
        _, S, _, D = values.shape

        if mask is not None:
            mask = ~mask[:, :, None, None].bool()
            keys = keys.masked_fill(mask, 0.0)

        Q = queries.contiguous().permute(0, 2, 1, 3)
        K = keys.contiguous().permute(0, 2, 1, 3)
        V = values.contiguous().permute(0, 2, 1, 3)

        KV = torch.einsum('...sd,...se->...de', K, V)
        Z = 1.0 / torch.einsum('...sd,...d->...s', Q, K.sum(dim=-2)+self.eps)
        V_new = torch.einsum('...de,...sd,...s->...se', KV, Q, Z)
        return V_new.contiguous().permute(0, 2, 1, 3)

class SparseStructured_HopfieldCore(nn.Module):
    '''
    The Attention operation with Mask
    '''
    def __init__(self, scale=None, mask_type='rand', rand_prob=0.2, win_size=2, dial_size=1, norm=False):
        super(SparseStructured_HopfieldCore, self).__init__()

        self.norm = norm
        self.scale = scale
        self.mask_type = mask_type
        self.rand_prob = rand_prob
        self.win_size = win_size
        self.dial_size = dial_size
        assert self.mask_type in ['rand', 'window', 'tvm', 'topk', 'rand_fast']
        if self.mask_type in ['window']:
            assert self.dial_size == 1
        self.softmax = nn.Softmax(dim=-1)

    def window_mask_attn(self, queries, keys, values, mask=None):

        attn_weights = sliding_chunks_matmul_qk(queries, keys, self.win_size, 0)
        mask_invalid_locations(attn_weights, self.win_size, self.dial_size, False)
        attn_weights_float = self.softmax(attn_weights)  # use fp32 for numerical stability
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights_float.type_as(attn_weights)
        # values = values.view(queries.size(1), queries(0), self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        attn += sliding_chunks_matmul_pv(attn_probs, values, self.win_size)
        attn = attn.transpose(0, 1).reshape(queries.size(1), queries.size(0), -1).contiguous()
        return attn

    def fast_longformer_attn(self, queries, keys, values, mask=None):

        attn_weights = diagonaled_mm(queries, keys, self.win_size, self.dial_size, False, 0, False)
        mask_invalid_locations(attn_weights, self.win_size, self.dial_size, False)
        attn_weights_float = self.softmax(attn_weights, dim=-1)  # use fp32 for numerical stability
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        # values = values.view(queries.size(1), queries(0), self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        values = values.float().contiguous()
        attn += diagonaled_mm(attn_probs, values, self.win_size, self.dial_size, True, 0, False)
        attn = attn.transpose(0, 1).reshape(queries.size(1), queries.size(0), -1).contiguous()
        return attn
    
    def top_k_attn(self, queries, keys, values, mask=None):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        topk, indices = torch.topk(scores, k=int(self.rand_prob*S), dim=-1, sorted=False)

        res = torch.autograd.Variable(torch.zeros_like(scores)).to(scores.device)
        scores = res.scatter(-1, indices, topk)

        A = self.softmax(scale * scores)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()

    def rand_masked_attn(self, queries, keys, values, mask=None):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.softmax(F.dropout(scale * scores, self.rand_prob, training=True))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

    def rand_masked_attn_fast(self, queries, keys, values, mask=None):

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, -1, S)

        mask_loc = F.dropout(torch.ones(L, S).to(queries.device), self.rand_prob, training=True).to_sparse_csr()

        scores = torch.sparse.sampled_addmm(mask_loc, queries, keys).to_dense()
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.softmax(scale * scores)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

    def forward(self, queries, keys, values, mask=None):

        if self.mask_type == 'rand':
            return self.rand_masked_attn(queries, keys, values, mask)
        if self.mask_type == 'rand_fast':
            return self.rand_masked_attn_fast(queries, keys, values, mask)
        elif self.mask_type == 'window':
            return self.window_mask_attn(queries, keys, values, mask)
        elif self.mask_type == 'tvm':
            return self.fast_longformer_attn(queries, keys, values, mask)
        elif self.mask_type == 'topk':
            return self.top_k_attn(queries, keys, values)

        # attn_output_weights.masked_fill_(attn_mask, float('-inf'))

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.0):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            dropout=0.1,
            scale=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model
        self.inner_attention = FullAttention(
            scale=scale, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, inputs):

        queries = inputs
        keys = inputs
        values = inputs

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)
        out = out.mean(1)

        return self.out_projection(out)

class HopfieldCore(nn.Module):
    '''
    The Hopfield operation
    '''

    def __init__(self, scale=None, attention_dropout=0.0, mode='sparsemax', norm=False):
        super(HopfieldCore, self).__init__()
        self.scale = scale
        self.norm = norm
        self.dropout = nn.Dropout(attention_dropout)
        if mode == 'sparsemax':
            self.softmax = Sparsemax(dim=-1)
        elif mode == 'entmax':
            self.softmax = Entmax15(dim=-1)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class Hopfield(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            update_steps=1,
            dropout=0.1,
            mode='softmax',
            scale=None):
        super(Hopfield, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = HopfieldCore(
            scale=scale, attention_dropout=dropout, mode=mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y, mask=None):

        B, L, _ = R.shape
        _, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for i in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)

class SparseStructured_Hopfield(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            update_steps=1,
            prob=0.1,
            mode='topk',
            scale=None,
            norm=False):
        super(SparseStructured_Hopfield, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model
        self.norm = norm

        if mode in ["topk", "rand", "window"]:
            self.inner_attention = SparseStructured_HopfieldCore(
                scale=scale, mask_type=mode, rand_prob=prob)
            
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y, mask=None):

        B, L, _ = R.shape
        _, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)

class HopfieldPooling(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            num_pattern=1,
            update_steps=1,
            dropout=0.1,
            mode='softmax',
            scale=None):
        super(HopfieldPooling, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = HopfieldCore(
            scale=scale, attention_dropout=dropout, mode=mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

        pooling_weight_size = d_model

        self.query = nn.Parameter(
            torch.randn(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    d_model if pooling_weight_size is None else pooling_weight_size), dtype=torch.float32),
            requires_grad=True)

    def forward(self, Y, mask=None):

        # queries : state pattern
        # keys : store pattern
        # values : should just be keys
        B, L, _ = self.query.shape
        B, S, _ = Y.shape
        H = self.n_heads
        q = self.query.repeat((*((B, 1)), 1))

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)
        return self.out_projection(out)

class HopfieldLayer(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=False,
            update_steps=1,
            dropout=0.0,
            mode='softmax',
            scale=None,
            prob=0.5,
            favor_mode="gaussian",
            kernel_fn="relu"):
        super(HopfieldLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

        if mode in ["sparsemax", "softmax"]:
            self.inner_attention = HopfieldCore(
                scale=scale, attention_dropout=0.0, mode=mode, norm=True)
        elif mode in ["rand", "topk", "window"]:
            self.inner_attention = SparseStructured_HopfieldCore(
                scale=scale, mask_type=mode, rand_prob=prob, norm=True)
        elif mode in ["favor", "linear"]:
            self.inner_attention = FastAttention(
                d_model, generalized_attention = (favor_mode == "gaussian"), kernel_fn=kernel_fn)

        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y):

        # R : query pattern
        # Y : memory pattern

        B, L, _ = R.shape
        B, S, _ = Y.shape
        H = self.n_heads
        # R, Y = self.ln(R), self.ln(Y)

        queries = R.view(B, L, H, -1)
        keys = Y.view(B, S, H, -1)
        values = Y.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
            )

        out = queries
        out = out.view(B, L, -1)
        return out

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling=0, generalized_attention = False, kernel_fn = "relu", no_projection = False, scale=0.1):
        super().__init__()

        nb_features = nb_features if nb_features is not None else int(dim_heads * math.log(dim_heads))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.kernel_fn_name = kernel_fn
        if kernel_fn == "relu":
            self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
            self.kernel_fn = nn.ReLU()
        elif kernel_fn == 'elu':
            self.kernel_fn = nn.ELU()

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention

        # kernel_fn: ReLU, ELU, etc.

        self.attn = LinearHopfieldCore()

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = False

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, mask=None):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.kernel_fn_name == "elu":
            q, k = torch.nn.functional.elu(q)+1, torch.nn.functional.elu(k)+1

        elif self.kernel_fn_name == "relu":
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        out = self.attn(q, k, v, mask)
        return out

class FavorPooling(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            num_pattern=1,
            update_steps=1,
            mode='gaussian',
            kernel_fn="relu"
            ):
        super(FavorPooling, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = FastAttention(
            d_keys, nb_features=None, ortho_scaling=0, generalized_attention=(mode=='gaussian'), kernel_fn=kernel_fn, no_projection=False)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

        pooling_weight_size = d_model

        self.query = nn.Parameter(
            torch.rand(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    d_model if pooling_weight_size is None else pooling_weight_size)),
            requires_grad=True)

    def forward(self, Y, mask=None):

        # queries : state pattern
        # keys : store pattern
        # values : should just be keys

        B, L, _ = self.query.shape
        B, S, _ = Y.shape
        H = self.n_heads
        q = self.query.repeat((*((B, 1)), 1))

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)

class NPH(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            update_steps=1,
            rand_prob=0.1,
            win_size=2,
            mode='rand',
            scale=None):
        super(NPH, self).__init__()

        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = SparseStructured_HopfieldCore(
            scale=scale, mask_type=mode, rand_prob=rand_prob, win_size=win_size)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y, mask=None):

        B, L, _ = R.shape
        B, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for i in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)
        return self.out_projection(out)

class Favor(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            update_steps=1,
            mode='favor',
            kernel_fn = 'relu'
            ):
        super(Favor, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = FastAttention(
            d_keys, nb_features=None, ortho_scaling=0, generalized_attention=(mode=='gaussian'), kernel_fn=kernel_fn, no_projection=False)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y, mask=None):

        # queries : state pattern
        # keys : store pattern
        # values : should just be keys

        B, L, _ = R.shape
        B, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)

class NPHPooling(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            num_pattern=1,
            update_steps=1,
            rand_prob=0.1,
            mode='rand',
            scale=None):
        super(NPHPooling, self).__init__()

        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)
        self.d_model = d_model

        if mode != 'linear':
            self.inner_attention = SparseStructured_HopfieldCore(
                scale=scale, mask_type=mode, rand_prob=rand_prob)
        else:
            self.inner_attention = LinearHopfieldCore()

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

        pooling_weight_size = d_model

        self.query = nn.Parameter(
            torch.rand(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    d_model if pooling_weight_size is None else pooling_weight_size)),
            requires_grad=True)

    def forward(self, Y, mask=None):

        # queries : state pattern
        # keys : store pattern
        # values : should just be keys
        keys = Y

        B, L, _ = self.query.shape
        B, S, _ = keys.shape
        H = self.n_heads
        q = self.query.repeat((*((B, 1)), 1))

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(keys)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for i in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)
        return self.out_projection(out)
    
