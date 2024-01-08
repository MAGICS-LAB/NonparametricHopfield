import torch
import torch.nn as nn

from layers import *

class BITModel(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            update_steps,
            dropout,
            mode,
            scale=None,
            n_heads=8,
            d_keys=8,
            d_values=8,
            num_pattern=1):
        super(BITModel, self).__init__()

        if mode in ["sparsemax", 'softmax']:
            self.layer = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                dropout=dropout,
                d_keys=d_keys,
                d_values=d_values,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)

        self.fc = nn.Linear(d_model*num_pattern, 1)
        self.gelu = nn.GELU()

    def forward(self, x):

        bz, N, d = x.size()
        out = self.gelu(self.layer(x))
        out = out.view(bz, -1)
        return self.fc(out).squeeze(-1)

class MNISTModel(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            n_heads,
            update_steps,
            dropout,
            mode,
            scale=None,
            num_pattern=1,
            favor_mode='gaussian'):
        super(MNISTModel, self).__init__()

        assert d_model % n_heads == 0
        self.emb = nn.Linear(input_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if mode in ["sparsemax", 'softmax']:
            self.layer = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                dropout=dropout,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)

        elif mode in ["rand", "window", "topk"]:
            self.layer = NPHPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                rand_prob=dropout,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)
        
        elif mode in ["favor"]:
            self.layer = FavorPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                num_pattern=num_pattern,
                update_steps=update_steps,
                mode=favor_mode,
                kernel_fn='relu'
                )

        elif mode in ["linear"]:
            self.layer = FavorPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                num_pattern=num_pattern,
                update_steps=update_steps,
                mode=favor_mode,
                kernel_fn='elu'
                )

        self.fc = nn.Linear(d_model*num_pattern, 1)
        self.gelu = nn.GELU()

    def forward(self, x):

        bz, N, c, h, w = x.size()
        x = x.view(bz, N, -1)
        x = self.ln(self.emb(x))
        out = self.ln2(self.gelu(self.layer(x)))
        out = out.view(bz, -1)
        return self.fc(out).squeeze(-1)

class TSModel(nn.Module):
    def __init__(
            self,
            out_len,
            d_model,
            n_heads,
            update_steps,
            dropout,
            mode,
            scale=None,
            kernel_fn='relu',
            win_size=2):
        super(TSModel, self).__init__()

        assert d_model % n_heads == 0
        self.emb = nn.Linear(1, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        if mode in ["sparsemax", 'softmax']:
            self.layer = Hopfield(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                dropout=dropout,
                mode=mode,
                scale=scale)

        elif mode in ["rand", "window", "topk", "tvm"]:
            self.layer = NPH(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                rand_prob=dropout,
                mode=mode,
                scale=scale,
                win_size=win_size)
        
        elif mode in ["favor", "linear"]:
            self.layer = Favor(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                kernel_fn=kernel_fn)

        if mode in ["sparsemax", 'softmax']:
            self.pool = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                mode=mode,
                scale=scale,
                num_pattern=1)

        elif mode in ["window"]:
            self.pool = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                mode="sparsemax",
                scale=scale,
                num_pattern=1)

        elif mode in ["rand", "topk"]:
            self.pool = NPHPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                rand_prob=dropout,
                mode=mode,
                scale=scale,
                num_pattern=1)
        
        elif mode in ["favor", "linear"]:
            self.pool = FavorPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                num_pattern=1,
                update_steps=update_steps,
                kernel_fn=kernel_fn
                )

        self.fc = nn.Linear(d_model, out_len)
        self.gelu = nn.GELU()

    def forward(self, x):

        bz, N, c = x.size()
        x = self.ln(self.emb(x)) # bz, N, emb_dim
        out = self.ln2(self.gelu(self.layer(x, x)))
        out = out.view(bz, N, -1)
        out = self.ln3(self.pool(out).squeeze(1))
        out = self.fc(out)
        return out.unsqueeze(-1)

class MILModel(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            n_heads,
            update_steps,
            dropout,
            mode,
            scale=None,
            num_pattern=1,
            prob=0.2,
            favor_mode='gaussian',
            kernel_fn="relu"):
        super(MILModel, self).__init__()

        assert d_model % n_heads == 0
        self.emb = nn.Linear(input_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if mode in ["sparsemax", 'softmax']:
            self.layer = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)

        elif mode in ["rand", "window", "topk"]:
            self.layer = NPHPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                rand_prob=prob,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)
        
        elif mode in ["favor", "linear"]:
            self.layer = FavorPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                num_pattern=num_pattern,
                update_steps=update_steps,
                mode=favor_mode,
                kernel_fn=kernel_fn
                )

        self.fc = nn.Linear(d_model*num_pattern, 1)
        self.gelu = nn.GELU()

    def forward(self, x, mask=None):
        
        bz, N, d = x.size()
        x = self.dropout(self.ln(self.emb(x)))
        out = self.dropout(self.ln2(self.gelu(self.layer(x, mask=mask))))
        out = out.view(bz, -1)

        return self.fc(out).squeeze(-1)