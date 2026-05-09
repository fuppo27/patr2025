import torch   
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import os
from .modules_modified import ISAB, SAB, PMA, OrdinalHead
import pandas as pd
import openpyxl
import warnings
from typing import Mapping, Iterable, Tuple, List, Union, Optional


# -----------------------------------------------Classifier Model Original-----------------------------------------------

# --- set transformer model ---
class SetTransformerClassifier(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8):
        super().__init__()
        
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self._meta_features_pma = None
        self.embedding = nn.Embedding(vocab_size + 1 , dim_in, padding_idx = 0)
        self.encoder = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        # x: (B, N, dim_in)
        x = self.embedding(x)
        x_enc = self.encoder(x)
        pma_pooled = self.decoder[0](x_enc)
        pma_flat = self.decoder[1](pma_pooled)
        self._meta_features = x_enc.mean(dim=1)
        self._meta_features_pma = pma_flat
        return self.decoder[2](pma_flat)
    
    def get_meta_features(self, source="mean"):
        if source == "pma":
            return self._meta_features_pma
        return self._meta_features
    
    # def forward(self, x):  
    #     # x: (B, N, dim_in)
    #     x = self.embedding(x)
    #     x_enc = self.encoder(x)  
    #     pooled = self.decoder[0](x_enc)  # 1. Apply PMA (Smart Pooling)
    #     pooled = self.decoder[1](pooled)  # 2. Flatten
    #     self._meta_features = pooled  # <--- Saves PMA Output
    #     return self.decoder[2](pooled)  # 3. Apply Linear
    
# --- deepset model ---
class DeepSetClassifier(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8):
        super().__init__()

        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self.embedding = nn.Embedding(vocab_size + 1, dim_in,  padding_idx=0)  # Embed hold indices

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, hold_idx):  # hold_idx: (B, N)
        x = self.embedding(hold_idx)   # (B, N, dim_in)
        x = self.encoder(x)            # (B, N, hidden)
        x = x.mean(dim=1)              # (B, hidden)
        self._meta_features = x
        out = self.decoder(x)          # (B, num_classes)
        return out
    
    def get_meta_features(self):
        return self._meta_features



# ----------------------------------------------- Classifier Model with XY-----------------------------------------------

# --- set transformer model ---
class SetTransformerClassifierXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()

        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self._meta_features_pma = None
        self.embedding = nn.Embedding(vocab_size + 1, dim_in, padding_idx = 0)  # hold embedding
        input_dim = dim_in + 1 + type_vec_dim + 2  # +1 for difficulty, +2 for (x, y)

        self.encoder = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )

        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    # def forward(self, inputs):
    #     hold_idx, difficulty, type_tensor, xy_tensor = inputs  # shapes: (B, N), (B, N), (B, N, T), (B, N, 2)
    #     x_embed = self.embedding(hold_idx)              # (B, N, dim_in)
    #     difficulty = difficulty.unsqueeze(-1)           # (B, N, 1)
    #     x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, D+1+T+2)
    #     x_enc = self.encoder(x)
    #     pooled = self.decoder[0](x_enc)   # 1. Apply PMA (smart pooling)
    #     pooled = self.decoder[1](pooled)  # 2. Flatten
    #     self._meta_features = pooled      # 3. Save pooled features
    #     return self.decoder[2](pooled)    # 4. Linear head

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # shapes: (B, N), (B, N), (B, N, T), (B, N, 2)
        x_embed = self.embedding(hold_idx)              # (B, N, dim_in)
        difficulty = difficulty.unsqueeze(-1)           # (B, N, 1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, D+1+T+2)
        x_enc = self.encoder(x)
        pma_pooled = self.decoder[0](x_enc)
        pma_flat = self.decoder[1](pma_pooled)
        self._meta_features = x_enc.mean(dim=1)
        self._meta_features_pma = pma_flat
        return self.decoder[2](pma_flat)
    
    def get_meta_features(self, source="mean"):
        if source == "pma":
            return self._meta_features_pma
        return self._meta_features


# revised set transformer, embedding for type + XY
class SetTransformerClassifierXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self._meta_features_pma = None
        # (1) hold ID → embedding
        self.hold_emb   = nn.Embedding(vocab_size + 1, feat_dim, padding_idx = 0)          # (B,N,feat_dim)
        # (2) difficulty (scalar) → linear projection to feat_dim
        self.diff_proj  = nn.Linear(1, feat_dim)                      # (B,N,1) → (B,N,feat_dim)
        # (3) type (multi-hot length T) → embedding matrix, then matmul
        self.type_emb   = nn.Parameter(torch.randn(type_vec_dim, feat_dim))  # (T,feat_dim)
        # (4) XY (2-dim) → MLP to feat_dim
        self.xy_mlp     = nn.Sequential(
                nn.Linear(2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        # Learnable scalar weights for each feature
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy   = nn.Parameter(torch.tensor(1.0))

        self.encoder = nn.Sequential(
            ISAB(feat_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, num_classes)
        )

    # def forward(self, inputs):
    #     hold_idx, difficulty, type_tensor, xy_tensor = inputs
    #     # shapes: (B,N)  (B,N)  (B,N,T)  (B,N,2)
    #     # hold ID
    #     h  = self.hold_emb(hold_idx)                       # (B,N,feat_dim)
    #     # hold difficulty
    #     d  = difficulty.unsqueeze(-1)                     # (B,N,1)
    #     d  = self.diff_proj(d)                            # (B,N,feat_dim)
    #     # hold types
    #     t  = type_tensor @ self.type_emb                  # (B,N,feat_dim)
    #     # XY position
    #     xy = self.xy_mlp(xy_tensor)                       # (B,N,feat_dim)
    #     # element-wise addition
    #     # x = h + d + t + xy                                # (B,N,feat_dim)
    #     # Weighted sum
    #     x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
    #     x_enc = self.encoder(x)
    #     pooled = self.decoder[0](x_enc)   # PMA pooling
    #     pooled = self.decoder[1](pooled)  # Flatten
    #     self._meta_features = pooled
    #     return self.decoder[2](pooled)    # Linear head

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        # shapes: (B,N)  (B,N)  (B,N,T)  (B,N,2)
        # hold ID
        h  = self.hold_emb(hold_idx)                       # (B,N,feat_dim)
        # hold difficulty
        d  = difficulty.unsqueeze(-1)                     # (B,N,1)
        d  = self.diff_proj(d)                            # (B,N,feat_dim)
        # hold types
        t  = type_tensor @ self.type_emb                  # (B,N,feat_dim)
        # XY position
        xy = self.xy_mlp(xy_tensor)                       # (B,N,feat_dim)
        # element-wise addition
        # x = h + d + t + xy                                # (B,N,feat_dim)
        # Weighted sum
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x_enc = self.encoder(x)
        pma_pooled = self.decoder[0](x_enc)
        pma_flat = self.decoder[1](pma_pooled)
        self._meta_features = x_enc.mean(dim=1)
        self._meta_features_pma = pma_flat
        return self.decoder[2](pma_flat)

    def get_meta_features(self, source="mean"):
        if source == "pma":
            return self._meta_features_pma
        return self._meta_features
    
    
# --- deepset model ---
class DeepSetClassifierXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()

        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self.embedding = nn.Embedding(vocab_size + 1, dim_in, padding_idx=0)
        input_dim = dim_in + 1 + type_vec_dim + 2  # hold_emb + difficulty + type_vec + (x, y)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # (B, N), (B, N), (B, N, T), (B, N, 2)

        x_embed = self.embedding(hold_idx)             # (B, N, dim_in)
        difficulty = difficulty.unsqueeze(-1)          # (B, N, 1)

        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)  # (B, N, total_input_dim)
        x = self.encoder(x)                            # (B, N, hidden_dim)
        x = x.mean(dim=1)                              # (B, hidden_dim)
        self._meta_features = x
        return self.decoder(x)                         # (B, num_classes)
    
    def get_meta_features(self):
        return self._meta_features
    
    
class DeepSetClassifierXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        # (1) hold ID → embedding
        self.hold_emb   = nn.Embedding(vocab_size + 1, feat_dim, padding_idx = 0)           # (B,N,feat_dim)
        # (2) difficulty (scalar) → linear projection to feat_dim
        self.diff_proj  = nn.Linear(1, feat_dim)                       # (B,N,1) → (B,N,feat_dim)
        # (3) type (multi-hot length T) → embedding matrix, then matmul
        self.type_emb   = nn.Parameter(torch.randn(type_vec_dim, feat_dim))  # (T,feat_dim)
        # (4) XY (2-dim) → MLP to feat_dim
        self.xy_mlp     = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        # Learnable scalar weights for each feature
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy   = nn.Parameter(torch.tensor(1.0))

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs  # (B, N), (B, N), (B, N, T), (B, N, 2)

        # Embed hold indices
        h = self.hold_emb(hold_idx)                    # (B, N, feat_dim)
        # Project difficulty
        d = self.diff_proj(difficulty.unsqueeze(-1))   # (B, N, feat_dim)
        # Embed type (multi-hot)
        t = type_tensor @ self.type_emb                # (B, N, feat_dim)
        # Process XY
        xy = self.xy_mlp(xy_tensor)                    # (B, N, feat_dim)

        # Weighted additive fusion
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy  # (B, N, feat_dim)

        # Encode + aggregate
        x = self.encoder(x)       # (B, N, hidden_dim)
        x = x.mean(dim=1)         # (B, hidden_dim)
        self._meta_features = x
        return self.decoder(x)    # (B, num_classes)
    
    def get_meta_features(self):
        return self._meta_features
