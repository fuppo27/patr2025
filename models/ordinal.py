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

from .utils_ordinal import cumulative_to_labels

TensorLike = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]


def _safe_logit(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert probabilities (0,1) to logits with numerical safety."""
    probs = probs.clamp(min=eps, max=1 - eps)
    return torch.log(probs / (1 - probs))


def class_probs_to_ordinal_probs(class_probs: torch.Tensor) -> torch.Tensor:
    """
    Convert per-class probabilities [B, C] into cumulative form P(y > k) of shape [B, C-1].
    """
    flipped = torch.flip(class_probs, dims=[-1])
    tail_sums = torch.cumsum(flipped, dim=-1)
    tail_sums = torch.flip(tail_sums, dims=[-1])
    return tail_sums[:, 1:]

# -----------------------------------------------Ordinal Models -------------------------------------------------

class SetTransformerOrdinalXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self.embedding = nn.Embedding(vocab_size + 1 , dim_in, padding_idx = 0)
        input_dim = dim_in + 1 + type_vec_dim + 2
        self.encoder = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    # def forward(self, inputs):
    #     hold_idx, difficulty, type_tensor, xy_tensor = inputs
    #     x_embed = self.embedding(hold_idx)
    #     difficulty = difficulty.unsqueeze(-1)
    #     x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)
    #     x_enc = self.encoder(x)
    #     pooled = self.pool[0](x_enc)    # PMA pooling
    #     pooled = self.pool[1](pooled)   # Flatten
    #     self._meta_features = pooled    # cache pooled representation
    #     features = self.pool[2](pooled) # Linear projection
    #     features = self.pool[3](features)  # ReLU
    #     return self.ordinal_head(features)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        x_embed = self.embedding(hold_idx)
        difficulty = difficulty.unsqueeze(-1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)
        x_enc = self.encoder(x)
        pooled = self.pool[0](x_enc)
        pooled = self.pool[1](pooled)
        self._meta_features = pooled
        features = self.pool[2](pooled)
        features = self.pool[3](features)
        return self.ordinal_head(features)

    def get_meta_features(self):
        return self._meta_features


class SetTransformerOrdinalXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self.hold_emb = nn.Embedding(vocab_size + 1, feat_dim, padding_idx = 0)
        self.diff_proj = nn.Linear(1, feat_dim)
        self.type_emb = nn.Parameter(torch.randn(type_vec_dim, feat_dim))
        self.xy_mlp = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.Sequential(
            ISAB(feat_dim, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    # def forward(self, inputs):
    #     hold_idx, difficulty, type_tensor, xy_tensor = inputs
    #     h = self.hold_emb(hold_idx)
    #     d = self.diff_proj(difficulty.unsqueeze(-1))
    #     t = type_tensor @ self.type_emb
    #     xy = self.xy_mlp(xy_tensor)
    #     x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
    #     x_enc = self.encoder(x)
    #     pooled = self.pool[0](x_enc)    # PMA pooling
    #     pooled = self.pool[1](pooled)   # Flatten
    #     self._meta_features = pooled    # cache pooled representation
    #     features = self.pool[2](pooled) # Linear projection
    #     features = self.pool[3](features)  # ReLU
    #     return self.ordinal_head(features)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        h = self.hold_emb(hold_idx)
        d = self.diff_proj(difficulty.unsqueeze(-1))
        t = type_tensor @ self.type_emb
        xy = self.xy_mlp(xy_tensor)
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x_enc = self.encoder(x)
        pooled = self.pool[0](x_enc)
        pooled = self.pool[1](pooled)
        self._meta_features = pooled
        features = self.pool[2](pooled)
        features = self.pool[3](features)
        return self.ordinal_head(features)

    def get_meta_features(self):
        return self._meta_features


class SetTransformerOrdinal(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_heads=4, num_inds=16, num_classes=8):
        super().__init__()
        self.meta_feature_dim = dim_hidden
        self._meta_features = None
        self.embedding = nn.Embedding(vocab_size + 1 , dim_in, padding_idx = 0)
        self.encoder = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=True),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
        )
        self.pool = nn.Sequential(
            PMA(dim_hidden, num_heads, 1, ln=True),
            nn.Flatten(start_dim=1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    # def forward(self, hold_idx):
    #     x = self.embedding(hold_idx)
    #     x_enc = self.encoder(x)
    #     pooled = self.pool[0](x_enc)    # PMA pooling
    #     pooled = self.pool[1](pooled)   # Flatten
    #     self._meta_features = pooled    # cache pooled representation
    #     features = self.pool[2](pooled) # Linear projection
    #     features = self.pool[3](features)  # ReLU
    #     return self.ordinal_head(features)

    def forward(self, hold_idx):
        x = self.embedding(hold_idx)
        x_enc = self.encoder(x)
        pooled = self.pool[0](x_enc)
        pooled = self.pool[1](pooled)
        self._meta_features = pooled
        features = self.pool[2](pooled)
        features = self.pool[3](features)
        return self.ordinal_head(features)

    def get_meta_features(self):
        return self._meta_features


class DeepSetOrdinalXY(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1 , dim_in, padding_idx = 0)
        input_dim = dim_in + 1 + type_vec_dim + 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        x_embed = self.embedding(hold_idx)
        difficulty = difficulty.unsqueeze(-1)
        x = torch.cat([x_embed, difficulty, type_tensor, xy_tensor], dim=-1)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


class DeepSetOrdinalXYAdditive(nn.Module):
    expects_tuple_input = True
    def __init__(self, vocab_size, feat_dim=64, dim_hidden=128, num_classes=8, type_vec_dim=10):
        super().__init__()
        self.hold_emb = nn.Embedding(vocab_size + 1 , feat_dim, padding_idx = 0)
        self.diff_proj = nn.Linear(1, feat_dim)
        self.type_emb = nn.Parameter(torch.randn(type_vec_dim, feat_dim))
        self.xy_mlp = nn.Sequential(
            nn.Linear(2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.w_hold = nn.Parameter(torch.tensor(1.0))
        self.w_diff = nn.Parameter(torch.tensor(1.0))
        self.w_type = nn.Parameter(torch.tensor(1.0))
        self.w_xy = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, inputs):
        hold_idx, difficulty, type_tensor, xy_tensor = inputs
        h = self.hold_emb(hold_idx)
        d = self.diff_proj(difficulty.unsqueeze(-1))
        t = type_tensor @ self.type_emb
        xy = self.xy_mlp(xy_tensor)
        x = self.w_hold * h + self.w_diff * d + self.w_type * t + self.w_xy * xy
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


class DeepSetOrdinal(nn.Module):
    expects_tuple_input = False
    def __init__(self, vocab_size, dim_in=64, dim_hidden=128, num_classes=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1 , dim_in, padding_idx = 0)
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.ordinal_head = OrdinalHead(dim_hidden, num_classes)

    def forward(self, hold_idx):
        x = self.embedding(hold_idx)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        features = self.proj(pooled)
        return self.ordinal_head(features)


# -----------------------------------------------Ordinal Ensembles -------------------------------------------------


class BaseOrdinalEnsemble(nn.Module):
    """
    Base class for ensembles that combine ordinal (cumulative) probabilities.
    Members must return a tuple (probs, logits) from their forward pass.
    """

    def __init__(
        self,
        models: Union[Mapping[str, nn.Module], Iterable[Tuple[str, nn.Module]]],
        weights: Optional[Union[Mapping[str, float], Iterable[float]]] = None,
        freeze_members: bool = True,
    ):
        super().__init__()
        if isinstance(models, dict):
            items = list(models.items())
        else:
            items = list(models)
            if not all(isinstance(it, (tuple, list)) and len(it) == 2 for it in items):
                raise ValueError("models must be a mapping or iterable of (name, module) pairs.")
        if not items:
            raise ValueError("At least one base model is required for an ensemble.")

        self._model_names = [name for name, _ in items]
        self.models = nn.ModuleDict((name, module) for name, module in items)
        self.register_buffer("_weights", self._resolve_weights(weights, self._model_names))

        if freeze_members:
            for module in self.models.values():
                for param in module.parameters():
                    param.requires_grad_(False)

        self.is_ensemble = True
        self.is_ordinal = True

    @staticmethod
    def _resolve_weights(weights, names):
        if weights is None:
            vals = torch.ones(len(names), dtype=torch.float32)
        elif isinstance(weights, dict):
            vals = torch.tensor([float(weights[n]) for n in names], dtype=torch.float32)
        else:
            vals = torch.tensor(list(weights), dtype=torch.float32)
            if vals.numel() != len(names):
                raise ValueError("weights length must match number of models.")
        total = vals.sum()
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return vals / total

    @staticmethod
    def _prepare_inputs(module: nn.Module, raw_inputs: TensorLike) -> TensorLike:
        expects_tuple = getattr(module, "expects_tuple_input", False)
        if expects_tuple:
            if not isinstance(raw_inputs, (tuple, list)):
                raise ValueError(
                    f"Model {module.__class__.__name__} expects tuple inputs but received {type(raw_inputs)}"
                )
            return raw_inputs
        if isinstance(raw_inputs, (tuple, list)):
            if not raw_inputs:
                raise ValueError("Empty inputs supplied to ensemble member.")
            return raw_inputs[0]
        return raw_inputs

    @staticmethod
    def _extract_probs_logits(outputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(outputs, (tuple, list)):
            tensor_items = [x for x in outputs if isinstance(x, torch.Tensor)]
            if len(tensor_items) >= 2:
                return tensor_items[0], tensor_items[1]
            raise ValueError("Ordinal ensemble members must return (probs, logits).")
        raise ValueError("Ordinal ensemble members must return tuple outputs.")

    def _member_probs(self, inputs: TensorLike) -> torch.Tensor:
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        probs = []
        for module in self.models.values():
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)
            member_probs, _ = self._extract_probs_logits(outputs)
            probs.append(member_probs)
        return torch.stack(probs, dim=0)  # [M, B, T]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        member_probs = self._member_probs(inputs)
        combined = self._combine(member_probs, self._weights)
        combined = combined.clamp(min=1e-6, max=1 - 1e-6)
        logits = _safe_logit(combined)
        return combined, logits

    def predict(self, inputs: TensorLike) -> torch.Tensor:
        probs, _ = self.forward(inputs)
        return cumulative_to_labels(probs)


class OrdinalSoftVotingEnsemble(BaseOrdinalEnsemble):
    """Weighted arithmetic mean over ordinal probabilities."""

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)


class OrdinalGeometricMeanEnsemble(BaseOrdinalEnsemble):
    """Weighted geometric mean of ordinal probabilities."""

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        logp = torch.log(member_probs.clamp_min(eps))
        w = weights.view(-1, 1, 1)
        return torch.exp((w * logp).sum(dim=0))


class OrdinalMedianEnsemble(BaseOrdinalEnsemble):
    """Per-threshold median, ignoring weights."""

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        _ = weights  # unused
        return member_probs.median(dim=0).values


class OrdinalTrimmedMeanEnsemble(BaseOrdinalEnsemble):
    """Trimmed mean across members, ignoring weights."""

    def __init__(self, models, weights=None, freeze_members=True, trim_frac: float = 0.2):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if not (0.0 <= trim_frac < 0.5):
            raise ValueError("trim_frac must be in [0, 0.5).")
        self.trim_frac = float(trim_frac)

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        _ = weights
        M = member_probs.shape[0]
        k = int(M * self.trim_frac)
        if k == 0:
            return member_probs.mean(dim=0)
        sorted_probs, _ = torch.sort(member_probs, dim=0)
        trimmed = sorted_probs[k: M - k]
        return trimmed.mean(dim=0)


class OrdinalStackingEnsemble(BaseOrdinalEnsemble):
    """
    Stacking ensemble for ordinal models. The meta-model may emit either
    class logits (size = num_classes) or ordinal logits (size = num_classes - 1).
    """

    def __init__(
        self,
        models,
        *,
        num_classes: int,
        meta_model: nn.Module,
        weights=None,
        freeze_members: bool = True,
        feature_source: str = "logits",
        combine: str = "concat",
    ):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal stacking.")
        if meta_model is None:
            raise ValueError("OrdinalStackingEnsemble requires a meta_model.")
        if combine not in {"concat", "mean"}:
            raise ValueError("combine must be 'concat' or 'mean'.")
        self.num_classes = int(num_classes)
        self.meta_model = meta_model
        self.feature_source = feature_source
        self.combine = combine
        self._base_feature_mode, self._use_internal = self._parse_feature_source(feature_source)

    @staticmethod
    def _parse_feature_source(source: str) -> Tuple[Optional[str], bool]:
        valid = {
            "logits": ("logits", False),
            "probs": ("probs", False),
            "internal": (None, True),
            "logits+internal": ("logits", True),
            "probs+internal": ("probs", True),
        }
        if source not in valid:
            raise ValueError(
                "feature_source must be one of "
                "'logits', 'probs', 'internal', 'logits+internal', or 'probs+internal'."
            )
        return valid[source]

    def _member_features(self, inputs: TensorLike) -> torch.Tensor:
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        feats = []
        for name, module in self.models.items():
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)
            probs, logits = self._extract_probs_logits(outputs)
            blocks = []
            if self._base_feature_mode == "probs":
                blocks.append(probs)
            elif self._base_feature_mode == "logits":
                blocks.append(logits)
            if self._use_internal:
                extra_dim = int(getattr(module, "meta_feature_dim", 0))
                extractor = getattr(module, "get_meta_features", None)
                if extra_dim <= 0 or extractor is None:
                    raise ValueError(
                        f"Model '{name}' does not expose encoder features required by feature_source='{self.feature_source}'."
                    )
                internal = extractor()
                if internal is None:
                    raise RuntimeError(
                        f"Model '{name}' failed to cache encoder features for stacking."
                    )
                if internal.ndim > 2:
                    internal = internal.reshape(internal.shape[0], -1)
                if internal.shape[-1] != extra_dim:
                    raise ValueError(
                        f"Model '{name}' reported meta_feature_dim={extra_dim} but produced {internal.shape[-1]}."
                    )
                blocks.append(internal)
            if not blocks:
                raise RuntimeError(f"No features constructed for model '{name}'.")
            feats.append(blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1))
        return torch.stack(feats, dim=0)  # [M, B, F]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)

    def _meta_logits_to_ordinal(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_dim = logits.shape[-1]
        if feature_dim == self.num_classes - 1:
            ordinal_logits = logits
            ordinal_probs = torch.sigmoid(logits)
        elif feature_dim == self.num_classes:
            class_probs = F.softmax(logits, dim=-1)
            ordinal_probs = class_probs_to_ordinal_probs(class_probs)
            ordinal_logits = _safe_logit(ordinal_probs)
        else:
            raise ValueError(
                f"Meta-model output dimension {feature_dim} incompatible with num_classes={self.num_classes}."
            )
        return ordinal_probs, ordinal_logits

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        member_feats = self._member_features(inputs)
        M, B, feat_dim = member_feats.shape
        if self.combine == "mean":
            feat = member_feats.mean(dim=0)
        else:
            feat = member_feats.permute(1, 0, 2).reshape(B, M * feat_dim)
        logits = self.meta_model(feat)
        ordinal_probs, ordinal_logits = self._meta_logits_to_ordinal(logits)
        return ordinal_probs, ordinal_logits


class OrdinalTreeMetaEnsemble(BaseOrdinalEnsemble):
    """
    Base class for ordinal stacking ensembles backed by tree-based meta learners.
    """

    def __init__(
        self,
        models,
        *,
        num_classes: int,
        meta_model,
        weights=None,
        freeze_members: bool = True,
        feature_source: str = "logits",
        combine: str = "concat",
    ):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if feature_source not in {"logits", "probs", "both"}:
            raise ValueError("feature_source must be 'logits', 'probs', or 'both'.")
        if combine not in {"concat", "mean"}:
            raise ValueError("combine must be 'concat' or 'mean'.")
        if meta_model is None:
            raise ValueError("OrdinalTreeMetaEnsemble requires a meta_model instance.")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal ensembles.")
        self.meta_model = meta_model
        self.feature_source = feature_source
        self.combine = combine
        self.num_classes = int(num_classes)
        self._is_fitted = False
        self._meta_classes: Optional[List[int]] = None

    def _member_features(self, inputs: TensorLike) -> torch.Tensor:
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        feats = []
        for module in self.models.values():
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)
            probs, logits = self._extract_probs_logits(outputs)
            if self.feature_source == "probs":
                feats.append(probs)
            elif self.feature_source == "both":
                feats.append(torch.cat([logits, probs], dim=-1))
            else:
                feats.append(logits)
        return torch.stack(feats, dim=0)  # [M, B, T]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)

    def _build_feature_matrix(self, member_feats: torch.Tensor) -> torch.Tensor:
        M, B, T = member_feats.shape
        if self.combine == "mean":
            return member_feats.mean(dim=0)
        return member_feats.permute(1, 0, 2).reshape(B, M * T)

    def fit_meta_model(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("features must be [N, F] and targets [N].")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must align along the first dimension.")
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets)
        if not hasattr(self.meta_model, "fit") or not hasattr(self.meta_model, "predict_proba"):
            raise TypeError("meta_model must implement fit() and predict_proba().")
        self.meta_model.fit(features, targets)
        classes = getattr(self.meta_model, "classes_", None)
        if classes is None:
            classes = np.arange(self.num_classes, dtype=int)
        else:
            classes = np.asarray(classes)
        if classes.ndim != 1:
            raise ValueError("meta_model.classes_ must be one-dimensional.")
        self._meta_classes = [int(c) for c in classes.tolist()]
        max_seen = max(self._meta_classes, default=self.num_classes - 1)
        if max_seen >= self.num_classes:
            self.num_classes = max_seen + 1
        self._is_fitted = True

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError("Tree meta-model not fitted. Call fit_meta_model first.")
        member_feats = self._member_features(inputs)
        feat = self._build_feature_matrix(member_feats)
        feat_np = feat.detach().cpu().numpy()
        probs_np = self.meta_model.predict_proba(feat_np)
        probs_np = np.asarray(probs_np)
        if probs_np.ndim == 1:
            probs_np = np.stack([1.0 - probs_np, probs_np], axis=1)
        B = probs_np.shape[0]
        full = np.zeros((B, self.num_classes), dtype=np.float32)
        classes = self._meta_classes or list(range(probs_np.shape[1]))
        for col_idx, cls in enumerate(classes):
            if 0 <= cls < self.num_classes:
                full[:, cls] = probs_np[:, col_idx]
        row_sums = full.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            full = np.divide(full, row_sums, out=np.zeros_like(full), where=row_sums > 0)
        zero_mask = (row_sums <= 0).reshape(-1)
        if np.any(zero_mask):
            full[zero_mask] = 1.0 / self.num_classes
        class_probs = torch.from_numpy(full).to(member_feats.device)
        ordinal_probs = class_probs_to_ordinal_probs(class_probs)
        ordinal_logits = _safe_logit(ordinal_probs)
        return ordinal_probs, ordinal_logits


class OrdinalGBMEnsemble(OrdinalTreeMetaEnsemble):
    def __init__(
        self,
        models,
        *,
        num_classes: int,
        weights=None,
        freeze_members: bool = True,
        feature_source: str = "logits",
        combine: str = "concat",
        meta_model=None,
        meta_kwargs: Optional[dict] = None,
    ):
        if meta_model is None:
            try:
                from sklearn.ensemble import GradientBoostingClassifier
            except ImportError as exc:
                raise ImportError(
                    "OrdinalGBMEnsemble requires scikit-learn. Install it via `pip install scikit-learn`."
                ) from exc
            meta_kwargs = meta_kwargs or {}
            meta_model = GradientBoostingClassifier(**meta_kwargs)
        super().__init__(
            models,
            num_classes=num_classes,
            meta_model=meta_model,
            weights=weights,
            freeze_members=freeze_members,
            feature_source=feature_source,
            combine=combine,
        )


class OrdinalXGBoostEnsemble(OrdinalTreeMetaEnsemble):
    def __init__(
        self,
        models,
        *,
        num_classes: int,
        weights=None,
        freeze_members: bool = True,
        feature_source: str = "logits",
        combine: str = "concat",
        meta_model=None,
        meta_kwargs: Optional[dict] = None,
    ):
        if meta_model is None:
            try:
                import xgboost as xgb  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OrdinalXGBoostEnsemble requires the xgboost package. Install it via `pip install xgboost`."
                ) from exc
            meta_kwargs = meta_kwargs or {}
            meta_kwargs.setdefault("use_label_encoder", False)
            meta_kwargs.setdefault("eval_metric", "mlogloss")
            meta_model = xgb.XGBClassifier(**meta_kwargs)
        super().__init__(
            models,
            num_classes=num_classes,
            meta_model=meta_model,
            weights=weights,
            freeze_members=freeze_members,
            feature_source=feature_source,
            combine=combine,
        )


class OrdinalLightGBMEnsemble(OrdinalTreeMetaEnsemble):
    def __init__(
        self,
        models,
        *,
        num_classes: int,
        weights=None,
        freeze_members: bool = True,
        feature_source: str = "logits",
        combine: str = "concat",
        meta_model=None,
        meta_kwargs: Optional[dict] = None,
    ):
        if meta_model is None:
            try:
                import lightgbm as lgb  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OrdinalLightGBMEnsemble requires the lightgbm package. Install it via `pip install lightgbm`."
                ) from exc
            meta_kwargs = meta_kwargs or {}
            meta_model = lgb.LGBMClassifier(**meta_kwargs)
        super().__init__(
            models,
            num_classes=num_classes,
            meta_model=meta_model,
            weights=weights,
            freeze_members=freeze_members,
            feature_source=feature_source,
            combine=combine,
        )


class OrdinalAdaBoostEnsemble(BaseOrdinalEnsemble):
    """Weighted combination for ordinal AdaBoost-style ensembles."""

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)
