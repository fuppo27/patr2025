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

# -----------------------------------------------ensemble model-----------------------------------------------
TensorLike = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]

# =========================
# Base + Helpers
# =========================
class BaseEnsemble(nn.Module):
    """
    Base class for classification ensembles that combine member *probabilities*.
    Subclasses should implement `_combine(probs: Tensor [M,B,C], weights: Tensor [M]) -> Tensor [B,C]`.
    - Members can declare `expects_tuple_input=True` to receive tuple inputs; otherwise they get inputs[0].
    - Members may return a tensor (logits) or a tuple where the first tensor-like is treated as logits.
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

        w = self._resolve_weights(weights, self._model_names)
        self.register_buffer("_weights", w)  # shape [M], sums to 1

        if freeze_members:
            for m in self.models.values():
                for p in m.parameters():
                    p.requires_grad_(False)

        # Hints for your pipeline
        self.is_ensemble = True
        self.is_ordinal = False

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
        s = vals.sum()
        if s <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        return vals / s

    @staticmethod
    def _extract_logits(outputs: TensorLike) -> torch.Tensor:
        """Prefer the last tensor from tuple/list outputs (typically logits)."""
        if isinstance(outputs, (tuple, list)):
            tensor_items = [x for x in outputs if isinstance(x, torch.Tensor)]
            if tensor_items:
                for candidate in reversed(tensor_items):
                    if not BaseEnsemble._looks_like_probs(candidate):
                        return candidate
                return tensor_items[-1]
            raise ValueError("Model returned a tuple/list without any tensor outputs.")
        if not isinstance(outputs, torch.Tensor):
            raise ValueError(f"Unsupported model output type: {type(outputs)}")
        return outputs

    @staticmethod
    def _looks_like_probs(tensor: torch.Tensor) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.ndim < 2 or tensor.shape[-1] <= 1:
            return False
        detached = tensor.detach()
        if not torch.all(torch.isfinite(detached)).item():
            return False
        row_sums = detached.sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        ):
            return False
        return torch.all((detached >= 0) & (detached <= 1)).item()

    @staticmethod
    def _prepare_inputs(module: nn.Module, raw_inputs: TensorLike) -> TensorLike:
        expects_tuple = getattr(module, "expects_tuple_input", False)
        if expects_tuple:
            if not isinstance(raw_inputs, (tuple, list)):
                raise ValueError(
                    f"Model {module.__class__.__name__} expects tuple inputs but received {type(raw_inputs)}"
                )
            return raw_inputs
        # Non-tuple model gets the first element (hold_idx)
        if isinstance(raw_inputs, (tuple, list)):
            if not raw_inputs:
                raise ValueError("Empty inputs given to ensemble.")
            return raw_inputs[0]
        return raw_inputs

    def _member_probs(self, inputs: TensorLike) -> torch.Tensor:
        """
        Run each member, collect *probabilities* stack of shape [M, B, C].
        """
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        probs_list = []
        for module in self.models.values():
            prepared = self._prepare_inputs(module, tuple_inputs)
            outputs = module(prepared)
            logits = self._extract_logits(outputs)
            use_as_probs = False
            if logits.ndim >= 2 and logits.shape[-1] > 1:
                if self._looks_like_probs(logits):
                    warnings.warn(
                        f"Ensemble member {module.__class__.__name__} appears to return probabilities; "
                        "expected raw logits.",
                        RuntimeWarning,
                    )
                    use_as_probs = True
            probs = logits if use_as_probs else F.softmax(logits, dim=-1)
            probs_list.append(probs)
        # [M, B, C]
        return torch.stack(probs_list, dim=0)

    # --- abstract-ish: combine member probs into [B, C]
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs:  [B, C]
            logits: [B, C] (log of probs for convenience)
        """
        member_probs = self._member_probs(inputs)    # [M, B, C]
        combined = self._combine(member_probs, self._weights)  # [B, C]
        # ensure normalization & numeric safety
        combined = combined.clamp_min(1e-12)
        combined = combined / combined.sum(dim=-1, keepdim=True)
        logits = torch.log(combined)
        return combined, logits

    def predict(self, inputs: TensorLike) -> torch.Tensor:
        probs, _ = self.forward(inputs)
        return probs.argmax(dim=-1)


# =========================
# bagging
# =========================

# 1) Soft Voting (Arithmetic Mean)
class SoftVotingEnsemble(BaseEnsemble):
    """Weighted arithmetic mean of member probabilities (classic soft voting)."""
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # member_probs: [M,B,C], weights: [M]
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)  # [B,C]


# 2) Geometric Mean
class GeometricMeanEnsemble(BaseEnsemble):
    """
    Weighted geometric mean of probabilities:
    exp( sum_i w_i * log(p_i) ), then renormalize.
    Tends to penalize classes that any member rates near zero.
    """
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        logp = torch.log(member_probs.clamp_min(eps))  # [M,B,C]
        w = weights.view(-1, 1, 1)                     # [M,1,1]
        g = torch.exp((w * logp).sum(dim=0))           # [B,C]
        return g


# 3) Median Ensemble
class MedianEnsemble(BaseEnsemble):
    """
    Element-wise median of probabilities across members.
    NOTE: Ignores weights (robust estimator).
    """
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # weights are ignored by design for median
        return member_probs.median(dim=0).values  # [B,C]


# 4) Trimmed Mean Ensemble
class TrimmedMeanEnsemble(BaseEnsemble):
    """
    Per-class trimmed mean across members.
    - trim_frac: fraction to drop from both ends (0 <= trim_frac < 0.5).
    Ignores weights (robust estimator).
    """

    def __init__(self, models, weights=None, freeze_members=True, trim_frac: float = 0.2):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        if not (0.0 <= trim_frac < 0.5):
            raise ValueError("trim_frac must be in [0, 0.5).")
        self.trim_frac = float(trim_frac)

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        M = member_probs.shape[0]
        k = int(M * self.trim_frac)
        if k == 0:
            # falls back to simple mean (unweighted) if nothing to trim
            return member_probs.mean(dim=0)
        # sort along member axis for each (B,C)
        sorted_probs, _ = torch.sort(member_probs, dim=0)  # [M,B,C]
        trimmed = sorted_probs[k: M - k]                   # [M-2k,B,C]
        return trimmed.mean(dim=0)                         # [B,C]


# =========================
# stacking
# =========================

# 1) basic stacking
class StackingEnsemble(BaseEnsemble):
    """
    Stacking: feed member outputs to a meta-learner.
    - meta_model: nn.Module mapping features -> logits [B,C]
    - feature_source: 'logits', 'probs', 'internal', 'logits+internal', or 'probs+internal'
                     (internal pulls pooled encoder states from members that expose them)
    - combine: 'concat' (default) to concatenate member features along last dim,
               or 'mean' to average features before meta-model.

    Typical usage:
        meta = nn.Linear(M*C, C)   # simple logistic regression layer
        ens  = StackingEnsemble(models, meta_model=meta, feature_source='logits', combine='concat')

    Train the `meta_model` on a held-out validation set using the frozen base models.
    """

    def __init__(
        self,
        models,
        weights=None,
        freeze_members=True,
        meta_model: Optional[nn.Module] = None,
        feature_source: str = "logits",   # 'logits', 'probs', 'internal', 'pma_internal', 'logits+internal', 'logits+pma_internal', 'probs+internal', 'probs+pma_internal'
        combine: str = "concat",          # 'concat' or 'mean'
    ):
        super().__init__(models, weights=weights, freeze_members=freeze_members)
        self._base_feature_mode, self._use_internal, self._internal_source = self._parse_feature_source(feature_source)
        if combine not in {"concat", "mean"}:
            raise ValueError("combine must be 'concat' or 'mean'.")
        if meta_model is None:
            raise ValueError("StackingEnsemble requires a meta_model.")
        self.meta_model = meta_model
        self.feature_source = feature_source
        self.combine = combine

    @staticmethod
    def _parse_feature_source(source: str) -> Tuple[Optional[str], bool, str]:
        valid = {
            "logits": ("logits", False, "mean"),
            "probs": ("probs", False, "mean"),
            "internal": (None, True, "mean"),
            "pma_internal": (None, True, "pma"),
            "logits+internal": ("logits", True, "mean"),
            "logits+pma_internal": ("logits", True, "pma"),
            "probs+internal": ("probs", True, "mean"),
            "probs+pma_internal": ("probs", True, "pma"),
        }
        if source not in valid:
            raise ValueError(
                "feature_source must be one of "
                "'logits', 'probs', 'internal', 'pma_internal', "
                "'logits+internal', 'logits+pma_internal', 'probs+internal', or 'probs+pma_internal'."
            )
        return valid[source]

    def _member_features(self, inputs: TensorLike) -> torch.Tensor:
        """
        Collect requested features per member.
        Returns tensor with shape [M,B,F], where F depends on feature_source.
        """
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        feats = []
        for name, module in self.models.items():
            prepared = self._prepare_inputs(module, tuple_inputs)
            out = module(prepared)
            logits = self._extract_logits(out)
            blocks = []
            if self._base_feature_mode == "probs":
                blocks.append(F.softmax(logits, dim=-1))
            elif self._base_feature_mode == "logits":
                blocks.append(logits)

            if self._use_internal:
                extra_dim = int(getattr(module, "meta_feature_dim", 0))
                extractor = getattr(module, "get_meta_features", None)
                if extra_dim <= 0 or extractor is None:
                    raise ValueError(
                        f"Model '{name}' does not expose encoder features "
                        f"required by feature_source='{self.feature_source}'."
                    )
                if self._internal_source == "pma":
                    internal_feat = extractor("pma")
                else:
                    internal_feat = extractor()
                if internal_feat is None:
                    raise RuntimeError(
                        f"Model '{name}' failed to cache encoder features for stacking. "
                        "Ensure its forward method stores them before returning."
                    )
                if internal_feat.ndim > 2:
                    internal_feat = internal_feat.reshape(internal_feat.shape[0], -1)
                if internal_feat.shape[-1] != extra_dim:
                    raise ValueError(
                        f"Model '{name}' reported meta_feature_dim={extra_dim} "
                        f"but produced features with size {internal_feat.shape[-1]}."
                    )
                blocks.append(internal_feat)

            if not blocks:
                raise RuntimeError(
                    f"feature_source '{self.feature_source}' produced no features for model '{name}'."
                )
            feats.append(blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1))
        return torch.stack(feats, dim=0)  # [M,B,F]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Not used; forward is overridden to use meta_model.
        Implemented to satisfy BaseEnsemble API; returns arithmetic mean.
        """
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        member_feats = self._member_features(inputs)  # [M,B,F]
        M, B, feat_dim = member_feats.shape

        if self.combine == "mean":
            feat = member_feats.mean(dim=0)           # [B,F]
        else:
            feat = member_feats.permute(1, 0, 2).reshape(B, M * feat_dim)  # [B, M*F]

        logits = self.meta_model(feat)                # [B,C]
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# =========================
# tree-based stacking
# =========================

class TreeMetaEnsemble(BaseEnsemble):
    """
    Base class for stacking ensembles that delegate to a tree-based (non-PyTorch)
    meta-learner such as GradientBoosting/XGBoost/LightGBM.

    The meta-learner must implement `fit(X, y)` and `predict_proba(X)`.
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
        self._base_feature_mode, self._use_internal, self._internal_source = self._parse_feature_source(feature_source)
        if combine not in {"concat", "mean"}:
            raise ValueError("combine must be 'concat' or 'mean'.")
        if meta_model is None:
            raise ValueError("TreeMetaEnsemble requires a meta_model instance.")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for classification.")

        self.meta_model = meta_model
        self.feature_source = feature_source
        self.combine = combine
        self.num_classes = int(num_classes)
        self._is_fitted = False
        self._meta_classes: Optional[List[int]] = None

    @staticmethod
    def _parse_feature_source(source: str) -> Tuple[Optional[str], bool, str]:
        valid = {
            "logits": ("logits", False, "mean"),
            "probs": ("probs", False, "mean"),
            "internal": (None, True, "mean"),
            "pma_internal": (None, True, "pma"),
            "logits+internal": ("logits", True, "mean"),
            "logits+pma_internal": ("logits", True, "pma"),
            "probs+internal": ("probs", True, "mean"),
            "probs+pma_internal": ("probs", True, "pma"),
        }
        if source not in valid:
            raise ValueError(
                "feature_source must be one of "
                "'logits', 'probs', 'internal', 'pma_internal', "
                "'logits+internal', 'logits+pma_internal', 'probs+internal', or 'probs+pma_internal'."
            )
        return valid[source]

    def _member_features(self, inputs: TensorLike) -> torch.Tensor:
        """
        Collect logits/probabilities from each frozen base learner.
        Returns tensor of shape [M, B, F].
        """
        tuple_inputs = tuple(inputs) if isinstance(inputs, list) else inputs
        feats = []
        for name, module in self.models.items():
            prepared = self._prepare_inputs(module, tuple_inputs)
            out = module(prepared)
            logits = self._extract_logits(out)
            blocks = []
            if self._base_feature_mode == "probs":
                blocks.append(F.softmax(logits, dim=-1))
            elif self._base_feature_mode == "logits":
                blocks.append(logits)

            if self._use_internal:
                extra_dim = int(getattr(module, "meta_feature_dim", 0))
                extractor = getattr(module, "get_meta_features", None)
                if extra_dim <= 0 or extractor is None:
                    raise ValueError(
                        f"Model '{name}' does not expose encoder features "
                        f"required by feature_source='{self.feature_source}'."
                    )
                if self._internal_source == "pma":
                    internal_feat = extractor("pma")
                else:
                    internal_feat = extractor()
                if internal_feat is None:
                    raise RuntimeError(
                        f"Model '{name}' failed to cache encoder features for stacking. "
                        "Ensure its forward method stores them before returning."
                    )
                if internal_feat.ndim > 2:
                    internal_feat = internal_feat.reshape(internal_feat.shape[0], -1)
                if internal_feat.shape[-1] != extra_dim:
                    raise ValueError(
                        f"Model '{name}' reported meta_feature_dim={extra_dim} "
                        f"but produced features with size {internal_feat.shape[-1]}."
                    )
                blocks.append(internal_feat)

            if not blocks:
                raise RuntimeError(
                    f"feature_source '{self.feature_source}' produced no features for model '{name}'."
                )
            feats.append(blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1))
        return torch.stack(feats, dim=0)  # [M,B,F]

    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Unused placeholder; tree ensembles override forward but BaseEnsemble
        expects _combine to exist.
        """
        w = weights.view(-1, 1, 1)
        return (member_probs * w).sum(dim=0)

    def _build_feature_matrix(self, member_feats: torch.Tensor) -> torch.Tensor:
        """
        Collapse member feature tensor into [B, F] according to combine mode.
        """
        M, B, C = member_feats.shape
        if self.combine == "mean":
            return member_feats.mean(dim=0)  # [B,C]
        return member_feats.permute(1, 0, 2).reshape(B, M * C)  # [B, M*C]

    def fit_meta_model(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit the underlying meta-learner.
        Args:
            features: numpy array [N, F]
            targets: numpy array [N]
        """
        if features.ndim != 2:
            raise ValueError("features must be 2D [N, F].")
        if targets.ndim != 1:
            raise ValueError("targets must be 1D [N].")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching first dimension.")

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
            raise ValueError("meta_model.classes_ must be 1-dimensional.")

        self._meta_classes = [int(c) for c in classes.tolist()]
        max_seen = max(self._meta_classes, default=self.num_classes - 1)
        if max_seen >= self.num_classes:
            self.num_classes = max_seen + 1
        self._is_fitted = True

    def forward(self, inputs: TensorLike) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError("TreeMetaEnsemble meta-model is not fitted. Call fit_meta_model first.")

        member_feats = self._member_features(inputs)
        feat = self._build_feature_matrix(member_feats)  # [B,F]
        feat_np = feat.detach().cpu().numpy()

        probs_np = self.meta_model.predict_proba(feat_np)
        probs_np = np.asarray(probs_np)
        if probs_np.ndim == 1:
            # Convert binary probabilities [B] -> [B,2]
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

        probs = torch.from_numpy(full).to(member_feats.device)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        logits = torch.log(probs)
        return probs, logits


class GBMEnsemble(TreeMetaEnsemble):
    """
    Stacking ensemble whose meta-learner is sklearn's GradientBoostingClassifier.
    """

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
                    "GBMEnsemble requires scikit-learn. Install it via `pip install scikit-learn`."
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


class XGBoostEnsemble(TreeMetaEnsemble):
    """
    Stacking ensemble whose meta-learner is xgboost.XGBClassifier.
    """

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
                    "XGBoostEnsemble requires the xgboost package. "
                    "Install it via `pip install xgboost` (CPU) or the appropriate GPU build."
                ) from exc
            meta_kwargs = meta_kwargs or {}
            # Disable legacy label encoder warnings by default
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


class LightGBMEnsemble(TreeMetaEnsemble):
    """
    Stacking ensemble whose meta-learner is lightgbm.LGBMClassifier.
    """

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
                    "LightGBMEnsemble requires the lightgbm package. "
                    "Install it via `pip install lightgbm`."
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


# =========================
# boosting
# =========================

class AdaBoostEnsemble(BaseEnsemble):
    """
    Inference-time holder for an AdaBoost-style ensemble.
    This class is functionally identical to SoftVotingEnsemble at inference time.
    
    The 'weights' (alphas) and 'models' (weak learners) are 
    determined by an external sequential training loop.
    """
    def _combine(self, member_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # member_probs: [M,B,C], weights: [M]
        w = weights.view(-1, 1, 1)
        # Weighted arithmetic mean of member probabilities
        return (member_probs * w).sum(dim=0)  # [B,C]
