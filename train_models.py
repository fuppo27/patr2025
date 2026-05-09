#!/usr/bin/env python
# coding: utf-8

import argparse
import concurrent.futures
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

import main as M


CLASSIFICATION_ENSEMBLE_TYPES = [
    "soft_voting_ensemble",
    "stacking_ensemble",
    "gbm_ensemble",
    "xgboost_ensemble",
    "lightgbm_ensemble",
    "geometric_mean_ensemble",
    "median_ensemble",
    "trimmed_mean_ensemble",
]

ORDINAL_ENSEMBLE_TYPES = [
    "ordinal_soft_voting_ensemble",
    "ordinal_stacking_ensemble",
    "ordinal_gbm_ensemble",
    "ordinal_xgboost_ensemble",
    "ordinal_adaboost_ensemble",
    "ordinal_lightgbm_ensemble",
    "ordinal_geometric_mean_ensemble",
    "ordinal_median_ensemble",
    "ordinal_trimmed_mean_ensemble",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train all models for each pre-saved split.")
    parser.add_argument("--splits-path", type=str, default="result/splits/split_indices.json")
    parser.add_argument("--save-root", type=str, default="result/split_models")
    parser.add_argument("--batch-size", type=int, default=M.batch_size)
    parser.add_argument("--epochs", type=int, default=M.epochs)
    parser.add_argument("--lr", type=float, default=M.lr)
    parser.add_argument("--boosting-num-stages", type=int, default=5)
    parser.add_argument("--boosting-weak-epochs", type=int, default=3)
    parser.add_argument("--stacking-meta-epochs", type=int, default=5)
    parser.add_argument("--stacking-meta-lr", type=float, default=1e-3)
    parser.add_argument(
        "--meta-kfolds",
        type=int,
        default=5,
        help="K for OOF meta training inside each train split",
    )
    parser.add_argument(
        "--meta-seed",
        type=int,
        default=42,
        help="Random seed for OOF K-fold splitting",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of concurrent workers for non-ensemble training inside one split.",
    )
    parser.add_argument("--skip-ensembles", action="store_true")
    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
        help="Specify a single split ID to train. If not specified, all splits will be trained.",
    )
    return parser.parse_args()


def load_splits(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload.get("splits", [])
    if not splits:
        raise ValueError(f"No splits found in {path}")
    return splits


def save_checkpoint(model, model_name, is_ordinal, split_dir):
    subdir = "ordinal" if is_ordinal else "classification"
    save_dir = os.path.join(split_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in model_name)
    save_path = os.path.join(save_dir, f"{safe_name}.pt")

    payload = {
        "model_name": model_name,
        "is_ordinal": bool(is_ordinal),
        "class_name": model.__class__.__name__,
        "model_object": model,
    }
    torch.save(payload, save_path)
    print(f"Saved checkpoint: {save_path}")


def build_checkpoint_path(model_name, is_ordinal, split_dir):
    subdir = "ordinal" if is_ordinal else "classification"
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in model_name)
    return os.path.join(split_dir, subdir, f"{safe_name}.pt")


def try_load_checkpoint(model_name, is_ordinal, split_dir):
    ckpt_path = build_checkpoint_path(model_name, is_ordinal, split_dir)
    if not os.path.isfile(ckpt_path):
        return None

    try:
        try:
            payload = torch.load(ckpt_path, map_location=M.device, weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_path, map_location=M.device)
        model = payload.get("model_object")
        if model is None:
            print(f"Checkpoint exists but model_object missing: {ckpt_path}")
            return None
        model = model.to(M.device)
        model.eval()
        print(f"Reusing checkpoint (skip training): {ckpt_path}")
        return model
    except Exception as exc:
        print(f"Failed to load checkpoint (will retrain): {ckpt_path} ({exc})")
        return None


def build_loaders_for_split(dataset, train_idx, test_idx, model_type, batch_size):
    collate_fn = M.make_collate_fn(model_type)
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


def classification_class_weights(dataset, train_idx):
    y_train = [M.grade_to_label[dataset.raw[i]["grade"]] for i in train_idx]
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float).to(M.device)


def build_model(model_type, vocab_size, num_classes, type_vec_dim):
    if model_type == "set_transformer":
        model = M.SetTransformerClassifier(vocab_size=vocab_size, dim_in=M.embed_dim, num_classes=num_classes)
    elif model_type == "set_transformer_xy":
        model = M.SetTransformerClassifierXY(
            vocab_size=vocab_size,
            dim_in=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "set_transformer_additive":
        model = M.SetTransformerClassifierXYAdditive(
            vocab_size=vocab_size,
            feat_dim=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "deepset":
        model = M.DeepSetClassifier(vocab_size=vocab_size, dim_in=M.embed_dim, num_classes=num_classes)
    elif model_type == "deepset_xy":
        model = M.DeepSetClassifierXY(
            vocab_size=vocab_size,
            dim_in=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "deepset_xy_additive":
        model = M.DeepSetClassifierXYAdditive(
            vocab_size=vocab_size,
            feat_dim=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "set_transformer_ordinal":
        model = M.SetTransformerOrdinal(vocab_size=vocab_size, dim_in=M.embed_dim, num_classes=num_classes)
    elif model_type == "set_transformer_ordinal_xy":
        model = M.SetTransformerOrdinalXY(
            vocab_size=vocab_size,
            dim_in=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "set_transformer_ordinal_xy_additive":
        model = M.SetTransformerOrdinalXYAdditive(
            vocab_size=vocab_size,
            feat_dim=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "deepset_ordinal":
        model = M.DeepSetOrdinal(vocab_size=vocab_size, dim_in=M.embed_dim, num_classes=num_classes)
    elif model_type == "deepset_ordinal_xy":
        model = M.DeepSetOrdinalXY(
            vocab_size=vocab_size,
            dim_in=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    elif model_type == "deepset_ordinal_xy_additive":
        model = M.DeepSetOrdinalXYAdditive(
            vocab_size=vocab_size,
            feat_dim=M.embed_dim,
            num_classes=num_classes,
            type_vec_dim=type_vec_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def train_single_model(model_type, dataset, train_idx, test_idx, args):
    targets = [M.grade_to_label[item["grade"]] for item in dataset.raw]
    num_classes = len(np.unique(targets))
    vocab_size = len(M.hold_to_idx)
    type_vec_dim = len(M.type_to_idx)
    is_ordinal = model_type in M.ORDINAL_MODELS

    train_loader, test_loader = build_loaders_for_split(dataset, train_idx, test_idx, model_type, args.batch_size)
    model = build_model(model_type, vocab_size, num_classes, type_vec_dim).to(M.device)
    model.is_ordinal = is_ordinal
    model.num_classes = num_classes

    if is_ordinal:
        def criterion_fn(logits, y):
            return M.ordinal_logistic_loss(logits, y)
    else:
        class_weights = classification_class_weights(dataset, train_idx)
        criterion_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trained = M.train_model(
        model,
        train_loader,
        test_loader,
        criterion_fn,
        optimizer,
        args.epochs,
        is_ordinal=is_ordinal,
    )
    return trained, train_loader, test_loader, num_classes


def train_boosting_with_split(dataset, train_idx, test_idx, model_types, args):
    if isinstance(model_types, str):
        model_types = [model_types]
    if not model_types:
        raise ValueError("At least one base model type is required for boosting.")

    targets = [M.grade_to_label[item["grade"]] for item in dataset.raw]
    num_classes = len(np.unique(targets))
    vocab_size = len(M.hold_to_idx)
    type_vec_dim = len(M.type_to_idx)

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    collate_fns = {mtype: M.make_collate_fn(mtype) for mtype in set(model_types)}
    train_eval_loaders = {
        mtype: DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fns[mtype])
        for mtype in collate_fns
    }
    test_loaders = {
        mtype: DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fns[mtype])
        for mtype in collate_fns
    }

    eval_collate_type = next((m for m in model_types if m in M.XY_MODELS), model_types[0])
    final_train_loader = train_eval_loaders[eval_collate_type]
    final_test_loader = test_loaders[eval_collate_type]

    num_train_samples = len(train_idx)
    sample_weights = torch.full((num_train_samples,), 1.0 / num_train_samples, device=M.device)
    trained_models_list = []
    model_alphas = []

    for stage in range(args.boosting_num_stages):
        stage_model_type = model_types[stage % len(model_types)]
        sampler = WeightedRandomSampler(sample_weights.cpu(), num_train_samples, replacement=True)
        stage_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fns[stage_model_type],
        )

        model = build_model(stage_model_type, vocab_size, num_classes, type_vec_dim).to(M.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model = M.train_model(
            model,
            stage_loader,
            test_loaders[stage_model_type],
            criterion,
            optimizer,
            args.boosting_weak_epochs,
            is_ordinal=False,
        )

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in train_eval_loaders[stage_model_type]:
                inputs = tuple(x.to(M.device) for x in X)
                payload = inputs[0] if len(inputs) == 1 else inputs
                logits = model(payload)
                all_preds.append(logits.argmax(dim=1))
                all_targets.append(y.to(M.device))

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        is_incorrect = (all_preds != all_targets).float()
        err_m = (is_incorrect * sample_weights).sum()

        if err_m <= 0 or err_m >= (1.0 - 1.0 / num_classes):
            if err_m <= 0:
                trained_models_list.append((f"{stage_model_type}_boost_model_{stage}", model))
                model_alphas.append(1.0)
            break

        alpha_m = torch.log((1.0 - err_m) / err_m) + torch.log(torch.tensor(num_classes - 1.0, device=M.device))
        sample_weights *= torch.exp(alpha_m * is_incorrect)
        sample_weights /= sample_weights.sum()

        trained_models_list.append((f"{stage_model_type}_boost_model_{stage}", model))
        model_alphas.append(alpha_m.item())

    if not trained_models_list:
        raise RuntimeError("Boosting training failed, no models were trained.")

    final_ensemble = M.AdaBoostEnsemble(trained_models_list, weights=model_alphas, freeze_members=True).to(M.device)
    final_ensemble.is_ordinal = False
    final_ensemble.num_classes = num_classes
    return final_ensemble, final_train_loader, final_test_loader, num_classes


def _run_non_ensemble_task(task, dataset, train_idx, test_idx, args):
    task_kind = task["kind"]
    model_name = task["name"]

    if task_kind == "classification_base":
        cached = try_load_checkpoint(model_name, False, args._split_dir)
        if cached is not None:
            return {
                "kind": task_kind,
                "name": model_name,
                "model": cached,
                "is_ordinal": False,
                "num_classes": len(M.grade_to_label),
                "loaded_from_checkpoint": True,
            }

        model, _train_loader, _test_loader, num_classes = train_single_model(
            model_name,
            dataset,
            train_idx,
            test_idx,
            args,
        )
        return {
            "kind": task_kind,
            "name": model_name,
            "model": model,
            "is_ordinal": False,
            "num_classes": num_classes,
            "loaded_from_checkpoint": False,
        }

    if task_kind == "classification_boosting":
        model, _train_loader, _test_loader, num_classes = train_boosting_with_split(
            dataset,
            train_idx,
            test_idx,
            task["base_list"],
            args,
        )
        return {
            "kind": task_kind,
            "name": model_name,
            "model": model,
            "is_ordinal": False,
            "num_classes": num_classes,
            "loaded_from_checkpoint": False,
        }

    if task_kind == "ordinal_base":
        cached = try_load_checkpoint(model_name, True, args._split_dir)
        if cached is not None:
            return {
                "kind": task_kind,
                "name": model_name,
                "model": cached,
                "is_ordinal": True,
                "num_classes": len(M.grade_to_label),
                "loaded_from_checkpoint": True,
            }

        model, _train_loader, _test_loader, num_classes = train_single_model(
            model_name,
            dataset,
            train_idx,
            test_idx,
            args,
        )
        return {
            "kind": task_kind,
            "name": model_name,
            "model": model,
            "is_ordinal": True,
            "num_classes": num_classes,
            "loaded_from_checkpoint": False,
        }

    raise ValueError(f"Unknown task kind: {task_kind}")


def train_non_ensemble_parallel(tasks, dataset, train_idx, test_idx, split_dir, args):

    args._split_dir = split_dir

    workers = max(1, int(args.parallel_workers))
    trained_class_models = {}
    trained_ordinal_models = {}
    num_classes = len(M.grade_to_label)

    if workers == 1:
        for task in tasks:
            print(f"[non-ensemble] training {task['name']}")
            result = _run_non_ensemble_task(task, dataset, train_idx, test_idx, args)
            if not result.get("loaded_from_checkpoint", False):
                save_checkpoint(result["model"], result["name"], result["is_ordinal"], split_dir)
            if result["kind"] == "classification_base":
                trained_class_models[result["name"]] = result["model"]
            elif result["kind"] == "ordinal_base":
                trained_ordinal_models[result["name"]] = result["model"]
            if result["num_classes"] is not None:
                num_classes = int(result["num_classes"])
        return trained_class_models, trained_ordinal_models, num_classes

    print(f"[non-ensemble] parallel training with workers={workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_run_non_ensemble_task, task, dataset, train_idx, test_idx, args): task
            for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            if not result.get("loaded_from_checkpoint", False):
                save_checkpoint(result["model"], result["name"], result["is_ordinal"], split_dir)

            if result["kind"] == "classification_base":
                trained_class_models[result["name"]] = result["model"]
            elif result["kind"] == "ordinal_base":
                trained_ordinal_models[result["name"]] = result["model"]

            if result["num_classes"] is not None:
                num_classes = int(result["num_classes"])

            print(f"[non-ensemble] finished {task['name']}")

    return trained_class_models, trained_ordinal_models, num_classes


def _adaboost_style_alphas(base_items, dataset, eval_idx, batch_size, num_classes):
    eval_subset = Subset(dataset, eval_idx)
    loader_cache = {}
    alphas = []
    max_err = 1.0 - (1.0 / max(2, int(num_classes)))
    for model_name, model in base_items:
        if model_name not in loader_cache:
            collate_fn = M.make_collate_fn(model_name)
            loader_cache[model_name] = DataLoader(
                eval_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        eval_loader = loader_cache[model_name]
        strict_acc, _loose_acc, _y_true, _y_pred = M.compute_accuracy(model, eval_loader, M.device)
        err = 1.0 - (float(strict_acc) / 100.0)
        err = min(max(err, 1e-6), max_err - 1e-6)
        alpha = float(np.log((1.0 - err) / err) + np.log(max(1, int(num_classes) - 1)))
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 1e-3
        alphas.append(alpha)
    return alphas


def build_adaboost_style_classification(base_items, dataset, eval_idx, batch_size, num_classes):
    alphas = _adaboost_style_alphas(base_items, dataset, eval_idx, batch_size, num_classes)
    ensemble = M.AdaBoostEnsemble(list(base_items), weights=alphas, freeze_members=True).to(M.device)
    ensemble.is_ordinal = False
    ensemble.num_classes = int(num_classes)
    return ensemble


def build_adaboost_style_ordinal(base_items, dataset, eval_idx, batch_size, num_classes):
    alphas = _adaboost_style_alphas(base_items, dataset, eval_idx, batch_size, num_classes)
    ensemble = M.OrdinalAdaBoostEnsemble(list(base_items), weights=alphas, freeze_members=True).to(M.device)
    ensemble.is_ordinal = True
    ensemble.num_classes = int(num_classes)
    return ensemble


def _resolve_oof_splits(train_idx, dataset, requested_k, seed):
    labels = np.array([M.grade_to_label[dataset.raw[i]["grade"]] for i in train_idx], dtype=np.int64)
    if labels.size < 2:
        raise ValueError("Not enough samples for OOF meta training.")

    class_counts = np.bincount(labels)
    nonzero_counts = class_counts[class_counts > 0]
    min_class_count = int(nonzero_counts.min()) if nonzero_counts.size else 1
    k = min(int(requested_k), min_class_count)
    if k < 2:
        raise ValueError(
            "OOF meta training requires at least 2 samples in every present class of train_idx."
        )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    rel_indices = np.arange(len(train_idx))
    folds = []
    for fold_id, (tr_rel, va_rel) in enumerate(skf.split(rel_indices, labels), start=1):
        fold_train_idx = [train_idx[int(i)] for i in tr_rel]
        fold_val_idx = [train_idx[int(i)] for i in va_rel]
        folds.append((fold_id, fold_train_idx, fold_val_idx))
    return folds


def _build_oof_feature_loader(dataset, indices, batch_size):
    collate_fn = M.make_collate_fn("set_transformer_xy")
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def _extract_stacking_features(ensemble_model, dataloader):
    feat_blocks = []
    target_blocks = []
    ensemble_model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            inputs = tuple(x.to(M.device) for x in X)
            member_feats = ensemble_model._member_features(inputs)
            members, batch, feat_dim = member_feats.shape
            if ensemble_model.combine == "mean":
                feat = member_feats.mean(dim=0)
            else:
                feat = member_feats.permute(1, 0, 2).reshape(batch, members * feat_dim)
            feat_blocks.append(feat.detach().cpu().numpy())
            target_blocks.append(y.detach().cpu().numpy())
    if not feat_blocks:
        raise RuntimeError("No OOF samples were produced for stacking meta training.")
    return np.concatenate(feat_blocks, axis=0), np.concatenate(target_blocks, axis=0)


def _extract_tree_features(ensemble_model, dataloader):
    feat_blocks = []
    target_blocks = []
    ensemble_model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            inputs = tuple(x.to(M.device) for x in X)
            member_feats = ensemble_model._member_features(inputs)
            feat = ensemble_model._build_feature_matrix(member_feats)
            feat_blocks.append(feat.detach().cpu().numpy())
            target_blocks.append(y.detach().cpu().numpy())
    if not feat_blocks:
        raise RuntimeError("No OOF samples were produced for tree meta training.")
    return np.concatenate(feat_blocks, axis=0), np.concatenate(target_blocks, axis=0)


def _fit_stacking_meta_from_arrays(stacking_model, features, targets, epochs, lr, is_ordinal=False):
    in_dim = int(features.shape[1])
    out_dim = int(stacking_model.num_classes - 1) if is_ordinal else int(stacking_model.num_classes)
    stacking_model.meta_model = nn.Linear(in_dim, out_dim).to(M.device)

    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=M.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(stacking_model.meta_model.parameters(), lr=lr)

    stacking_model.meta_model.train()
    for _epoch in range(max(1, int(epochs))):
        for xb, yb in loader:
            xb = xb.to(M.device)
            yb = yb.to(M.device)
            logits = stacking_model.meta_model(xb)
            if is_ordinal:
                loss = M.ordinal_logistic_loss(logits, yb)
            else:
                loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    stacking_model.meta_model.eval()


def _instantiate_classification_meta_ensemble(ensemble_name, base_items, num_classes, feature_source):
    if ensemble_name == "stacking_ensemble":
        placeholder_dim = max(1, len(base_items) * num_classes)
        meta_model = nn.Linear(placeholder_dim, num_classes).to(M.device)
        return M.StackingEnsemble(
            base_items,
            weights=None,
            freeze_members=True,
            meta_model=meta_model,
            feature_source=feature_source,
            combine="concat",
        ).to(M.device)
    if ensemble_name == "gbm_ensemble":
        return M.GBMEnsemble(
            base_items,
            weights=None,
            freeze_members=True,
            num_classes=num_classes,
            feature_source="logits",
            combine="concat",
            meta_kwargs={"random_state": 42},
        ).to(M.device)
    if ensemble_name == "xgboost_ensemble":
        return M.XGBoostEnsemble(
            base_items,
            weights=None,
            freeze_members=True,
            num_classes=num_classes,
            feature_source="logits",
            combine="concat",
            meta_kwargs={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
        ).to(M.device)
    if ensemble_name == "lightgbm_ensemble":
        return M.LightGBMEnsemble(
            base_items,
            weights=None,
            freeze_members=True,
            num_classes=num_classes,
            feature_source="logits",
            combine="concat",
            meta_kwargs={"n_estimators": 300, "learning_rate": 0.05, "max_depth": -1},
        ).to(M.device)
    raise ValueError(f"Unsupported classification meta ensemble: {ensemble_name}")


def _instantiate_ordinal_meta_ensemble(ensemble_name, base_items, num_classes, feature_source):
    if ensemble_name == "ordinal_stacking_ensemble":
        placeholder_dim = max(1, len(base_items) * (num_classes - 1))
        meta_model = nn.Linear(placeholder_dim, num_classes - 1).to(M.device)
        return M.OrdinalStackingEnsemble(
            base_items,
            num_classes=num_classes,
            weights=None,
            freeze_members=True,
            meta_model=meta_model,
            feature_source=feature_source,
            combine="concat",
        ).to(M.device)
    if ensemble_name == "ordinal_gbm_ensemble":
        return M.OrdinalGBMEnsemble(
            base_items,
            num_classes=num_classes,
            weights=None,
            freeze_members=True,
            feature_source="both",
            combine="concat",
        ).to(M.device)
    if ensemble_name == "ordinal_xgboost_ensemble":
        return M.OrdinalXGBoostEnsemble(
            base_items,
            num_classes=num_classes,
            weights=None,
            freeze_members=True,
            feature_source="both",
            combine="concat",
        ).to(M.device)
    if ensemble_name == "ordinal_lightgbm_ensemble":
        return M.OrdinalLightGBMEnsemble(
            base_items,
            num_classes=num_classes,
            weights=None,
            freeze_members=True,
            feature_source="both",
            combine="concat",
        ).to(M.device)
    raise ValueError(f"Unsupported ordinal meta ensemble: {ensemble_name}")


def _train_meta_ensemble_with_oof(
    ensemble_name,
    final_base_items,
    dataset,
    train_idx,
    args,
    num_classes,
    is_ordinal,
    feature_source,
):
    folds = _resolve_oof_splits(train_idx, dataset, args.meta_kfolds, args.meta_seed)
    all_features = []
    all_targets = []
    model_names = [name for name, _ in final_base_items]

    for fold_id, fold_train_idx, fold_val_idx in folds:
        print(
            f"[meta-oof] {ensemble_name} fold {fold_id}/{len(folds)} "
            f"train={len(fold_train_idx)} val={len(fold_val_idx)}"
        )
        fold_items = []
        for model_name in model_names:
            fold_model, _tr, _va, _nc = train_single_model(
                model_name,
                dataset,
                fold_train_idx,
                fold_val_idx,
                args,
            )
            fold_items.append((model_name, fold_model))

        if is_ordinal:
            fold_ensemble = _instantiate_ordinal_meta_ensemble(
                ensemble_name,
                fold_items,
                num_classes,
                feature_source,
            )
        else:
            fold_ensemble = _instantiate_classification_meta_ensemble(
                ensemble_name,
                fold_items,
                num_classes,
                feature_source,
            )

        val_loader = _build_oof_feature_loader(dataset, fold_val_idx, args.batch_size)
        if "stacking" in ensemble_name:
            fold_features, fold_targets = _extract_stacking_features(fold_ensemble, val_loader)
        else:
            fold_features, fold_targets = _extract_tree_features(fold_ensemble, val_loader)
        all_features.append(fold_features)
        all_targets.append(fold_targets)

    oof_features = np.concatenate(all_features, axis=0)
    oof_targets = np.concatenate(all_targets, axis=0)

    if is_ordinal:
        final_ensemble = _instantiate_ordinal_meta_ensemble(
            ensemble_name,
            list(final_base_items),
            num_classes,
            feature_source,
        )
    else:
        final_ensemble = _instantiate_classification_meta_ensemble(
            ensemble_name,
            list(final_base_items),
            num_classes,
            feature_source,
        )

    if "stacking" in ensemble_name:
        _fit_stacking_meta_from_arrays(
            final_ensemble,
            oof_features,
            oof_targets,
            epochs=args.stacking_meta_epochs,
            lr=args.stacking_meta_lr,
            is_ordinal=is_ordinal,
        )
    else:
        final_ensemble.fit_meta_model(oof_features, oof_targets)

    return final_ensemble


def train_for_one_split(dataset, split_record, args):
    split_id = int(split_record.get("split_id"))
    train_idx = [int(x) for x in split_record["train_idx"]]
    test_idx = [int(x) for x in split_record["test_idx"]]
    split_dir = os.path.join(args.save_root, f"split_{split_id:03d}")
    os.makedirs(split_dir, exist_ok=True)

    print(f"================ split {split_id} ================")
    print(f"train={len(train_idx)}, test={len(test_idx)}")

    classification_base_tasks = [
        {"kind": "classification_base", "name": model_type}
        for model_type in M.BASE_MODEL_TYPES
    ]
    ordinal_base_tasks = [
        {"kind": "ordinal_base", "name": model_type}
        for model_type in M.ORDINAL_BASE_MODEL_TYPES
    ]

    print("[stage] classification base models")
    trained_class_models, _, num_classes = train_non_ensemble_parallel(
        classification_base_tasks,
        dataset,
        train_idx,
        test_idx,
        split_dir,
        args,
    )

    if not args.skip_ensembles:
        collate_fn = M.make_collate_fn("set_transformer_xy")
        ensemble_train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        class_groups = [
            ("all", list(trained_class_models.items())),
            ("set_transformer", [(k, v) for k, v in trained_class_models.items() if "set_transformer" in k]),
            ("deepset", [(k, v) for k, v in trained_class_models.items() if "deepset" in k]),
        ]
        for group_name, items in class_groups:
            if not items:
                continue
            print(f"[classification] training ensemble group={group_name}")
            simple_types = [
                "soft_voting_ensemble",
                "geometric_mean_ensemble",
                "median_ensemble",
                "trimmed_mean_ensemble",
            ]
            ensembles = M.build_ensemble_models(
                simple_types,
                items,
                ensemble_weights=None,
                num_classes=num_classes,
                device=M.device,
                train_loader=ensemble_train_loader,
                stacking_meta_epochs=args.stacking_meta_epochs,
                stacking_meta_lr=args.stacking_meta_lr,
                label_suffix=f"_{group_name}",
            )

            meta_types = ["stacking_ensemble", "gbm_ensemble", "xgboost_ensemble", "lightgbm_ensemble"]
            for meta_name in meta_types:
                try:
                    meta_model = _train_meta_ensemble_with_oof(
                        meta_name,
                        items,
                        dataset,
                        train_idx,
                        args,
                        num_classes=num_classes,
                        is_ordinal=False,
                        feature_source="logits+internal",
                    )
                    ensembles[f"{meta_name}_{group_name}"] = meta_model
                except ImportError as exc:
                    print(f"Skipping {meta_name}_{group_name}: {exc}")
                except ValueError as exc:
                    print(f"Skipping {meta_name}_{group_name}: {exc}")

            if group_name == "set_transformer":
                try:
                    pma_meta_model = _train_meta_ensemble_with_oof(
                        "stacking_ensemble",
                        items,
                        dataset,
                        train_idx,
                        args,
                        num_classes=num_classes,
                        is_ordinal=False,
                        feature_source="logits+pma_internal",
                    )
                    ensembles["stacking_ensemble_set_transformer_pma_internal"] = pma_meta_model
                except ImportError as exc:
                    print(f"Skipping stacking_ensemble_set_transformer_pma_internal: {exc}")
                except ValueError as exc:
                    print(f"Skipping stacking_ensemble_set_transformer_pma_internal: {exc}")

            adaboost_name_by_group = {
                "all": "adaboost_all",
                "set_transformer": "adaboost_set_transformer",
                "deepset": "adaboost_deepset",
            }
            adaboost_name = adaboost_name_by_group.get(group_name)
            if adaboost_name:
                ada_model = build_adaboost_style_classification(
                    items,
                    dataset,
                    train_idx,
                    args.batch_size,
                    num_classes,
                )
                ensembles[adaboost_name] = ada_model

            for name, model in ensembles.items():
                save_checkpoint(model, name, False, split_dir)

    print("[stage] ordinal base models")
    _, trained_ordinal_models, _ = train_non_ensemble_parallel(
        ordinal_base_tasks,
        dataset,
        train_idx,
        test_idx,
        split_dir,
        args,
    )

    if not args.skip_ensembles:
        collate_fn = M.make_collate_fn("set_transformer_xy")
        ensemble_train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        ord_types = []
        for t in ORDINAL_ENSEMBLE_TYPES:
            if t not in ord_types:
                ord_types.append(t)

        use_adaboost_style = "ordinal_adaboost_ensemble" in ord_types
        non_boosting = [t for t in ord_types if t != "ordinal_adaboost_ensemble"]
        ord_groups = [
            ("all", list(trained_ordinal_models.items())),
            ("set_transformer", [(k, v) for k, v in trained_ordinal_models.items() if "set_transformer" in k]),
            ("deepset", [(k, v) for k, v in trained_ordinal_models.items() if "deepset" in k]),
        ]

        for group_name, items in ord_groups:
            if not items:
                continue
            print(f"[ordinal] training ensemble group={group_name}")
            feature_src = "logits+internal" if group_name == "set_transformer" else "logits"

            simple_ord_types = [
                "ordinal_soft_voting_ensemble",
                "ordinal_geometric_mean_ensemble",
                "ordinal_median_ensemble",
                "ordinal_trimmed_mean_ensemble",
            ]
            ensembles = M.build_ordinal_ensemble_models(
                [t for t in simple_ord_types if t in non_boosting],
                items,
                num_classes=len(M.grade_to_label),
                device=M.device,
                train_loader=ensemble_train_loader,
                stacking_meta_epochs=args.stacking_meta_epochs,
                stacking_meta_lr=args.stacking_meta_lr,
                stacking_feature_source=feature_src,
                ensemble_weights=None,
                label_suffix=f"_{group_name}",
            )
            for meta_name in [
                "ordinal_stacking_ensemble",
                "ordinal_gbm_ensemble",
                "ordinal_xgboost_ensemble",
                "ordinal_lightgbm_ensemble",
            ]:
                if meta_name not in non_boosting:
                    continue
                try:
                    meta_model = _train_meta_ensemble_with_oof(
                        meta_name,
                        items,
                        dataset,
                        train_idx,
                        args,
                        num_classes=len(M.grade_to_label),
                        is_ordinal=True,
                        feature_source=feature_src,
                    )
                    ensembles[f"{meta_name}_{group_name}"] = meta_model
                except ImportError as exc:
                    print(f"Skipping {meta_name}_{group_name}: {exc}")
                except ValueError as exc:
                    print(f"Skipping {meta_name}_{group_name}: {exc}")

            for name, model in ensembles.items():
                save_checkpoint(model, name, True, split_dir)

            if use_adaboost_style:
                ada_model = build_adaboost_style_ordinal(
                    items,
                    dataset,
                    train_idx,
                    args.batch_size,
                    len(M.grade_to_label),
                )
                save_checkpoint(ada_model, f"ordinal_adaboost_ensemble_{group_name}", True, split_dir)


def main():
    args = parse_args()
    M.batch_size = args.batch_size
    M.epochs = args.epochs
    M.lr = args.lr

    print(f"Using device: {M.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    dataset = M.load_dataset(
        M.json_path,
        M.hold_to_idx,
        M.grade_to_label,
        M.hold_difficulty,
        M.type_to_idx,
        M.hold_to_coord,
    )
    splits = load_splits(args.splits_path)
    os.makedirs(args.save_root, exist_ok=True)

    if args.split_id is not None:
        target_split = None
        for split_record in splits:
            if int(split_record.get("split_id")) == args.split_id:
                target_split = split_record
                break
        if target_split is None:
            print(f"Error: Split ID {args.split_id} not found in splits.")
            return
        train_for_one_split(dataset, target_split, args)
        print(f"Training for split {args.split_id} is complete.")
    else:
        for split_record in splits:
            train_for_one_split(dataset, split_record, args)
        print("All split-based trainings are complete.")


if __name__ == "__main__":
    main()
