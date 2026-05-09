#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import main as M


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on split-specific test sets.")
    parser.add_argument("--splits-path", type=str, default="result/splits/split_indices.json")
    parser.add_argument("--model-root", type=str, default="result/split_models")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-target",
        type=str,
        choices=("all", "classification", "ordinal"),
        default="all",
        help="Select model family to evaluate",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for ordinal cumulative outputs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="result/inference_split_iterations.csv",
        help="Per-split per-model metrics",
    )
    parser.add_argument(
        "--output-excel",
        type=str,
        default="result/inference_split_summary.xlsx",
        help="Excel output path",
    )
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload.get("splits", [])
    if not splits:
        raise ValueError(f"No splits found in {path}")
    return splits


def infer_collate_model_type(model_name, model_obj):
    lower_name = model_name.lower()
    class_name = model_obj.__class__.__name__.lower()

    is_ordinal = (
        ("ordinal" in lower_name)
        or ("ordinal" in class_name)
        or bool(getattr(model_obj, "is_ordinal", False))
    )
    uses_xy = ("_xy" in lower_name) or ("additive" in lower_name) or ("xy" in class_name)

    members = getattr(model_obj, "models", None)
    if members is not None:
        try:
            for member in members.values():
                if bool(getattr(member, "expects_tuple_input", False)):
                    uses_xy = True
                    break
        except Exception:
            pass

    if is_ordinal and uses_xy:
        return "set_transformer_ordinal_xy"
    if is_ordinal:
        return "set_transformer_ordinal"
    if uses_xy:
        return "set_transformer_xy"
    return "set_transformer"


def is_ordinal_model(model_name, model_obj):
    lower_name = model_name.lower()
    class_name = model_obj.__class__.__name__.lower()
    return (
        ("ordinal" in lower_name)
        or ("ordinal" in class_name)
        or bool(getattr(model_obj, "is_ordinal", False))
    )


def load_split_checkpoints(split_dir, device, eval_target="all"):
    checkpoints = []
    if eval_target == "all":
        subdirs = ("classification", "ordinal")
    elif eval_target == "classification":
        subdirs = ("classification",)
    else:
        subdirs = ("ordinal",)

    for subdir in subdirs:
        target = os.path.join(split_dir, subdir)
        if not os.path.isdir(target):
            continue
        for pt_path in sorted(glob.glob(os.path.join(target, "*.pt"))):
            model_name = os.path.splitext(os.path.basename(pt_path))[0]
            try:
                try:
                    payload = torch.load(pt_path, map_location=device, weights_only=False)
                except TypeError:
                    payload = torch.load(pt_path, map_location=device)
                model = payload.get("model_object")
                if model is None:
                    print(f"Skipping {pt_path}: model_object not found")
                    continue
                model = model.to(device)
                model.eval()
                checkpoints.append((model_name, model))
            except Exception as exc:
                print(f"Failed to load {pt_path}: {exc}")
    return checkpoints


def predict(model, dataloader, device, is_ordinal=False, decision_threshold=0.5):
    y_true, y_pred = [], []
    probs_all = []
    targets_all = []
    with torch.inference_mode():
        for X, y in dataloader:
            inputs = tuple(x.to(device) for x in X)
            y = y.to(device)
            payload = inputs[0] if len(inputs) == 1 else inputs
            outputs = model(payload)

            if isinstance(outputs, tuple):
                if is_ordinal:
                    probs = outputs[0]
                    preds = M.cumulative_to_labels(probs, threshold=decision_threshold)
                    probs_all.append(probs.detach().cpu())
                    targets_all.append(y.detach().cpu())
                else:
                    probs = outputs[0]
                    preds = probs.argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)

            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    cumulative_threshold_acc = None
    per_threshold_acc_pct = None
    if is_ordinal and probs_all:
        probs_tensor = torch.cat(probs_all, dim=0)
        targets_tensor = torch.cat(targets_all, dim=0)
        per_threshold_acc = M.threshold_accuracy(
            probs_tensor, targets_tensor, threshold=decision_threshold
        )
        per_threshold_acc_pct = (per_threshold_acc.cpu().numpy() * 100.0).tolist()
        cumulative_threshold_acc = float(per_threshold_acc.mean().item() * 100.0)

    return np.asarray(y_true), np.asarray(y_pred), cumulative_threshold_acc, per_threshold_acc_pct


def compute_metrics(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        return 0.0, 0.0
    strict = float((y_true == y_pred).sum() * 100.0 / total)
    loose = float((np.abs(y_true - y_pred) <= 1).sum() * 100.0 / total)
    return strict, loose


def collect_classwise(y_true, y_pred, split_id, model_name):
    records = []
    classes = sorted(np.unique(y_true))
    for class_idx in classes:
        mask = (y_true == class_idx)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float((y_pred[mask] == y_true[mask]).mean() * 100.0)
        records.append(
            {
                "Split": split_id,
                "Model": model_name,
                "Class Index": int(class_idx),
                "Class": M.label_to_grade.get(int(class_idx), f"Class_{int(class_idx)}"),
                "Class Accuracy (%)": acc,
                "Num Samples": count,
            }
        )
    return records


def summarize_metrics(df):
    grouped = (
        df.groupby(["Model", "Primary Metric"])[
            ["Primary Accuracy (%)", "+-1 Grade Accuracy (%)"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "Model",
        "Primary Metric",
        "Primary Mean (%)",
        "Primary Std (%)",
        "+-1 Mean (%)",
        "+-1 Std (%)",
    ]
    grouped["Primary Std (%)"] = grouped["Primary Std (%)"].fillna(0.0)
    grouped["+-1 Std (%)"] = grouped["+-1 Std (%)"].fillna(0.0)
    return grouped.sort_values(["Model", "Primary Metric"]).reset_index(drop=True)


def summarize_classwise(df):
    summary = (
        df.groupby(["Model", "Class", "Class Index"])["Class Accuracy (%)"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "Model",
        "Class",
        "Class Index",
        "Class Accuracy Mean (%)",
        "Class Accuracy Std (%)",
        "Num Splits",
    ]
    summary["Class Accuracy Std (%)"] = summary["Class Accuracy Std (%)"].fillna(0.0)
    return summary.sort_values(["Model", "Class Index"]).reset_index(drop=True)


def collect_ordinal_threshold_records(per_threshold_acc_pct, split_id, model_name):
    records = []
    if not per_threshold_acc_pct:
        return records

    for idx, acc in enumerate(per_threshold_acc_pct):
        grade = M.label_to_grade.get(int(idx), f"label_{int(idx)}")
        records.append(
            {
                "Split": split_id,
                "Model": model_name,
                "Threshold Index": int(idx),
                "Threshold": f"P(>{grade})",
                "Threshold Accuracy (%)": float(acc),
            }
        )
    return records


def summarize_ordinal_thresholds(df):
    summary = (
        df.groupby(["Model", "Threshold", "Threshold Index"])["Threshold Accuracy (%)"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "Model",
        "Threshold",
        "Threshold Index",
        "Threshold Accuracy Mean (%)",
        "Threshold Accuracy Std (%)",
        "Num Splits",
    ]
    summary["Threshold Accuracy Std (%)"] = summary["Threshold Accuracy Std (%)"].fillna(0.0)
    return summary.sort_values(["Model", "Threshold Index"]).reset_index(drop=True)


def run_inference_only(
    splits_path="result/splits/split_indices.json",
    model_root="result/split_models",
    batch_size=16,
    eval_target="all",
    decision_threshold=0.5,
    output_csv="result/inference_split_iterations.csv",
    output_excel="result/inference_split_summary.xlsx",
    **legacy_kwargs,
):
    if legacy_kwargs:
        print(
            "Warning: legacy arguments were provided and ignored in split-based inference mode: "
            f"{sorted(legacy_kwargs.keys())}"
        )

    device = get_device()
    print(f"Using device: {device}")

    dataset = M.load_dataset(
        M.json_path,
        M.hold_to_idx,
        M.grade_to_label,
        M.hold_difficulty,
        M.type_to_idx,
        M.hold_to_coord,
    )

    splits = load_splits(splits_path)
    records = []
    classwise_records = []
    ordinal_threshold_records = []

    for split in splits:
        split_id = int(split.get("split_id"))
        test_idx = [int(x) for x in split.get("test_idx", [])]
        split_dir = os.path.join(model_root, f"split_{split_id:03d}")

        if not os.path.isdir(split_dir):
            print(f"Skipping split {split_id}: model directory not found -> {split_dir}")
            continue

        checkpoints = load_split_checkpoints(split_dir, device, eval_target=eval_target)
        if not checkpoints:
            print(
                f"Skipping split {split_id}: no checkpoints found "
                f"for eval_target={eval_target}"
            )
            continue

        print(f"---------- split {split_id}: evaluating {len(checkpoints)} models ----------")
        loader_cache = {}
        for model_name, model in checkpoints:
            ordinal_flag = is_ordinal_model(model_name, model)
            collate_model_type = infer_collate_model_type(model_name, model)
            if collate_model_type not in loader_cache:
                collate_fn = M.make_collate_fn(collate_model_type)
                loader_cache[collate_model_type] = DataLoader(
                    Subset(dataset, test_idx),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )

            test_loader = loader_cache[collate_model_type]
            y_true, y_pred, cumulative_threshold_acc, per_threshold_acc_pct = predict(
                model,
                test_loader,
                device,
                is_ordinal=ordinal_flag,
                decision_threshold=decision_threshold,
            )
            strict_acc, loose_acc = compute_metrics(y_true, y_pred)
            primary_metric = (
                "Cumulative Threshold Accuracy (%)"
                if ordinal_flag
                else "Strict Accuracy (%)"
            )
            primary_acc = cumulative_threshold_acc if ordinal_flag else strict_acc

            records.append(
                {
                    "Split": split_id,
                    "Model": model_name,
                    "Primary Metric": primary_metric,
                    "Primary Accuracy (%)": primary_acc,
                    "Strict Accuracy (%)": strict_acc,
                    "Cumulative Threshold Accuracy (%)": cumulative_threshold_acc,
                    "+-1 Grade Accuracy (%)": loose_acc,
                    "Num Samples": int(len(y_true)),
                }
            )
            if not ordinal_flag:
                classwise_records.extend(
                    collect_classwise(y_true, y_pred, split_id, model_name)
                )
            else:
                ordinal_threshold_records.extend(
                    collect_ordinal_threshold_records(
                        per_threshold_acc_pct, split_id, model_name
                    )
                )

    if not records:
        print("No inference records collected.")
        return None, None

    df_iter = pd.DataFrame(records)
    df_summary = summarize_metrics(df_iter)
    if classwise_records:
        df_classwise_iter = pd.DataFrame(classwise_records)
        df_classwise_summary = summarize_classwise(df_classwise_iter)
    else:
        df_classwise_iter = pd.DataFrame(
            columns=[
                "Split",
                "Model",
                "Class Index",
                "Class",
                "Class Accuracy (%)",
                "Num Samples",
            ]
        )
        df_classwise_summary = pd.DataFrame(
            columns=[
                "Model",
                "Class",
                "Class Index",
                "Class Accuracy Mean (%)",
                "Class Accuracy Std (%)",
                "Num Splits",
            ]
        )

    if ordinal_threshold_records:
        df_ordinal_threshold_iter = pd.DataFrame(ordinal_threshold_records)
        df_ordinal_threshold_summary = summarize_ordinal_thresholds(df_ordinal_threshold_iter)
    else:
        df_ordinal_threshold_iter = pd.DataFrame(
            columns=[
                "Split",
                "Model",
                "Threshold Index",
                "Threshold",
                "Threshold Accuracy (%)",
            ]
        )
        df_ordinal_threshold_summary = pd.DataFrame(
            columns=[
                "Model",
                "Threshold",
                "Threshold Index",
                "Threshold Accuracy Mean (%)",
                "Threshold Accuracy Std (%)",
                "Num Splits",
            ]
        )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_iter.to_csv(output_csv, index=False)

    out_root, out_ext = os.path.splitext(output_csv)
    classwise_iter_csv = f"{out_root}_classwise{out_ext or '.csv'}"
    classwise_summary_csv = f"{out_root}_classwise_summary{out_ext or '.csv'}"
    ordinal_threshold_iter_csv = f"{out_root}_ordinal_thresholds{out_ext or '.csv'}"
    ordinal_threshold_summary_csv = (
        f"{out_root}_ordinal_thresholds_summary{out_ext or '.csv'}"
    )
    df_classwise_iter.to_csv(classwise_iter_csv, index=False)
    df_classwise_summary.to_csv(classwise_summary_csv, index=False)
    df_ordinal_threshold_iter.to_csv(ordinal_threshold_iter_csv, index=False)
    df_ordinal_threshold_summary.to_csv(ordinal_threshold_summary_csv, index=False)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_iter.to_excel(writer, sheet_name="iterations", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_classwise_iter.to_excel(writer, sheet_name="classwise_iterations", index=False)
        df_classwise_summary.to_excel(writer, sheet_name="classwise_summary", index=False)
        df_ordinal_threshold_iter.to_excel(
            writer, sheet_name="ordinal_threshold_iterations", index=False
        )
        df_ordinal_threshold_summary.to_excel(
            writer, sheet_name="ordinal_threshold_summary", index=False
        )

    print(f"Saved iteration records: {output_csv}")
    print(f"Saved summary: {output_excel}")
    print(f"Saved classwise iteration records: {classwise_iter_csv}")
    print(f"Saved classwise summary records: {classwise_summary_csv}")
    print(f"Saved ordinal threshold iteration records: {ordinal_threshold_iter_csv}")
    print(f"Saved ordinal threshold summary records: {ordinal_threshold_summary_csv}")
    print(df_summary)

    return df_iter, df_summary


def main():
    args = parse_args()
    run_inference_only(
        splits_path=args.splits_path,
        model_root=args.model_root,
        batch_size=args.batch_size,
        eval_target=args.eval_target,
        decision_threshold=args.decision_threshold,
        output_csv=args.output_csv,
        output_excel=args.output_excel,
    )


if __name__ == "__main__":
    main()
