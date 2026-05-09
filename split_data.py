#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split

import main as M


def parse_args():
    parser = argparse.ArgumentParser(description="Create and save multiple train/test splits.")
    parser.add_argument("--num-splits", type=int, default=25, help="Number of split patterns")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-path",
        type=str,
        default="result/split_indices.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = M.load_dataset(
        M.json_path,
        M.hold_to_idx,
        M.grade_to_label,
        M.hold_difficulty,
        M.type_to_idx,
        M.hold_to_coord,
    )
    targets = [M.grade_to_label[item["grade"]] for item in dataset.raw]
    all_indices = list(range(len(dataset)))

    split_records = []
    for i in range(args.num_splits):
        seed = args.base_seed + i
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=args.test_size,
            stratify=targets,
            random_state=seed,
        )
        split_records.append(
            {
                "split_id": i + 1,
                "seed": seed,
                "train_idx": sorted(int(x) for x in train_idx),
                "test_idx": sorted(int(x) for x in test_idx),
            }
        )

    payload = {
        "dataset_path": M.json_path,
        "num_samples": len(dataset),
        "num_splits": args.num_splits,
        "test_size": args.test_size,
        "base_seed": args.base_seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": split_records,
    }

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.num_splits} splits to {args.output_path}")


if __name__ == "__main__":
    main()
