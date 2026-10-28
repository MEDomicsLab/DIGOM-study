"""
Filename: sampling.py

Author: Hakima Laribi

Description: Patient-level sampling modules: picking one unique occurrence per patient (to
avoid leakage across folds), and building (nested) stratified k-fold train/valid/test splits.
"""
import json
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def extract_unique_occurrence(df: pd.DataFrame,
                              unique_by: Literal['hadm_id', 'stay_id'],
                              path_to_ids: str,
                              group_by: Literal['subject_id'] = 'subject_id') -> list:
    """
    Samples one occurrence (admission or stay) per group (patient) and saves the resulting ids.
    Args:
        df: dataframe containing one row per occurrence, with at least `unique_by`/`group_by` columns
        unique_by: column identifying each occurrence (e.g. 'hadm_id' or 'stay_id')
        path_to_ids: path to save the sampled ids to, as JSON
        group_by: column identifying each patient, one occurrence is sampled per group
    Returns:
        the sampled unique occurrence ids
    """
    sampled_df = df.groupby(group_by).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
    unique_ids = sampled_df[unique_by].values.tolist()

    with open(path_to_ids, "w") as file:
        json.dump(unique_ids, file, indent=True)

    return unique_ids


def load_unique_occurrence(path_to_ids: str,
                           df=None,
                           unique_by=None,
                           group_by=None):
    """
    Loads previously sampled unique occurrence ids, or samples and saves them if not found.
    Args:
        path_to_ids: path to load the sampled ids from (or save them to, if missing)
        df: dataframe to sample from if `path_to_ids` doesn't exist yet
        unique_by: column identifying each occurrence, see extract_unique_occurrence
        group_by: column identifying each patient, see extract_unique_occurrence
    Returns:
        the unique occurrence ids
    """
    try:
        with open(path_to_ids, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return extract_unique_occurrence(df, unique_by, path_to_ids, group_by)


def sample_k_folds(ids, targets, k, valid_size, keep_targets=False):
    """
    Performs k-fold cross-validation to designate train/test splits, and validation if specified
    Args:
        ids: sample ids to split
        targets: labels used to stratify the splits, aligned with `ids`
        k: number of folds
        valid_size: fraction of the training fold held out for validation; no validation split if 0
        keep_targets: if True, also return each fold's training targets (needed to nest folds)
    Returns:
        masks: {fold: {"train": ids, "valid": ids, "test": ids}}
        train_targets: per-fold training targets, only if `keep_targets` is True
    """
    masks = {}
    train_targets = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    x, y = np.array(ids), np.array(targets)

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        x_valid = []
        if valid_size > 0:
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=42)
            x_valid = x_valid.tolist()

        masks[fold] = {"train": x_train.tolist(), "valid": x_valid, "test": x_test.tolist()}
        if keep_targets:
            train_targets.append(y_train)

    if keep_targets:
        return masks, train_targets

    return masks


def sample_nested_k_folds(ids, targets, k, valid_size, inner_k, inner_valid_size):
    """
    Performs nested k-fold cross-validation to designate train/test splits, and validation if specified
    Args:
        ids: sample ids to split
        targets: labels used to stratify the splits, aligned with `ids`
        k: number of outer folds
        valid_size: fraction of each outer training fold held out for validation; none if 0
        inner_k: number of inner folds to nest within each outer training fold; skipped if 0
        inner_valid_size: fraction of each inner training fold held out for validation
    Returns:
        the outer folds' masks, each augmented with an "inner" key holding its nested folds
        (see sample_k_folds) when `inner_k` > 0
    """
    masks, y_targets = sample_k_folds(ids, targets, k, valid_size, keep_targets=True)

    # Inner loop for nested cross-validation
    if inner_k > 0:
        for fold in range(k):
            masks[fold]["inner"] = sample_k_folds(masks[fold]['train'], y_targets[fold], inner_k, inner_valid_size)

    return masks

