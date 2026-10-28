"""
Filename: utils.py

Author: Hakima Laribi

Description: General-purpose helpers: type auto-casting, statistical comparison of model
performances (NRI, Wilcoxon significance), graph adjacency normalization, and seeding.
"""
import ast
import json
import os
import random
from re import search
from typing import List

import dgl
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch


def auto_cast(value):
    """
    Casts a string value to the most specific Python type it represents (bool, int, float,
    or a literal via ast.literal_eval), falling back to the original string if none apply.
    Args:
        value: the value to cast
    Returns:
        value cast to bool, int, float, a literal (list/tuple/dict/...), or the original value
    """
    # Already not a string → return as is
    if not isinstance(value, str):
        return value

    # Try boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try list / tuple / etc.
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # Fallback: keep as string
    return value


def compute_NRI(
    model_1: str,
    model_2: str,
    n_splits: int,
    saving_dir: str,
):
    """
    Computes the Net Reclassification Improvement (NRI) between two models' predictions for
    each split, and writes the per-split and mean/std values to <saving_dir>/p_values.json.
    Args:
        model_1: path to the directory where the experiment of model 1 is recorded
        model_2: path to the directory where the experiment of model 2 is recorded
        n_splits: number of splits used in both experiments, must be identical
        saving_dir: path to the directory where to save the computed NRI values
    """
    # Create saving directory
    os.makedirs(saving_dir, exist_ok=True)
    nri = {}

    # Iterate over all the splits
    for i in range(n_splits):
        # Read the data recorded at each split for both models
        with open(os.path.join(model_1, 'fold_' + str(i), 'records.json'), "r") as file:
            model_1_data = json.load(file)

        with open(os.path.join(model_2, 'fold_' + str(i), 'records.json'), "r") as file:
            model_2_data = json.load(file)

        for section in model_1_data.keys():
            # Compare AUCs of testing sets only
            if search("(test).*(predictions)$", section):
                if section not in nri.keys():
                    nri[section] = {}

                # Get prediction and target of each
                model_1_predictions, model_2_predictions = model_1_data[section], model_2_data[section]

                # Reorganize the dictionary of dictionaries as dictionary of tables
                model_1_values = {patient_id: list(pred_target.values())
                                  for patient_id, pred_target in model_1_predictions.items()}

                model_2_values = {patient_id: list(pred_target.values())
                                  for patient_id, pred_target in model_2_predictions.items()}

                # Transform to dataframe
                df_1 = pd.DataFrame.from_dict(model_1_values,
                                              orient='index',
                                              columns=['target', 'probability', 'prediction', ])
                df_2 = pd.DataFrame.from_dict(model_2_values,
                                              orient='index',
                                              columns=['target', 'old_probability', 'old_prediction', ])

                # Verify that indexes are the same
                if not df_1.index.equals(df_2.index):
                    print('WARNING: indexes not matching')
                    print(model_2)

                # Verify that targets are the same
                if not df_1['target'].equals(df_2['target']):
                    print('WARNING: targets not matching')
                    print(model_2)
                else:
                    df_1.drop(columns=['target'], inplace=True)

                # Merge both datasets
                df = pd.concat([df_1, df_2], axis=1)

                # Filter cases and controls
                cases = df[df['target'] == 1]
                controls = df[df['target'] == 0]

                # compute NRI
                cases_up = ((cases['old_prediction'] == 0) & (cases['prediction'] == 1)).sum()
                cases_down = ((cases['old_prediction'] == 1) & (cases['prediction'] == 0)).sum()
                total_cases = cases.shape[0]

                controls_up = ((controls['old_prediction'] == 0) & (controls['prediction'] == 1)).sum()
                controls_down = ((controls['old_prediction'] == 1) & (controls['prediction'] == 0)).sum()
                total_controls = controls.shape[0]

                nri_cases = (cases_up / total_cases) - (cases_down / total_cases) if total_cases > 0 else 0
                nri_controls = (controls_down / total_controls) - (
                        controls_up / total_controls) if total_controls > 0 else 0

                # Save AUCs
                nri[section][str(i)] = nri_cases + nri_controls

    for section, nri_section in nri.items():
        nri[section]['mean'] = np.mean(list(nri_section.values()))
        nri[section]['std'] = np.std(list(nri_section.values()))

    with open(os.path.join(saving_dir, 'p_values.json'), "w") as file:
        json.dump(nri, file, indent=True)


def compute_significant_difference(
    model_1: str,
    model_2: str,
    n_splits: int,
    saving_dir: str,
    metrics: List[str],
):
    """
    Computes the Wilcoxon signed-rank p-value of the difference between two models' metrics
    across splits, and writes the per-split values and p-values to <saving_dir>/p_values.json.
    Args:
        model_1: path to the directory where the experiment of model 1 is recorded
        model_2: path to the directory where the experiment of model 2 is recorded
        n_splits: number of splits used in both experiments, must be identical
        saving_dir: path to the directory where to save the computed p-values
        metrics: names of the metrics to compare (e.g. 'AUROC', 'AUPRC')
    """

    p_values = {}

    # Iterate over all the splits
    for i in range(n_splits):
        # Read the data recorded at each split for both models
        with open(os.path.join(model_1, 'fold_' + str(i), 'records.json'), "r") as file:
            model_1_data = json.load(file)

        with open(os.path.join(model_2, 'fold_' + str(i), 'records.json'), "r") as file:
            model_2_data = json.load(file)

        model_1_metrics, model_2_metrics = [], []
        for section in model_1_data.keys():
            # Compare AUCs of testing sets only
            if search("(test).*(metrics)$", section):
                if section not in p_values.keys():
                    p_values[section] = {metric: {} for metric in metrics}

                # save AUCs
                for metric in metrics:
                    p_values[section][metric][str(i)] = [model_1_data[section][metric], model_2_data[section][metric]]

    for section, p_val_section in p_values.items():
        for metric in metrics:
            m_1 = np.array([float(p_val_section[metric][str(m)][0]) for m in range(n_splits)])
            m_2 = np.array([float(p_val_section[metric][str(m)][1]) for m in range(n_splits)])
            p_values[section][metric]['wilcoxon'] = wilcoxon(m_1, m_2, alternative='greater')[1]
            p_values[section][metric]['wilcoxon_less'] = wilcoxon(m_1, m_2, alternative='less')[1]

    # Create saving directory
    os.makedirs(saving_dir, exist_ok=True)

    with open(os.path.join(saving_dir, 'p_values.json'), "w") as file:
        json.dump(p_values, file, indent=True)


def normalize_adj(adj):
    """
    Symmetrically normalizes an adjacency matrix: D^-0.5 * adj * D^-0.5.
    Args:
        adj: adjacency matrix to normalize
    Returns:
        the symmetrically normalized adjacency matrix
    """
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj


def set_seed(seed: int):
    """
    Set the seed for reproducibility
    Args:
        seed: the seed to set
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.use_libxsmm(False)
