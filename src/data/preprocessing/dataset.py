"""
Filename: dataset.py

Author: Hakima Laribi

Description: Multi-modal dataset classes.
"""
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import Dataset

from src.data.preprocessing import constants as c


class MMDataset:
    """
    Multi-modal dataset used for MIMIC cohorts
    """

    def __init__(self,
                 df_modalities: Dict[str, pd.DataFrame],
                 target_column: pd.DataFrame,
                 temporal_analysis: bool,
                 temporal_modalities: List[bool],
                 to_tensor: bool = True,
                 id_column: str = 'hadm_id',
                 impute_cont_cols: Optional[Literal['mean', 'zero', 'minus_one']] = None,
                 modality_miss=None,
                 ):
        """
        Args:
            df_modalities: per-modality dataframes, all sharing `id_column` and in the same order
            target_column: dataframe with `id_column` and the label as its last column
            temporal_analysis: True if samples are organized as temporal sequences
            temporal_modalities: per-modality flag, True if that modality has temporal sequences
            to_tensor: if True, exposes torch tensors; else numpy arrays
            id_column: name of the identifier column shared across `df_modalities`
            impute_cont_cols: strategy to impute missing continuous values ('mean', 'zero',
                'minus_one'), or None to leave them as NaN
            modality_miss: modality (or list of modalities) to simulate as missing/corrupted
        """
        self._df_modalities = df_modalities
        self._target_column = target_column
        self._temporal_analysis = temporal_analysis
        self._temporal_modalities = temporal_modalities
        self._to_tensor = to_tensor
        self._id_column = id_column
        self._impute_cont_cols = impute_cont_cols
        self.modality_miss = modality_miss if (modality_miss is None) | (isinstance(modality_miss, list)) else [modality_miss]

        # Verify that we have all ids in all modality dfs and target df, and they appear in the same order
        if not self.validate_ids():
            raise ValueError('identifier columns do not match across different modality datasets')

        # Get modality variables and find their different types
        self._modality_vars = self.get_vars_per_modality()
        self._bin_cols = self.get_binary_cols()
        self._cont_cols = self.get_continuous_cols()
        self._cat_cols = self.get_categorical_cols()

        # Impute demographics and one hot encode categorical columns
        self.impute_demographics(self._df_modalities['demo']) if 'demo' in self._df_modalities.keys() else None
        self.encode_categorical()

        # Reorder columns in each modality dataset to have continuous, then binary then categorical columns
        self.reorder_columns()

        # Get indexes from the dataset
        self._ids = self._target_column[self._id_column].unique().tolist()
        self._cont_idx = self.get_continuous_idx()
        self._ids_to_idx = self.from_ids_to_indexes()  # needed if temporal, all indexes of each id in each modality df

        # Set masks and arrays/tensors that will be used for prediction
        self._means, self._stds = None, None
        self._train_ids, self._valid_ids, self._test_ids = None, None, None
        self._train_indexes, self._valid_indexes, self._test_indexes = None, None, None
        self._x = {modality: None for modality in self._df_modalities.keys()}
        self._y = self._target_column.drop_duplicates().iloc[:, -1].to_numpy().astype(int)
        if self._to_tensor:
            self._y = torch.from_numpy(self._y)

        self._df_modalities_tmp = deepcopy(self._df_modalities)

    def __getitem__(self, indexes):
        """
        Args:
            indexes: sample indexes to select
        Returns:
            x, y: per-modality features for `indexes` (each temporal modality kept as a list of
            variable-length sequences, each static modality stacked into one tensor/array), and
            the corresponding labels
        """
        if self._temporal_analysis:
            x = []
            for mod, x_modality in enumerate(self._x.values()):
                modality = [x_modality[i] for i in indexes]
                if not self._temporal_modalities[mod]:
                    modality = torch.stack(modality, dim=0).squeeze(1)
                x.append(modality)
        else:
            x = [x_modality[indexes] for x_modality in self._x.values()]
        y = self._y[indexes]
        return x, y

    @property
    def x(self):
        """Returns: per-modality feature tensors/arrays for the current masks."""
        return self._x

    @property
    def y(self):
        """Returns: the labels of all samples."""
        return self._y

    @property
    def train_ids(self):
        """Returns: the training sample ids."""
        return self._train_ids

    @property
    def valid_ids(self):
        """Returns: the validation sample ids."""
        return self._valid_ids

    @property
    def test_ids(self):
        """Returns: the test sample ids."""
        return self._test_ids

    @property
    def train_indexes(self):
        """Returns: the training sample indexes."""
        return self._train_indexes

    @property
    def valid_indexes(self):
        """Returns: the validation sample indexes."""
        return self._valid_indexes

    @property
    def test_indexes(self):
        """Returns: the test sample indexes."""
        return self._test_indexes

    @property
    def task(self):
        """Returns: the name of the target (label) column."""
        return self._target_column.columns[-1]

    @property
    def id_column(self):
        """Returns: the name of the sample identifier column."""
        return self._id_column

    @property
    def temporal_analysis(self):
        """Returns: True if samples are organized as temporal sequences."""
        return self._temporal_analysis

    @property
    def temporal_modalities(self):
        """Returns: the per-modality flag, True if that modality has temporal sequences."""
        return self._temporal_modalities

    @property
    def modalities(self):
        """Returns: the names of all modalities."""
        return list(self._df_modalities.keys())

    def reorder_columns(self):
        """Reorders each modality's columns as continuous, then binary, then categorical."""
        for modality, df_modality in self._df_modalities.items():
            mod_cols = df_modality.columns.tolist()
            mod_cont_cols = [col for col in self._cont_cols if col in mod_cols]
            mod_cat_cols = [col for col in self._cat_cols if col in mod_cols]
            mod_bin_cols = [col for col in self._bin_cols if col in mod_cols]
            mod_pred_cols = mod_cont_cols + mod_bin_cols + mod_cat_cols
            reordered_cols = [col for col in mod_cols if col not in mod_pred_cols] + mod_pred_cols
            self._df_modalities[modality] = df_modality[reordered_cols]
            self._modality_vars[modality] = mod_pred_cols

    def get_binary_cols(self):
        """Returns: all binary columns (0/1-valued, excluding identifier/target columns), across modalities."""
        non_pred_cols = [c.SUBJECT, c.ADM, c.STAY, self._target_column.columns[-1], 'visit_id', 'patient_id']
        binary_cols = []
        for modality, df_modality in self._df_modalities.items():
            if modality not in ['cxr', 'rad', 'disch']:  # all are continuous embeddings
                binaries = [col for col in df_modality.columns
                            if (set(df_modality[col].unique()) <= {0, 1}) & (col not in non_pred_cols)]
                df_modality[binaries] = df_modality[binaries].astype(int)
                binary_cols.extend(binaries)
        return binary_cols

    def get_continuous_cols(self):
        """Returns: all numeric, non-binary columns (excluding identifier/target columns), across modalities."""
        non_pred_cols = [c.SUBJECT, c.ADM, c.STAY, self._target_column.columns[-1], 'visit_id', 'patient_id']
        cont_cols = []
        for modality, df_modality in self._df_modalities.items():
            cont = [col for col in df_modality.columns
                    if (pd.api.types.is_numeric_dtype(df_modality[col])) & (col not in non_pred_cols + self._bin_cols)]
            cont_cols.extend(cont)
        return cont_cols

    def get_categorical_cols(self):
        """Returns: all remaining (non-continuous, non-binary) columns, across modalities."""
        non_pred_cols = [c.SUBJECT, c.ADM, c.STAY, self._target_column.columns[-1], 'visit_id', 'patient_id']
        cat_cols = []
        for modality, df_modality in self._df_modalities.items():
            cont = [col for col in df_modality.columns
                    if (col not in self._cont_cols + self._bin_cols) & (col not in non_pred_cols)]
            cat_cols.extend(cont)
        return cat_cols

    def get_vars_per_modality(self):
        """Returns: per-modality list of predictor columns (excluding identifier/target columns)."""
        non_pred_cols = [c.SUBJECT, c.ADM, c.STAY, self._target_column.columns[-1], 'visit_id', 'patient_id']
        return {modality: [col for col in df.columns if col not in non_pred_cols]
                for modality, df in self._df_modalities.items()}

    def get_continuous_idx(self):
        """Returns: per-modality positional indexes (within its predictor columns) of continuous columns."""
        non_pred_cols = [c.SUBJECT, c.ADM, c.STAY, self._target_column.columns[-1], 'visit_id', 'patient_id']
        continuous_idx = {}
        for modality, df in self._df_modalities.items():
            variables = [col for col in df.columns if col not in non_pred_cols]
            cont_vars = [var for var in variables if var in self._cont_cols]
            idx_cont_vars = [variables.index(var) for var in cont_vars]
            continuous_idx[modality] = idx_cont_vars
        return continuous_idx

    def validate_ids(self):
        """
        Sorts every modality dataframe (and the target column) by `id_column`.
        Returns:
            True if all modalities share the exact same set of ids, in the same order
        """
        for modality, df_modality in self._df_modalities.items():
            self._df_modalities[modality] = df_modality.sort_values(by=self._id_column).reset_index(drop=True)
        self._target_column = self._target_column.sort_values(by=self._id_column).reset_index(drop=True)
        unique_ids = self._target_column[self._id_column].unique()
        match = all(df_modality[self._id_column].unique().tolist() == unique_ids.tolist()
                    for _, df_modality in self._df_modalities.items())
        return match

    def from_ids_to_indexes(self):
        """Returns: per-modality {id: [row indexes]}, used to locate a temporal sequence's rows."""
        return {modality: df[self._id_column].groupby(df[self._id_column]).apply(lambda x: list(x.index)).to_dict()
                for modality, df in self._df_modalities.items()}

    def impute_demographics(self, df: pd.DataFrame):
        """Fills missing demographic values with 'unknown'."""
        demo_vars = self._modality_vars['demo']
        df[demo_vars] = df[demo_vars].fillna('unknown')

    def encode_categorical(self):
        """For each modality, encodes gender as binary and one-hot encodes other categorical variables."""
        for modality, df_modality in self._df_modalities.items():
            # Start with sex
            if 'gender' in df_modality.columns:
                df_modality['gender'] = np.where(df_modality['gender'] == 'M', 1, 0).astype(int)
            # One hot encode categorical columns
            mod_cat_cols = [col for col in self._modality_vars[modality] if col in self._cat_cols]
            if len(mod_cat_cols) > 0:
                df = pd.get_dummies(df_modality[mod_cat_cols]).astype(int)
                # Concatenate one hot encoded columns with other columns of the data modalities
                other_cols = [col for col in df_modality.columns if col not in mod_cat_cols]
                # Update the modality dataset with the one hot encoded columns
                self._df_modalities[modality] = pd.concat([df_modality[other_cols], df], axis=1)
                # Update the variable names (after one hot encoding) and categorical variables
                self._modality_vars[modality] = df.columns.tolist() + [col for col in self._modality_vars[modality]
                                                                       if col not in mod_cat_cols]
                self._cat_cols = df.columns.tolist() + [col for col in self._cat_cols if col not in mod_cat_cols]

    def temporalize_dataset(self, flat_x, modality):
        """
        Regroups a modality's flat rows into one variable-length sequence per unique id.
        Args:
            flat_x: flat (non-temporal) feature tensor/array for `modality`
            modality: modality to temporalize
        Returns:
            a list of per-id sequences, ordered by first appearance in the dataframe
        """
        df_modality = self._df_modalities[modality]
        x = []
        # respect the order of appearance in df, we expect the first _id to be in index 0
        for _id in df_modality[self._id_column].unique():
            idx = self._ids_to_idx[modality][_id]
            x.append(flat_x[idx])
        return x

    def get_indexes_from_ids(self):
        """Sets `_train_indexes`/`_valid_indexes`/`_test_indexes` from the corresponding ids."""
        modality = list(self._df_modalities.keys())[0]  # any modality works, ids are shared
        df_modality = self._df_modalities[modality]
        unique_ids = df_modality[self._id_column].unique()  # ids in the same order of appearance in df
        for mask in ['train', 'valid', 'test']:
            ids = getattr(self, f'_{mask}_ids')
            indexes = [int(np.where(unique_ids == value)[0][0]) for value in ids]
            setattr(self, f'_{mask}_indexes', indexes)

    def get_statistical_values(self, flat_x, modality):
        """
        Computes the mean and std of the continuous variables of `modality`, from training data
        only. Drops any continuous column that is entirely NaN in the training set.
        Args:
            flat_x: flat feature tensor/array for `modality`
            modality: modality to compute statistics for
        Returns:
            mean, std, flat_x: per-continuous-column mean and std (None if there are none), and
            `flat_x` with any all-NaN training columns dropped
        """
        cont_idx = self._cont_idx[modality]
        if len(cont_idx) > 0:
            if self._temporal_analysis:
                org_train_idx = []
                # Get the training indexes, with all the occurrences in the temporal dataset
                for _id in self._train_ids:
                    org_train_idx.extend(self._ids_to_idx[modality][_id])
            else:
                org_train_idx = self._train_indexes

            if torch.isnan(flat_x[org_train_idx][:, cont_idx]).all(axis=0).any():
                # Get columns with iny Nan values
                nan_cont_idx = torch.where(torch.isnan(flat_x[org_train_idx][:, cont_idx]).all(axis=0))[
                    0].numpy().tolist()
                nan_id_cols = [cont_idx[i] for i in nan_cont_idx]
                nan_cols = [col for i, col in enumerate(self._modality_vars[modality]) if i in nan_id_cols]

                # Drop these columns from the dataset of the modality
                self._cont_idx[modality] = [i for i in self._cont_idx[modality] if i not in nan_id_cols]
                cont_idx = self._cont_idx[modality]
                self._modality_vars[modality] = [col for col in self._modality_vars[modality] if col not in nan_cols]
                self._cont_cols = [col for col in self._cont_cols if col not in nan_cols]

                # Drop these columns from the flat_x
                flat_x = flat_x[:, [i for i in range(flat_x.shape[1]) if i not in nan_id_cols]]

            mean = np.nanmean(flat_x[org_train_idx][:, cont_idx].numpy(), axis=0, keepdims=True)
            std = np.nanstd(flat_x[org_train_idx][:, cont_idx].numpy(), axis=0, keepdims=True)

            # Use only training data to compute mean and std
            if self._to_tensor:
                mean = torch.from_numpy(mean)
                std = torch.from_numpy(std)
            return mean, std, flat_x
        return None, None, flat_x

    def impute_continuous_columns(self, flat_x, modality):
        """
        Imputes the continuous columns of `modality` per `self._impute_cont_cols`, and the
        categorical columns with their training-set mode.
        Args:
            flat_x: flat feature tensor/array for `modality`
            modality: modality to impute
        Returns:
            flat_x with missing values imputed
        """
        cont_idx = self._cont_idx[modality]

        if len(cont_idx) > 0:
            if self._impute_cont_cols == 'mean':
                mean = self._means[modality]
                if self._to_tensor:
                    flat_x[:, cont_idx] = torch.where(torch.isnan(flat_x[:, cont_idx]), mean, flat_x[:, cont_idx])
                else:
                    flat_x[:, cont_idx] = np.where(np.isnan(flat_x[:, cont_idx]), mean, flat_x[:, cont_idx])
            elif self._impute_cont_cols == 'zero':
                if self._to_tensor:
                    flat_x[:, cont_idx] = torch.nan_to_num(flat_x[:, cont_idx], nan=0.0)
                else:
                    flat_x[:, cont_idx] = np.nan_to_num(flat_x[:, cont_idx], nan=0.0)
            elif self._impute_cont_cols == 'minus_one':
                if self._to_tensor:
                    flat_x[:, cont_idx] = torch.nan_to_num(flat_x[:, cont_idx], nan=-1.0)
                else:
                    flat_x[:, cont_idx] = np.nan_to_num(flat_x[:, cont_idx], nan=-1.0)

        # columns are ordered as : continuous, binary, categorical
        cat_start = max(cont_idx) + 1 if len(cont_idx) > 0 else 0
        cat_cols = [i for i in range(cat_start, flat_x.shape[1])]
        if len(cat_cols) > 0:
            # For categorical columns, impute with the most frequent value in the training set
            x_train_col = flat_x[self._train_indexes][:, cat_cols]
            modes = stats.mode(x_train_col, axis=0, nan_policy='omit', keepdims=True)[0]

            if not self._to_tensor:
                flat_x[:, cat_cols] = np.where(np.isnan(flat_x[:, cat_cols]), modes, flat_x[:, cat_cols])
            else:
                flat_x[:, cat_cols] = torch.where(
                    torch.isnan(flat_x[:, cat_cols]),
                    torch.tensor(modes, dtype=flat_x.dtype, device=flat_x.device),
                    flat_x[:, cat_cols]
                )

        return flat_x

    def normalize(self, flat_x, modality):
        """
        Standardizes the continuous columns of `modality` using the training mean/std, then
        drops any column that is constant across the whole dataset (std 0, non-finite after scaling).
        Args:
            flat_x: flat feature tensor/array for `modality`
            modality: modality to normalize
        Returns:
            flat_x, normalized and with non-finite columns dropped
        """
        cont_idx = self._cont_idx[modality]

        if len(cont_idx) > 0:
            mean = self._means[modality]
            std = self._stds[modality]

            flat_x[:, cont_idx] = (flat_x[:, cont_idx] - mean) / std

            if self._to_tensor:
                flat_x = flat_x[:, torch.isfinite(flat_x).all(dim=0)]
            else:
                flat_x = flat_x[:, np.isfinite(flat_x).all(axis=0)]

        return flat_x

    def set_masks(self, train_ids, valid_ids, test_ids):
        """
        Sets the train/valid/test ids and indexes, then rebuilds each modality's tensor/array
        (re-detecting imputation statistics and reapplying imputation/normalization/temporalizing).
        We talk about ids, not indexes, here.
        Args:
            train_ids: ids of the training samples
            valid_ids: ids of the validation samples
            test_ids: ids of the test samples
        """
        self._train_ids, self._valid_ids, self._test_ids = train_ids, valid_ids, test_ids
        # indexes hold the position of appearance of ids in the dataset. eg: [p_a, p_b, p_c] (ids) -> [0, 1, 2] (idx)
        self._train_indexes, self._valid_indexes, self._test_indexes = [], [], []

        self._x = {modality: None for modality in self._df_modalities.keys()}

        self.get_indexes_from_ids()

        self._df_modalities = deepcopy(self._df_modalities_tmp)

        # Create missing values
        if self.modality_miss is not None:
            for moddality in self._df_modalities:
                if moddality in self.modality_miss:
                    indices = self.test_indexes if not self._temporal_analysis else [idx
                                                                                     for _id in self.test_ids
                                                                                     for idx in
                                                                                     self._ids_to_idx[moddality][_id]]
                    columns = self._df_modalities[moddality].columns.difference(
                        ['stay_id', 'hadm_id', 'subject_id', 'visit_id', 'patient_id', self.task])
                    self._df_modalities[moddality].loc[indices, columns] = np.nan

        # Set means and stds
        self._means = {modality: None for modality in self._df_modalities.keys()}
        self._stds = {modality: None for modality in self._df_modalities.keys()}

        # Transform dataframes to usable arrays or tensors
        for modality, df_modality in self._df_modalities.items():
            flat_x = df_modality[self._modality_vars[modality]].to_numpy().astype(np.float32)
            if self._to_tensor:
                flat_x = torch.from_numpy(flat_x)

            self._means[modality], self._stds[modality], flat_x = self.get_statistical_values(flat_x, modality)
            flat_x = self.impute_continuous_columns(flat_x, modality)
            flat_x = self.normalize(flat_x, modality)

            self._x[modality] = flat_x if not self._temporal_analysis else self.temporalize_dataset(flat_x, modality)


class MMTuberculosisDataset:
    """
    Array-based (TADPOLE) multi-modal dataset: non-temporal, with a precomputed imputation mask

    """

    def __init__(self,
                 df_modalities: Dict[str, np.array],
                 target_column: np.array,
                 mask_imputation: Dict[str, np.array],
                 to_tensor: bool = True,
                 modality_miss=None
                 ):
        """
        Args:
            df_modalities: per-modality feature arrays, all sharing row order
            target_column: array of labels, aligned with `df_modalities`' rows
            mask_imputation: per-modality boolean array, True where a value is observed (not imputed)
            to_tensor: if True, exposes torch tensors; else numpy arrays
            modality_miss: modality (or list of modalities) to simulate as missing/corrupted
        """
        self._df_modalities = df_modalities
        self._target_column = target_column
        self._temporal_analysis = False
        self._temporal_modalities = [False] * len(df_modalities)
        self._mask_imputation = mask_imputation
        self._to_tensor = to_tensor
        self._id_column = 'index'
        self.modality_miss = modality_miss if (modality_miss is None) | (isinstance(modality_miss, list)) else [modality_miss]

        # use row positions as ids, since this dataset has no identifier column
        self._ids = np.arange(self._target_column.shape[0])

        # Get indexes of continuous variables for each modality, needed for normalization
        self._cont_idx = {
            mod_name: np.arange(mod_vals.shape[1]).tolist() if mod_name in ['cont_feat', 'ct_feat', 'snp_feat']
            else [] for mod_name, mod_vals in self._df_modalities.items()
        }

        self._means, self._stds = None, None
        self._train_ids, self._valid_ids, self._test_ids = None, None, None
        self._train_indexes, self._valid_indexes, self._test_indexes = None, None, None
        self._x = {modality: None for modality in self._df_modalities.keys()}
        self._y = self._target_column.astype(int)
        if self._to_tensor:
            self._y = torch.from_numpy(self._y)

        self._mask_imputation_tmp = deepcopy(self._mask_imputation)
        self._df_modalities_tmp = deepcopy(self._df_modalities)

    def __getitem__(self, indexes):
        """
        Args:
            indexes: sample indexes to select
        Returns:
            x, y: per-modality features for `indexes`, and the corresponding labels
        """
        x = [x_modality[indexes] for x_modality in self._x.values()]
        y = self._y[indexes]
        return x, y

    @property
    def x(self):
        """Returns: per-modality feature tensors/arrays for the current masks."""
        return self._x

    @property
    def y(self):
        """Returns: the labels of all samples."""
        return self._y

    @property
    def train_ids(self):
        """Returns: the training sample ids."""
        return self._train_ids

    @property
    def valid_ids(self):
        """Returns: the validation sample ids."""
        return self._valid_ids

    @property
    def test_ids(self):
        """Returns: the test sample ids."""
        return self._test_ids

    @property
    def train_indexes(self):
        """Returns: the training sample indexes."""
        return self._train_indexes

    @property
    def valid_indexes(self):
        """Returns: the validation sample indexes."""
        return self._valid_indexes

    @property
    def test_indexes(self):
        """Returns: the test sample indexes."""
        return self._test_indexes

    @property
    def task(self):
        """Returns: the name of the prediction task."""
        return 'tuberculosis'

    @property
    def id_column(self):
        """Returns: the name of the sample identifier column."""
        return self._id_column

    @property
    def temporal_analysis(self):
        """Returns: True if samples are organized as temporal sequences (always False here)."""
        return self._temporal_analysis

    @property
    def temporal_modalities(self):
        """Returns: the per-modality flag, True if that modality has temporal sequences."""
        return self._temporal_modalities

    @property
    def modalities(self):
        """Returns: the names of all modalities."""
        return list(self._df_modalities.keys())

    def get_statistical_values(self, x, modality):
        """
        Computes the mean and std of the continuous variables of `modality`, from non-imputed
        training data only.
        Args:
            x: feature array for `modality`
            modality: modality to compute statistics for
        Returns:
            mean, std (None if there are no continuous columns)
        """
        cont_idx = self._cont_idx[modality]
        if len(cont_idx) > 0:
            # ignore originally-imputed values when computing statistics
            x_train = np.where(self._mask_imputation[modality][self._train_indexes], x[self._train_indexes], np.nan)

            mean = np.nanmean(x_train[:, cont_idx], axis=0, keepdims=True)
            std = np.nanstd(x_train[:, cont_idx], axis=0, keepdims=True)

            if self._to_tensor:
                mean = torch.from_numpy(mean)
                std = torch.from_numpy(std)

            return mean, std
        return None, None

    def normalize(self, x_modality, modality):
        """
        Standardizes the continuous columns of `modality` using the training mean/std, then
        drops any column that is constant across the whole dataset (std 0, non-finite after scaling).
        Args:
            x_modality: feature tensor/array for `modality`
            modality: modality to normalize
        Returns:
            x_modality, normalized and with non-finite columns dropped
        """
        cont_idx = self._cont_idx[modality]
        if len(cont_idx) > 0:
            mean = self._means[modality]
            std = self._stds[modality]

            x_modality[:, cont_idx] = (x_modality[:, cont_idx] - mean) / std

            if self._to_tensor:
                x_modality = x_modality[:, torch.isfinite(x_modality).all(dim=0)]
            else:
                x_modality = x_modality[:, np.isfinite(x_modality).all(axis=0)]

        return x_modality

    def impute_continuous_columns(self, x, modality):
        """
        Imputes the continuous columns of `modality` with their training mean, and the
        categorical columns with their training-set mode.
        Args:
            x: feature array for `modality`
            modality: modality to impute
        Returns:
            x with missing (per `self._mask_imputation`) values imputed
        """
        cont_idx = self._cont_idx[modality]
        if len(cont_idx) > 0:
            x = np.where(self._mask_imputation[modality], x, np.nan)
            mean = self._means[modality]
            x[:, cont_idx] = np.where(np.isnan(x[:, cont_idx]), mean, x[:, cont_idx])

        cat_cols = [i for i in range(x.shape[1]) if i not in cont_idx]
        if len(cat_cols) > 0:
            # For categorical columns, impute with the most frequent value in the training set
            x_train_col = np.where(self._mask_imputation[modality][self._train_indexes],
                                   x[self._train_indexes][:, cat_cols],
                                   np.nan)
            modes = stats.mode(x_train_col, axis=0, nan_policy='omit', keepdims=True)[0]

            x[:, cat_cols] = np.where(np.isnan(x[:, cat_cols]), modes, x[:, cat_cols])

        return x

    def set_masks(self, train_ids, valid_ids, test_ids):
        """
        Sets the train/valid/test ids and indexes (identical here, since ids are row positions),
        then rebuilds each modality's array (re-detecting imputation statistics and reapplying
        imputation/normalization).
        Args:
            train_ids: ids (row positions) of the training samples
            valid_ids: ids (row positions) of the validation samples
            test_ids: ids (row positions) of the test samples
        """
        self._train_ids, self._valid_ids, self._test_ids = train_ids, valid_ids, test_ids
        # indexes hold the position of appearance of ids in the dataset. eg: [p_a, p_b, p_c] (ids) -> [0, 1, 2] (idx)
        self._train_indexes, self._valid_indexes, self._test_indexes = train_ids, valid_ids, test_ids

        self._x = {modality: None for modality in self._df_modalities.keys()}

        self._mask_imputation = deepcopy(self._mask_imputation_tmp)
        self._df_modalities = deepcopy(self._df_modalities_tmp)

        # Create missing values
        if self.modality_miss is not None:
            for moddality in self._df_modalities:
                if moddality in self.modality_miss:
                    indices = self.test_indexes
                    self._mask_imputation[moddality][indices, :] = False
                    self._df_modalities[moddality][indices, :] = np.nan

        self._means = {modality: None for modality in self._df_modalities.keys()}
        self._stds = {modality: None for modality in self._df_modalities.keys()}

        # Transform dataframes to usable arrays or tensors
        for modality, x_modality in self._df_modalities.items():
            self._means[modality], self._stds[modality] = self.get_statistical_values(x_modality, modality)
            x_modality = self.impute_continuous_columns(x_modality, modality)

            if self._to_tensor:
                x_modality = torch.from_numpy(x_modality.copy())

            self._x[modality] = self.normalize(x_modality, modality)


class TorchDataset(Dataset):
    """Adapts MMDataset/MMTuberculosisDataset to torch.utils.data.Dataset, for use in a DataLoader."""

    def __init__(self,
                 x: List[Union[torch.Tensor, np.ndarray]],
                 y: Union[torch.Tensor, np.ndarray],
                 ids: Union[torch.Tensor, np.ndarray],
                 temporal_analysis: bool):
        """
        Args:
            x: (N, D) N sequences of different length having D features, sequence length = 1 if no temporal analysis
            y: (N,) tensor or array with classification labels
            ids: (N,) tensor or array with original identifiers of each sample (admission ids, or stay ids)
            temporal_analysis: True if samples are organized as temporal sequences
        """
        self.x = x
        self.y = y
        self.ids = ids
        self.temporal_analysis = temporal_analysis

    def __len__(self):
        """Returns: the number of samples."""
        return len(self.y)

    def __getitem__(self,
                    indices: Union[int, List[int]]
                    ) -> Tuple[List[torch.Tensor], torch.Tensor, Union[int, List[int]]]:
        """
        Args:
            indices: sample index (or indices) to select
        Returns:
            x, y, indices: per-modality features, labels, and the (unchanged) selected indices
        """
        x = [x_modality[indices] for x_modality in self.x]
        y = self.y[indices]
        return x, y, indices
