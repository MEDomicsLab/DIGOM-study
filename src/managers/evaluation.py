"""
Filename: evaluation.py

Author: Hakima Laribi

Description: Outer-loop evaluation of multimodal models. For each outer fold, the Evaluator optionally tunes
hyperparameters on the inner folds, trains the model with the best hyperparameters, predicts on the train/valid/test
sets, records metrics and predictions per fold, and writes a mean +- std summary across folds.
"""
import gc
import json
import os
import pickle
import time
import warnings
from os import path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
from dgl import DGLGraph
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from src.data.preprocessing.dataset import MMDataset
from src.data.preprocessing.graph_dataset import GraphDatasetMixin
from src.managers.tuning import Tuner
from src.trainers.digom_training import DIGOMTrainer
from src.trainers.mmgl_training import MMGLTrainer
from src.trainers.maxcorrmgnn_training import MaxCorrMGNNTrainer
from src.trainers.training import Trainer
from src.utils.utils import set_seed

warnings.filterwarnings("ignore", category=ExperimentalWarning)


class Evaluator:
    """Runs nested cross-validation: per-fold tuning, training, inference and metrics recording."""

    def __init__(self,
                 dataset: MMDataset,
                 masks: Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]],
                 fixed_hps: Dict[str, object],
                 n_trials: int,
                 path_to_experiment: str,
                 seed: int,
                 hps_ranges: Optional[Dict[str, object]] = None,
                 model_hps: Optional[List[str]] = None,
                 manager_hps: Optional[List[str]] = None,
                 resume_fold: int = None,
                 manager_constructor=Trainer,
                 load_model: str = None,
                 n_classes: int = None
                 ):
        """
        Args:
            dataset: multimodal dataset to evaluate on
            masks: per-fold {'train': ids, 'valid': ids, 'test': ids, 'inner': inner-fold masks for tuning}
            fixed_hps: hyperparameters not tuned (or a list with one dict per fold when n_trials == 0)
            n_trials: number of tuning trials per fold; 0 disables tuning
            path_to_experiment: directory where fold records and the summary are written
            seed: random seed used for tuning and training
            hps_ranges: search space for tuning; required if n_trials > 0
            model_hps: names of hyperparameters passed to the model
            manager_hps: names of hyperparameters passed to the training manager
            resume_fold: fold to resume from; previous folds are loaded from their records.json
            manager_constructor: training manager class to instantiate per fold
            load_model: per-fold paths to pretrained models; if provided, training is skipped
            n_classes: number of classes for multiclass tasks; None for binary classification
        """
        # set protected attributes
        self._dataset = dataset
        self._masks = masks
        self._fixed_hps = fixed_hps
        self._hps_ranges = hps_ranges
        self._path_to_experiment = path_to_experiment
        self._n_trials = n_trials
        self._seed = seed
        self._model_hps = model_hps
        self._manager_hps = manager_hps
        self._resume_fold = resume_fold
        self._manager_constructor = manager_constructor
        self._load_model = load_model
        self.n_classes = n_classes

        # Validate inputs
        if (self._n_trials > 0) & (self._hps_ranges is None):
            raise ValueError('If n_trials > 0, hps_ranges must be provided')

        # Set recorder
        self._recording_files = []
        self._folder_file = None

    def evaluate(self):
        """
        Runs the evaluation over all outer folds, saving per-fold best_hps.json and records.json,
        then writes summary.json in the experiment directory.
        """
        start_eval = False
        for fold_pos, (fold, mask) in enumerate(self._masks.items()):
            if self._resume_fold is not None:
                if (fold != self._resume_fold) & (not start_eval):
                    # read the recording file
                    with open(path.join(self._path_to_experiment, f'fold_{fold}', 'records.json'), 'r') as f:
                        record_f = json.load(f)
                    self._recording_files.append(record_f)
                    print(f'Skipping fold {fold} ..')
                    continue
                else:
                    start_eval = True
            # Create experiment fold
            path_exp = path.join(self._path_to_experiment, f'fold_{fold}')
            os.makedirs(path_exp, exist_ok=True)

            # get mask ids
            train, test, valid = mask['train'], mask['test'], mask['valid']

            if self._n_trials > 0:
                tuner = Tuner(dataset=self._dataset,
                              hps_config=self._hps_ranges,
                              fixed_hps=self._fixed_hps,
                              model_hps=self._model_hps,
                              manager_hps=self._manager_hps,
                              metric={'function': roc_auc_score, 'direction': 'max'},
                              masks=mask['inner'],
                              trainer_constructor=Trainer,
                              path_to_experiment=path_exp,
                              seed=self._seed,
                              verbose=True,
                              manager_constructor=self._manager_constructor,
                              n_classes=self.n_classes)

                # Concatenate hps, if a hp is provided in fixed_hps and in hps_ranges, we consider the tuned hp
                f = tuner.tune(n_trials=self._n_trials, save_figures=True)
                best_params = self._fixed_hps | f
                # free memory
                del tuner
                torch.cuda.empty_cache()

            else:
                best_params = self._fixed_hps[fold] if isinstance(self._fixed_hps, list) else self._fixed_hps

            # save best params
            best_params_f = {k: str(v) for k, v in best_params.items()}
            with open(path.join(path_exp, 'best_hps.json'), "w") as f:
                json.dump(best_params_f, f, indent=True)

            # Reset outer masks
            if not isinstance(self._dataset, GraphDatasetMixin):
                self._dataset.set_masks(train_ids=train, valid_ids=valid, test_ids=test)

            else:
                self._dataset.set_masks(train_ids=train,
                                        valid_ids=valid,
                                        test_ids=test,
                                        top_percent=best_params['top_percent'],
                                        distance_modalities=best_params['distance_modalities'])

            # Get the model hps and manager hps
            model_params = {hps: value for hps, value in best_params.items() if hps in self._model_hps}
            model_params['input_sizes'] = [x[0].shape[1] if isinstance(x, list) else x.shape[1] for x in
                                           self._dataset.x.values()]
            manager_params = {hps: value for hps, value in best_params.items() if hps in self._manager_hps}

            # Create the training manager,
            set_seed(self._seed)

            train_manager = self._manager_constructor(**manager_params,
                                                      model_params=model_params,
                                                      path_to_experiment=path_exp,
                                                      load_model=self._load_model[fold] if self._load_model is not None else None,
                                                      n_classes=self.n_classes)
            if self._load_model is None:
                # fit the model
                print(f'Training at fold {fold} ..')
                train_manager.fit(self._dataset)
            # initialize saving file and add the used hyperparameters
            file = self.init_records_file(fold)

            file['hyperparameters'] = {key: round(value, 5) if isinstance(value, float) | isinstance(value, int)
            else str(value) for key, value in best_params.items()}

            # for each mask set, predict probabilities, compute metrics, save in json
            for mask_type, mask_ids in mask.items():
                if mask_type != 'inner':
                    # Get the mask ids of the current mask type
                    mask_idx = getattr(self._dataset, f'{mask_type}_indexes')

                    # Get inference configuration
                    predict_proba, args, indexes = self.get_inference_config(train_manager,
                                                                             mask_type,
                                                                             mask_idx,
                                                                             mask_ids,
                                                                             mask)

                    # Get the feature we need for prediction, and the targets of samples of the current mask
                    x, _ = self._dataset[indexes]
                    _, y = self._dataset[mask_idx]

                    # predict probabilities
                    begin = time.time()
                    proba, g = predict_proba(x, **args)
                    end = time.time()
                    proba = proba[-len(mask_ids):]
                    y, proba = y.cpu().numpy(), proba.cpu().numpy()

                    # use train data to find the best threshold
                    if mask_type == 'train':
                        threshold = train_manager.find_optimal_threshold(proba, y) if self.n_classes is None else -1
                        file['info']['threshold'] = round(float(threshold), 3)

                    # Get the threshold and class predicted
                    if self.n_classes is None:
                        preds = (proba >= threshold).astype(int)
                    else:
                        preds = np.argmax(proba, axis=1)

                    # add running time
                    file = self.record_hps(file, f'time_{mask_type}', end-begin)

                    # add graph characteristics
                    if isinstance(g, DGLGraph):
                        file = self.record_hps(file, f'n_edges_{mask_type}', int(g.num_edges() / g.num_nodes()))

                    if hasattr(train_manager._model, 'cross_modal'):
                        file = self.record_hps(file, f'A_edges_{mask_type}', int(train_manager._model.A_edges))
                        file = self.record_hps(file, f'C_edges_{mask_type}', int(train_manager._model.C_edges))
                        file = self.record_hps(file, f'n1_edges_{mask_type}', int(train_manager._model.n1_edges))
                        file = self.record_hps(file, f'n2_edges_{mask_type}', int(train_manager._model.n2_edges))

                    if hasattr(train_manager._model, 'n_con') & (mask_type == 'test'):
                        # save the dictionary as a pickle file
                        with open(path.join(path_exp,"n_connections.pkl"), "wb") as f:
                            pickle.dump((train_manager._model.n_con, y), f)

                    # Compute metrics
                    if self.n_classes is None:
                        metrics = self.compute_binary_class_metrics(y, proba, preds)
                    else:
                        metrics = self.compute_multilabel_class_metrics(y, preds, proba, self.n_classes)
                        proba = np.max(proba, axis=1)
                    #  generate metrics file
                    file = self.record_metrics_and_predictions(y, mask_ids, preds, proba, mask_type, metrics, file)

                    del x, y, proba, preds, metrics
                    torch.cuda.empty_cache()

            # save metrics file
            self._recording_files.append(file)
            self.save_metrics_file(path_exp, fold_pos)

            # free memory
            del train_manager
            gc.collect()
            torch.cuda.empty_cache()

        # generate summary
        self.record_summary()

    def get_inference_config(self, train_manager, mask_type, mask_idx, mask_ids, mask):
        """
        Selects the prediction function, its extra arguments and the sample indexes needed for inference.

        Args:
            train_manager: fitted training manager
            mask_type: 'train', 'valid' or 'test'
            mask_idx: dataset indexes of the samples in the current mask
            mask_ids: sample ids of the current mask
            mask: full fold mask dict ('train', 'valid', 'test', ...)

        Returns:
            (predict_proba callable, kwargs dict for predict_proba, indexes of the features to feed to the model)
        """
        # Get the idx of all sample features we will need for prediction (relevant for graph datasets)
        if isinstance(train_manager, MMGLTrainer) | isinstance(train_manager, MaxCorrMGNNTrainer) | (
        isinstance(train_manager, DIGOMTrainer)):
            if mask_type == 'valid':
                indexes = self._dataset.train_indexes + mask_idx
            elif mask_type == 'test':
                indexes = self._dataset.train_indexes + self._dataset.valid_indexes + mask_idx
            else:
                indexes = mask_idx
            predict_proba = train_manager.predict_proba
            args = {'lengths': None,
                    'idx_anchors': [i for i in range(len(mask['train']))] if mask_type == 'valid'
                    else [i for i in range(len(mask['train'] + mask['valid']))] if mask_type == 'test'
                    else None}

        else:
            if isinstance(self._dataset, GraphDatasetMixin):
                indexes = self.get_neighbors_idx(mask_type)
                mask_graph = getattr(self._dataset, f'{mask_type}_graph')
                args = {'g': mask_graph}
                nodes_ids = mask_graph.nodes()[-len(mask_ids):]
                predict_proba = train_manager.predict_proba_with_graphs(nodes_ids)

            else:
                args = {'g': None}
                indexes = mask_idx
                predict_proba = train_manager.predict_proba

        return predict_proba, args, indexes

    def record_hps(self, file, hps_name, hps_value):
        """
        Args:
            file: record dict of the current fold
            hps_name: key to store under file['info']
            hps_value: value to store

        Returns:
            the updated record dict
        """
        file['info'][hps_name] = hps_value
        return file

    def record_summary(self):
        """Writes summary.json with the mean +- std of each metric and numeric hyperparameter across folds."""
        summary = {}
        file = self._recording_files[0]
        for section_name, section_content in file.items():
            if ('metrics' in section_name) | (section_name == 'hyperparameters'):
                summary[section_name] = {}
                for metric, value in section_content.items():
                    # do not report str hps since they are not optimized
                    if not isinstance(value, str) & (section_name == 'hyperparameters'):
                        try:
                            mean = np.round(np.mean([f[section_name][metric] for f in self._recording_files]), 4)
                            std = np.round(np.std([f[section_name][metric] for f in self._recording_files]), 4)
                            summary[section_name][metric] = f'{mean} +- {std}'
                        except:
                            pass

        with open(path.join(self._path_to_experiment, 'summary.json'), "w") as f:
            json.dump(summary, f, indent=True)

    def init_records_file(self,
                          fold: int):
        """
        Args:
            fold: fold index, used to report the size of each mask

        Returns:
            a new record dict with the 'info' section filled with dataset and fold characteristics
        """
        file = {'info': {'task': self._dataset.task,
                         'id': self._dataset.id_column,
                         'analysis': 'temporal' if self._dataset.temporal_analysis else 'static',
                         'modalities': list(self._dataset.modalities),
                         'analysis_per_modality': ['temporal' if temp else 'static'
                                                   for temp in self._dataset.temporal_modalities],
                         'n_train': len(self._masks[fold]['train']),
                         'n_valid': len(self._masks[fold]['valid']),
                         'n_test': len(self._masks[fold]['test'])}}

        return file

    def save_metrics_file(self, path_exp, fold):
        """
        Writes the record of a fold to records.json with sections in a fixed order.

        Args:
            path_exp: directory of the fold
            fold: position of the fold in the recording files
        """
        # reorder columns
        ordered_cols = ['info', 'hyperparameters'] + [f'{mask_type}_{section}' for section in ['metrics', 'predictions']
                                                      for mask_type in self._masks[fold].keys() if mask_type != 'inner']
        recorded_file = {key: self._recording_files[fold][key] for key in ordered_cols}
        with open(path.join(path_exp, 'records.json'), "w") as f:
            json.dump(recorded_file, f, indent=True)

    @staticmethod
    def compute_binary_class_metrics(targets: np.ndarray,
                                     proba: np.ndarray,
                                     preds):
        """
        Args:
            targets: (N,) binary ground truth
            proba: (N,) predicted probabilities of the positive class
            preds: (N,) binary predictions

        Returns:
            dict of prevalence, AUROC, AUPRC, sensitivity, specificity, precision and NPV
        """
        # Confusion matrix: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        # Compute metrics
        metrics = {
            'Prevalence': round(float(targets.sum() / targets.shape[0]), 5),
            'AUROC': round(float(roc_auc_score(targets, proba)), 5),
            'AUPRC': round(float(average_precision_score(targets, proba)), 5),
            'Sensitivity': round(float(tp / (tp + fn)), 5) if (tp + fn) > 0 else 0.0,
            'Specificity': round(float(tn / (tn + fp)), 5) if (tn + fp) > 0 else 0.0,
            'Precision': round(float(tp / (tp + fp)), 5) if (tp + fp) > 0 else 0.0,
            'NPV': round(float(tn / (tn + fn)), 5) if (tn + fn) > 0 else 0.0
        }
        return metrics

    @staticmethod
    def compute_multilabel_class_metrics(targets, preds, proba, n_classes):
        """
        Computes one-vs-rest metrics per class, plus their prevalence-weighted averages.

        Args:
            targets: (N,) class labels
            preds: (N,) predicted class labels
            proba: (N, L) predicted probabilities per class
            n_classes: number of classes L

        Returns:
            dict of per-class metrics (suffixed _Class{l}) and weighted AUROC, AUPRC, sensitivity,
            specificity, precision and NPV
        """
        targets = label_binarize(targets, classes=np.arange(n_classes))

        N, L = targets.shape

        # sklearn multilabel weighted = uniform weights (each label has N samples)
        weights = np.sum(targets, axis=0).squeeze() / N

        # Initialize full dictionary
        metrics = {}

        # Arrays for weighted computation
        prevalences = []
        aucs = []
        auprcs = []
        sensitivites = []
        specificities = []
        precisions = []
        npvs = []

        for l in range(L):
            y_true = targets[:, l]
            y_pred = preds == l
            y_prob = proba[:, l]

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Class-wise metrics
            metrics[f'Prevalence_Class{l}'] = round(y_true.mean(), 5)
            metrics[f'AUROC_Class{l}'] = round(roc_auc_score(y_true, y_prob), 5)
            metrics[f'AUPRC_Class{l}'] = round(average_precision_score(y_true, y_prob), 5)
            metrics[f'Sensitivity_Class{l}'] = round(float(tp / (tp + fn)), 5) if (tp + fn) > 0 else 0.0
            metrics[f'Specificity_Class{l}'] = round(float(tn / (tn + fp)), 5) if (tn + fp) > 0 else 0.0
            metrics[f'Precision_Class{l}'] = round(float(tp / (tp + fp)), 5) if (tp + fp) > 0 else 0.0
            metrics[f'NPV_Class{l}'] = round(float(tn / (tn + fn)), 5) if (tn + fn) > 0 else 0.0

            # Save for weighted averaging
            prevalences.append(metrics[f'Prevalence_Class{l}'])
            aucs.append(metrics[f'AUROC_Class{l}'])
            auprcs.append(metrics[f'AUPRC_Class{l}'])
            sensitivites.append(metrics[f'Sensitivity_Class{l}'])
            specificities.append(metrics[f'Specificity_Class{l}'])
            precisions.append(metrics[f'Precision_Class{l}'])
            npvs.append(metrics[f'NPV_Class{l}'])

        # Weighted summary (sklearn-style)
        metrics['AUROC'] = round(float(np.sum(aucs * weights)), 5)
        metrics['AUPRC'] = round(float(np.sum(auprcs * weights)), 5)
        metrics['Sensitivity'] = round(float(np.sum(sensitivites * weights)), 5)
        metrics['Specificity'] = round(float(np.sum(specificities * weights)), 5)
        metrics['Precision'] = round(float(np.sum(precisions * weights)), 5)
        metrics['NPV'] = round(float(np.sum(npvs * weights)), 5)
        return metrics

    @staticmethod
    def record_metrics_and_predictions(targets,
                                       ids,
                                       preds,
                                       probas,
                                       mask_type,
                                       metrics,
                                       file):
        """
        Args:
            targets: ground truth of the mask samples
            ids: sample ids of the mask
            preds: predicted classes
            probas: predicted probabilities
            mask_type: 'train', 'valid' or 'test'
            metrics: metrics dict to record
            file: record dict of the current fold

        Returns:
            the record dict with the '{mask_type}_metrics' and '{mask_type}_predictions' sections added
        """
        file[f'{mask_type}_metrics'] = metrics
        file[f'{mask_type}_predictions'] = {_id: {'target': int(targets[i]),
                                                  'probability': float(probas[i]),
                                                  'prediction': int(preds[i])} for i, _id in enumerate(ids)}
        return file

    def get_neighbors_idx(self, mask_type: str = 'train'):
        """
        Args:
            mask_type: 'train', 'valid' or 'test'

        Returns:
            dataset indexes of the mask samples and all samples of the preceding masks (graph neighbors)
        """
        if mask_type == 'valid':
            indexes = self._dataset.train_indexes + self._dataset.valid_indexes
        elif mask_type == 'test':
            indexes = self._dataset.train_indexes + self._dataset.valid_indexes + self._dataset.test_indexes
        else:
            indexes = self._dataset.train_indexes

        return indexes
