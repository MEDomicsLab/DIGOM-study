"""
Filename: tuning.py

Author: Hakima Laribi

Description: Hyperparameter tuning via Ray Tune + Optuna (TPE sampler). Tuner.objective runs a
k-fold cross-validation for one suggested hyperparameter set and reports the mean test metric;
Tuner.tune drives the search and returns the best hyperparameters found. TrialProgress is a Ray
actor tracking a shared trial counter/progress bar across parallel trials.
"""
import gc
import os
import shutil
from typing import Callable, Dict, List, Union

import numpy as np
import optuna
import ray
import torch
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization.matplotlib import plot_optimization_history, plot_parallel_coordinate, plot_param_importances
from ray import tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from src.data.preprocessing.dataset import MMDataset
from src.data.preprocessing.graph_dataset import GraphDatasetMixin
from src.trainers.digom_training import DIGOMTrainer
from src.trainers.mmgl_training import MMGLTrainer
from src.trainers.maxcorrmgnn_training import MaxCorrMGNNTrainer
from src.utils.utils import set_seed


class Tuner:
    """Runs Optuna-driven hyperparameter search (via Ray Tune) over k-fold cross-validation."""

    def __init__(self,
                 dataset: MMDataset,
                 fixed_hps: Dict[str, object],
                 hps_config: Dict[str, object],
                 metric: Dict[str, Union[str, Callable]],
                 masks: Dict[int, Dict[str, List[int]]],
                 trainer_constructor: Callable,
                 path_to_experiment: str,
                 seed: int,
                 model_hps: List[str],
                 manager_constructor: Callable,
                 manager_hps: List[str] = None,
                 verbose: bool = True,
                 n_classes: int=None
                 ):
        """
        Args:
            dataset: dataset to run k-fold cross-validation on (or a Ray object ref to it)
            fixed_hps: hyperparameters not being searched, merged with each suggested trial
            hps_config: Ray Tune search space passed as `param_space`
            metric: {'direction': 'max'|'min', 'function': scoring callable(y, proba, ...)}
            masks: per-fold {'train': ids, 'valid': ids, 'test': ids}
            trainer_constructor: unused; kept for API parity (manager_constructor is used instead)
            path_to_experiment: directory where tuning artifacts (figures) are saved
            seed: random seed for reproducibility
            model_hps: names of the hyperparameters to forward to the model constructor
            manager_constructor: Trainer subclass to instantiate for each trial
            manager_hps: names of the hyperparameters to forward to the trainer constructor
            verbose: if True, prints each trial's score
            n_classes: number of classes; None for binary classification
        """
        self._dataset = dataset
        self._metric_direction = metric['direction']
        self._metric = metric['function']
        self._path_to_experiment = path_to_experiment
        self._masks = masks
        self._hps_config = hps_config
        self._trainer_constructor = trainer_constructor
        self._seed = seed
        self._verbose = verbose
        self._temporal_analysis = dataset.temporal_analysis
        self._model_hps = model_hps
        self._manager_hps = manager_hps
        self._manager_constructor = manager_constructor
        self.n_classes = n_classes
        # reduce patience, don't fix a minimum number of epochs, look for hps which leads to convergence faster
        self._fixed_hps = {key: None if key == 'min_epochs' else min(15, value) if key == 'patience'
                           else 'cuda' if ((key == 'device') & (value != 'cpu')) else value
                           for key, value in fixed_hps.items()}

        self._progress_trials = None

    def objective(self, suggested_hps):
        """
        Runs a k-fold cross-validation for one suggested hyperparameter set and reports the
        mean test metric to Ray Tune.
        Args:
            suggested_hps: hyperparameters suggested by the search algorithm for this trial
        """
        set_seed(self._seed)

        dataset = ray.get(self._dataset)
        masks = ray.get(self._masks)
        trainer_constructor = ray.get(self._manager_constructor)

        # run a k-fold cross-validation using the suggested hps
        metric_vals = []
        for fold, mask in masks.items():
            train, test, valid = mask['train'], mask['test'], mask['valid']

            hps = {**suggested_hps, **self._fixed_hps}

            if not isinstance(dataset, GraphDatasetMixin):
                dataset.set_masks(train_ids=train, valid_ids=valid, test_ids=test)

            else:
                dataset.set_masks(train_ids=train,
                                  valid_ids=valid,
                                  test_ids=test,
                                  top_percent=hps['top_percent'],
                                  distance_modalities=hps['distance_modalities'])

            model_params = {hps: value for hps, value in hps.items() if hps in self._model_hps}
            model_params['input_sizes'] = [x[0].shape[1] if isinstance(x, list) else x.shape[1] for x in
                                           dataset.x.values()]
            manager_params = {hps: value for hps, value in hps.items() if hps in self._manager_hps}

            train_manager = trainer_constructor(**manager_params,
                                                model_params=model_params,
                                                path_to_experiment=self._path_to_experiment,
                                                save_model=False,
                                                verbose=False,
                                                n_classes=self.n_classes)
            train_manager.fit(dataset)

            # adaptive-learning trainers need idx_anchors to predict on train+valid+test nodes at once
            if isinstance(train_manager, (MMGLTrainer, MaxCorrMGNNTrainer, DIGOMTrainer)):
                indexes = dataset.train_indexes + dataset.valid_indexes + dataset.test_indexes
                predict_proba = train_manager.predict_proba
                args = {'idx_anchors': [i for i in range(len(mask['train'] + mask['valid']))]}

            else:
                if isinstance(dataset, GraphDatasetMixin):
                    indexes = dataset.train_indexes + dataset.valid_indexes + dataset.test_indexes
                    test_graph = dataset.test_graph
                    nodes_ids = test_graph.nodes()[-len(dataset.test_indexes):]
                    args = {'g': test_graph}
                    predict_proba = train_manager.predict_proba_with_graphs(nodes_ids)

                else:
                    indexes = dataset.test_indexes
                    args = {'g': None}
                    predict_proba = train_manager.predict_proba

            x, _ = dataset[indexes]  # test features and their neighbors' features (case of graphs)
            _, y = dataset[dataset.test_indexes]  # we only need the test targets

            proba, _ = predict_proba(x, **args)
            proba = proba[-len(dataset.test_indexes):]
            y, proba = y.detach().cpu().numpy(), proba.detach().cpu().numpy()

            if self.n_classes is not None:
                y_bin = label_binarize(y, classes=np.arange(self.n_classes))
                metric_val = self._metric(y_bin, proba, multi_class='ovr', average='weighted')
            else:
                metric_val = self._metric(y, proba)
            metric_vals.append(metric_val)

            del train_manager
            del x, y, proba

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        score = np.mean(metric_vals)

        n_trial = ray.get(self._progress_trials.update.remote())
        if self._verbose:
            print(f'Trial {n_trial} finished with score: {score} using hps: {suggested_hps}')

        tune.report({'metric': score})

    def tune(self, n_trials: int, save_figures: bool):
        """
        Runs the hyperparameter search and returns the best hyperparameters found.
        Args:
            n_trials: number of trials (suggested hyperparameter sets) to evaluate
            save_figures: if True, saves Optuna's importance/history/parallel-coordinate plots
                to `self._path_to_experiment`
        Returns:
            the best hyperparameter config found, per `self._metric_direction`
        """
        search_algo = OptunaSearch(metric="metric",
                                   mode=self._metric_direction,
                                   sampler=optuna.samplers.TPESampler(seed=self._seed, n_startup_trials=20))
        scheduler = FIFOScheduler()

        # shared counter/progress bar across parallel trials
        ray.init()
        self._progress_trials = TrialProgress.remote(n_trials=n_trials)

        # put heavy objects in the ray object store once, instead of copying them to every trial
        self._dataset = ray.put(self._dataset)
        self._masks = ray.put(self._masks)
        self._manager_constructor = ray.put(self._manager_constructor)

        if self._fixed_hps['device'] != 'cpu':
            tuner = tune.Tuner(tune.with_resources(self.objective, {"gpu": 1}),
                               tune_config=tune.TuneConfig(search_alg=search_algo,
                                                           scheduler=scheduler,
                                                           num_samples=n_trials,
                                                           max_concurrent_trials=torch.cuda.device_count()),
                               run_config=tune.RunConfig(verbose=0, storage_path=None),
                               param_space=self._hps_config)
        else:
            tuner = tune.Tuner(self.objective,
                               tune_config=tune.TuneConfig(search_alg=search_algo,
                                                           scheduler=scheduler,
                                                           num_samples=n_trials,
                                                           max_concurrent_trials=3),
                               run_config=tune.RunConfig(verbose=0, storage_path=None),
                               param_space=self._hps_config)
        results = tuner.fit()
        ray.shutdown()

        del tuner
        torch.cuda.empty_cache()

        best_result = results.get_best_result(metric="metric", mode=self._metric_direction)
        best_config = best_result.config

        if save_figures:
            study = search_algo._ot_study
            fig = plot_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=self._seed)).get_figure()
            self.save_optimization_figure(fig, 'hps_importances.png')
            fig = plot_optimization_history(study).get_figure()
            self.save_optimization_figure(fig, 'optimization_history.png')
            fig = plot_parallel_coordinate(study).get_figure()
            self.save_optimization_figure(fig, 'parallel_coordinate.png')

        # delete the directories created by ray tune for the hps tuning
        shutil.rmtree(os.path.expanduser('~/ray_results'), ignore_errors=True)

        return best_config

    def save_optimization_figure(self, fig, figure_name: str):
        """
        Args:
            fig: matplotlib figure to save
            figure_name: filename to save it under, in `self._path_to_experiment`
        """
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        fig.savefig(os.path.join(self._path_to_experiment, figure_name))


@ray.remote
class TrialProgress:
    """Ray actor tracking a shared trial counter/progress bar across parallel trials."""

    def __init__(self, n_trials: int = 0):
        """
        Args:
            n_trials: total number of trials, used to size the progress bar
        """
        self.count = 0
        self._progress_bar = tqdm(range(n_trials), disable=False)

    def update(self):
        """
        Returns:
            the updated trial count, after incrementing it and advancing the progress bar by 1
        """
        self.count += 1
        self._progress_bar.update(1)
        return self.count
