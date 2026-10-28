"""
Filename: pipeline.py

Author: Hakima Laribi

Description: Shared code of the experiments: loading the dataset of a task, sampling the nested cross-validation
folds, the command line arguments and the evaluation (tuning on the inner folds, then training and testing on the
outer folds) common to all models.
"""
import argparse
import glob
import json
import os
import pickle
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.preprocessing import constants as c
from src.data.preprocessing.dataset import MMDataset, MMTuberculosisDataset
from src.data.preprocessing.graph_dataset import GMMDataset, GMMTuberculosisDataset
from src.data.preprocessing.sampling import load_unique_occurrence, sample_nested_k_folds
from src.managers.evaluation import Evaluator
from src.trainers.training import Trainer
from src.utils.utils import auto_cast

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATASETS_DIR = 'csvs/datasets'

MIMIC_TASKS = ['30dr', 'mortality_48h', 'disch_alive_48h']
TB_TASKS = ['tuberculosis', 'tuberculosis_died']
TASKS = MIMIC_TASKS + TB_TASKS

# Modalities of each task, in the order of the dataset
MODALITIES = {
    '30dr': ['demo', 'cxr', 'dx', 'rad', 'proc', 'lab', 'med', 'disch'],
    'mortality_48h': ['demo', 'cxr', 'rad', 'proc', 'lab', 'chart'],
    'disch_alive_48h': ['demo', 'cxr', 'rad', 'proc', 'lab', 'chart'],
    'tuberculosis': ['demographic_feat', 'ct_feat', 'snp_feat', 'cont_feat', 'regimen_feat', 'clinical_feat'],
    'tuberculosis_died': ['demographic_feat', 'ct_feat', 'snp_feat', 'cont_feat', 'regimen_feat', 'clinical_feat'],
}


def load_mimic_dataset(task: str,
                       graph: bool = False,
                       modality_miss: Optional[str] = None) -> Tuple[Union[MMDataset, GMMDataset], list, list]:
    """
    Loads the cohort of a MIMIC task, keeping one admission (30dr) or ICU stay (ICU tasks) per patient.

    Args:
        task: '30dr', 'mortality_48h' or 'disch_alive_48h'
        graph: if True, returns a graph dataset
        modality_miss: modality simulated as missing

    Returns:
        dataset, sample ids, targets
    """
    if task == '30dr':
        df_cxr = pd.read_csv(f'{DATASETS_DIR}/30dr_cxr.csv')
        df_static = pd.read_csv(f'{DATASETS_DIR}/30dr_notes_ehr_demo.csv')
        id_cols = ['subject_id', 'hadm_id']
        unique_by = 'hadm_id'
        demo_cols = c.DEMO + ['length_of_stay']
    else:
        df_cxr = pd.read_csv(f'{DATASETS_DIR}/icu_cxr_24h_mode.csv')
        df_static = pd.read_csv(f'{DATASETS_DIR}/icu_rad_demo_events_24h_mode.csv')
        id_cols = ['subject_id', 'hadm_id', 'stay_id']
        unique_by = 'stay_id'
        demo_cols = c.DEMO

    # Keep a single admission/stay per patient, sampled on the first run and saved for the next ones
    unique_ids = load_unique_occurrence(f'{DATASETS_DIR}/{task}_ids.json',
                                        df_static[id_cols].drop_duplicates(),
                                        unique_by,
                                        'subject_id')
    df_cxr = df_cxr[df_cxr[unique_by].isin(unique_ids)].reset_index(drop=True)
    df_static = df_static[df_static[unique_by].isin(unique_ids)].reset_index(drop=True)
    assert set(df_cxr[unique_by]) == set(df_static[unique_by]), "ids do not match between datasets"

    # Get unique ids and targets
    y = df_static[id_cols + [task]].drop_duplicates()[task].values.tolist()
    ids = df_static[id_cols].drop_duplicates()[unique_by].values.tolist()
    target = df_static[id_cols + [task]].drop_duplicates()

    # Get the modalities
    def columns(candidates: List[str]) -> List[str]:
        return [col for col in candidates if col in df_static.columns]

    df_modalities = {
        'demo': df_static[id_cols + demo_cols].drop_duplicates(),
        'cxr': df_cxr[id_cols + [f'feature_{i}' for i in range(c.N_IMG_FEAT_ICU)]],
        'rad': df_static[id_cols + [f'rad_{i}' for i in range(c.N_NOTE_FEAT)]],
    }
    if task == '30dr':
        df_modalities['dx'] = df_static[id_cols + columns(c.DX_30DR)]
        df_modalities['proc'] = df_static[id_cols + columns(c.PROCEDURE_30DR)]
        df_modalities['lab'] = df_static[id_cols + [f'{lab}{suffix}' for lab in columns(c.LAB_30DR)
                                                    for suffix in ['', '_present']]]
        df_modalities['med'] = df_static[id_cols + columns(c.MED_30DR)]
        df_modalities['disch'] = df_static[id_cols + [f'disch_{i}' for i in range(c.N_NOTE_FEAT)]]
    else:
        df_modalities['proc'] = df_static[id_cols + columns(c.SUM_PROC_ICU)]
        df_modalities['lab'] = df_static[id_cols + columns(c.TS_LAB_ICU)]
        df_modalities['chart'] = df_static[id_cols + columns(c.TS_CHART_ICU)]
    df_modalities = {modality: df_modalities[modality] for modality in MODALITIES[task]}

    # Images are sequences, the other modalities are static
    temporal_modalities = [modality == 'cxr' for modality in df_modalities]

    dataset_constructor = GMMDataset if graph else MMDataset
    dataset = dataset_constructor(df_modalities=df_modalities,
                                  target_column=target,
                                  temporal_analysis=True,
                                  temporal_modalities=temporal_modalities,
                                  to_tensor=True,
                                  id_column=unique_by,
                                  impute_cont_cols='mean',
                                  modality_miss=modality_miss)
    return dataset, ids, y


def load_tb_dataset(task: str,
                    graph: bool = False,
                    modality_miss: Optional[str] = None):
    """
    Loads the tuberculosis dataset.

    Args:
        task: 'tuberculosis' (multiclass treatment outcome) or 'tuberculosis_died' (death)
        graph: if True, returns a graph dataset
        modality_miss: modality simulated as missing

    Returns:
        dataset, sample ids, targets
    """
    with open(f'{DATASETS_DIR}/TB_data.pickle', 'rb') as f:
        df_dict = pickle.load(f)

    # Get modalities with imputation mask
    df_modalities = {key: value[0].astype('float32') for key, value in df_dict.items() if key != 'outcomes'}
    mask_imputation = {key: value[1] for key, value in df_dict.items() if key != 'outcomes'}

    # Get targets, use indexes as ids
    y = df_dict['outcomes'] if task == 'tuberculosis' else df_dict['outcomes'] == 0
    ids = np.arange(y.shape[0])

    dataset_constructor = GMMTuberculosisDataset if graph else MMTuberculosisDataset
    dataset = dataset_constructor(df_modalities=df_modalities,
                                  mask_imputation=mask_imputation,
                                  target_column=y,
                                  to_tensor=True,
                                  modality_miss=modality_miss)
    return dataset, ids, y


def load_dataset(task: str, graph: bool = False, modality_miss: Optional[str] = None):
    """
    Loads the dataset of a task and samples the nested 5-fold cross-validation masks.

    Args:
        task: one of TASKS
        graph: if True, returns a graph dataset
        modality_miss: modality simulated as missing

    Returns:
        dataset, masks, number of classes (None for binary tasks)
    """
    if task in MIMIC_TASKS:
        dataset, ids, y = load_mimic_dataset(task, graph, modality_miss)
    else:
        dataset, ids, y = load_tb_dataset(task, graph, modality_miss)

    n_classes = len(np.unique(y)) if task == 'tuberculosis' else None
    masks = sample_nested_k_folds(ids, y, k=5, valid_size=0.1, inner_k=5, inner_valid_size=0.1)

    return dataset, masks, n_classes


def parse_args(description: str,
               models: Optional[List[str]] = None,
               tasks: List[str] = TASKS,
               add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> argparse.Namespace:
    """
    Args:
        description: description of the experiment
        models: names of the models that can be selected with --model, if any
        tasks: tasks that can be selected with --task
        add_arguments: function adding model-specific arguments to the parser, if any

    Returns:
        the parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    if models is not None:
        parser.add_argument('--model', choices=models, required=True, help='model to evaluate')
    parser.add_argument('--task', choices=tasks, required=True, help='prediction task')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='number of hyperparameter tuning trials per outer fold')
    parser.add_argument('--best_hps_dir', default=None,
                        help='directory of a previous experiment: reuses its fold_{i}/best_hps.json instead of tuning')
    parser.add_argument('--load_model', action='store_true',
                        help='with --best_hps_dir, loads the trained model of each fold instead of training it')
    parser.add_argument('--modality_miss', default=None, help='modality simulated as missing')
    parser.add_argument('--output_dir', default=None, help='directory where the results are saved')
    parser.add_argument('--resume_fold', type=int, default=None, help='outer fold to resume from')
    parser.add_argument('--seed', type=int, default=101)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()

    if args.load_model and args.best_hps_dir is None:
        parser.error('--load_model requires --best_hps_dir')
    if args.modality_miss is not None and args.modality_miss not in MODALITIES[args.task]:
        parser.error(f'--modality_miss must be one of {MODALITIES[args.task]} for task {args.task}')

    return args


def get_fixed_hps(fixed_hps: Dict[str, object],
                  best_hps_dir: Optional[str],
                  n_folds: int) -> Union[Dict[str, object], List[Dict[str, object]]]:
    """
    Args:
        fixed_hps: hyperparameters that are not tuned
        best_hps_dir: directory of a previous experiment whose best hyperparameters are reused, or None to tune
        n_folds: number of outer folds

    Returns:
        fixed_hps if tuning, else for each fold fixed_hps updated with the best hyperparameters of the fold
    """
    if best_hps_dir is None:
        return fixed_hps

    folds_hps = []
    for fold in range(n_folds):
        with open(os.path.join(best_hps_dir, f'fold_{fold}', 'best_hps.json'), 'r') as f:
            best_hps = json.load(f)
        folds_hps.append(fixed_hps | {hp: auto_cast(value) for hp, value in best_hps.items()
                                      if hp not in ['model', 'temporal_analysis']})
    return folds_hps


def get_pretrained_models(best_hps_dir: Optional[str], n_folds: int) -> Optional[List[str]]:
    """
    Args:
        best_hps_dir: directory of a previous experiment, or None
        n_folds: number of outer folds

    Returns:
        the path to the trained model (.pth) of each fold of the experiment, or None if best_hps_dir is None
    """
    if best_hps_dir is None:
        return None

    models = []
    for fold in range(n_folds):
        checkpoints = glob.glob(os.path.join(best_hps_dir, f'fold_{fold}', '*.pth'))
        if len(checkpoints) != 1:
            raise FileNotFoundError(f'expected one .pth file in {best_hps_dir}/fold_{fold}, found {len(checkpoints)}')
        models.append(checkpoints[0])
    return models


def run_experiment(args: argparse.Namespace,
                   model_name: str,
                   dataset,
                   masks,
                   n_classes: Optional[int],
                   fixed_hps: Dict[str, object],
                   hps_ranges: Dict[str, object],
                   model_hps: List[str],
                   manager_hps: List[str],
                   manager_constructor=Trainer):
    """
    Evaluates a model with nested cross-validation: hyperparameters are tuned on the inner folds of each outer fold
    (or reused from args.best_hps_dir), then the model is trained and tested on the outer fold. With args.load_model,
    the models trained in args.best_hps_dir are loaded instead, and only evaluated (e.g. with a missing modality).

    Args:
        args: command line arguments (see parse_args)
        model_name: name of the model, used to name the output directory
        dataset: dataset of the task
        masks: nested cross-validation masks
        n_classes: number of classes, None for binary tasks
        fixed_hps: hyperparameters that are not tuned
        hps_ranges: search space of the tuned hyperparameters
        model_hps: hyperparameters passed to the model
        manager_hps: hyperparameters passed to the training manager
        manager_constructor: training manager class
    """
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f'records/{model_name}_{args.task}'
        if args.modality_miss is not None:
            output_dir += f'_missing_{args.modality_miss}'

    print(f'Running {model_name} for task {args.task} ..')
    evaluator = Evaluator(dataset=dataset,
                          masks=masks,
                          path_to_experiment=output_dir,
                          hps_ranges=hps_ranges,
                          model_hps=model_hps,
                          manager_hps=manager_hps,
                          seed=args.seed,
                          n_trials=0 if args.best_hps_dir is not None else args.n_trials,
                          fixed_hps=get_fixed_hps(fixed_hps, args.best_hps_dir, len(masks)),
                          resume_fold=args.resume_fold,
                          manager_constructor=manager_constructor,
                          load_model=get_pretrained_models(args.best_hps_dir, len(masks)) if args.load_model else None,
                          n_classes=n_classes)
    evaluator.evaluate()
