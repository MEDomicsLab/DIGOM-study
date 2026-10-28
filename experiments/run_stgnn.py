"""
Filename: run_stgnn.py

Author: Hakima Laribi

Description: Evaluates the STGNN baseline (MIMIC-IV tasks only).

Example:
    python -m experiments.run_stgnn --task 30dr
"""
from experiments.pipeline import MIMIC_TASKS, load_dataset, parse_args, run_experiment
from hps.hps_config import stgnn_fuser_config, trainer_config
from hps.models_hps import STGNNFuser_hps
from src.models.stgnn import STGNNFuser

if __name__ == '__main__':
    args = parse_args('STGNN', tasks=MIMIC_TASKS)

    dataset, masks, n_classes = load_dataset(args.task, graph=True, modality_miss=args.modality_miss)

    fixed_hps = {"model": STGNNFuser,
                 "distance_modalities": ['demo'],
                 "top_percent": 0.005 if args.task == '30dr' else 0.05,
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "patience": 50,
                 "device": "cuda",
                 "gnn_type": 'sage',
                 "aggregator_type": 'mean',
                 "max_seq_len": None}
    manager_hps = list(trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device', 'n_epochs',
                                                 'patience']

    run_experiment(args,
                   model_name='STGNN',
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**trainer_config, **stgnn_fuser_config},
                   model_hps=STGNNFuser_hps,
                   manager_hps=manager_hps)
