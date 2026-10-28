"""
Filename: run_sgnn.py

Author: Hakima Laribi

Description: Evaluates the SGNN baseline, on a patient population graph built from demographic similarities.

Example:
    python -m experiments.run_sgnn --task 30dr
"""
from experiments.pipeline import MIMIC_TASKS, load_dataset, parse_args, run_experiment
from hps.hps_config import sgnn_fuser_config, trainer_config
from src.models.sgnn import SGNNFuser

if __name__ == '__main__':
    args = parse_args('SGNN')

    dataset, masks, n_classes = load_dataset(args.task, graph=True, modality_miss=args.modality_miss)

    fixed_hps = {"model": SGNNFuser,
                 "distance_modalities": ['demo'] if args.task in MIMIC_TASKS else ['demographic_feat'],
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "patience": 50,
                 "device": "cuda",
                 "top_percent": 0.005 if args.task == '30dr' else 0.05,
                 "aggregator": 'mean'}
    manager_hps = list(trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device', 'n_epochs',
                                                 'patience']

    run_experiment(args,
                   model_name='SGNN',
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**trainer_config, **sgnn_fuser_config},
                   model_hps=list(sgnn_fuser_config.keys()),
                   manager_hps=manager_hps)
