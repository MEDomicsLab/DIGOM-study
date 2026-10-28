"""
Filename: run_maxcorrmgnn.py

Author: Hakima Laribi

Description: Evaluates the MaxCorrMGNN baseline. On MIMIC-IV, its graph is always sparsified with top-k; on NIH-TB,
the original method is evaluated, or its top-k variant with --top_k.

Example:
    python -m experiments.run_maxcorrmgnn --task 30dr
    python -m experiments.run_maxcorrmgnn --task tuberculosis --top_k
"""
from experiments.pipeline import TB_TASKS, load_dataset, parse_args, run_experiment
from hps.hps_config import maxcorrmgnn_fuser_config, maxcorrmgnn_trainer_config
from hps.models_hps import maxcorrmgnn_fuser_hps
from src.models.maxcorrmgnn import MaxCorrMGNNFuser
from src.trainers.maxcorrmgnn_training import MaxCorrMGNNTrainer


def add_top_k(parser):
    """Adds the --top_k argument, which sparsifies the graph with top-k on NIH-TB."""
    parser.add_argument('--top_k', action='store_true', help='top-k variant on the tuberculosis tasks')


if __name__ == '__main__':
    args = parse_args('MaxCorrMGNN', add_arguments=add_top_k)

    dataset, masks, n_classes = load_dataset(args.task, modality_miss=args.modality_miss)

    fixed_hps = {"model": MaxCorrMGNNFuser,
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "patience": 5,
                 "device": "cuda",
                 "batch_size_inference": 5000,
                 "k": 0.005 if args.task == '30dr' else 1. if args.task in TB_TASKS and not args.top_k else 0.05}
    manager_hps = list(maxcorrmgnn_trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device',
                                                             'n_epochs', 'batch_size_inference', 'patience']

    run_experiment(args,
                   model_name='MaxCorrMGNN_topk' if args.top_k else 'MaxCorrMGNN',
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**maxcorrmgnn_trainer_config, **maxcorrmgnn_fuser_config},
                   model_hps=maxcorrmgnn_fuser_hps,
                   manager_hps=manager_hps,
                   manager_constructor=MaxCorrMGNNTrainer)
