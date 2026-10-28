"""
Filename: run_digom.py

Author: Hakima Laribi

Description: Evaluates DIGOM, or one of its interaction type ablations with --ablation.

Example:
    python -m experiments.run_digom --task 30dr
    python -m experiments.run_digom --task 30dr --ablation no_interaction
"""
from ray import tune

from experiments.pipeline import TB_TASKS, load_dataset, parse_args, run_experiment
from hps.hps_config import digom_fuser_config, digom_trainer_config
from hps.models_hps import digom_fuser_hps
from src.models.digom import DIGOMFuser
from src.trainers.digom_training import DIGOMTrainer

ABLATIONS = ['no_interaction', 'intra_patient_cross_modal', 'cross_patient_same_modal', 'cross_patient_cross_modal']


def add_ablation(parser):
    """Adds the --ablation argument, which keeps a single interaction type (or none) in the graph."""
    parser.add_argument('--ablation', choices=ABLATIONS, default=None, help='interaction type kept in the graph')


if __name__ == '__main__':
    args = parse_args('DIGOM', add_arguments=add_ablation)

    dataset, masks, n_classes = load_dataset(args.task, modality_miss=args.modality_miss)

    fixed_hps = {"model": DIGOMFuser,
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "min_epochs": 20 if args.task in TB_TASKS else None,
                 "patience": 5,
                 "device": "cuda",
                 "batch_size_inference": 1000,
                 "warmup_epochs": 15 if args.task == '30dr' else 5,
                 "ablation": args.ablation}
    manager_hps = list(digom_trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device',
                                                     'n_epochs', 'batch_size_inference', 'patience', 'min_epochs',
                                                     'warmup_epochs']

    # sparser target densities for the larger 30-day readmission cohort
    trainer_config = digom_trainer_config | ({"rho": tune.choice([0.001, 0.002, 0.003, 0.004, 0.005])}
                                             if args.task == '30dr' else {})

    if args.output_dir is None and args.ablation is not None:
        args.output_dir = f'records/DIGOM_{args.task}_{args.ablation}'

    run_experiment(args,
                   model_name='DIGOM',
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**trainer_config, **digom_fuser_config},
                   model_hps=digom_fuser_hps,
                   manager_hps=manager_hps,
                   manager_constructor=DIGOMTrainer)
