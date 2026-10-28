"""
Filename: run_baselines.py

Author: Hakima Laribi

Description: Evaluates the Early, Joint and Late Fusion baselines.

Example:
    python -m experiments.run_baselines --model late --task 30dr
"""
from experiments.pipeline import load_dataset, parse_args, run_experiment
from hps.hps_config import early_fuser_config, joint_fuser_config, late_fuser_config, trainer_config
from src.models.baselines import EarlyFuser, JointFuser, LateFuser

FUSERS = {
    'early': ('EarlyFusion', EarlyFuser, early_fuser_config),
    'late': ('LateFusion', LateFuser, late_fuser_config),
    'joint': ('JointFusion', JointFuser, joint_fuser_config),
}

if __name__ == '__main__':
    args = parse_args('Early, Joint and Late Fusion', models=list(FUSERS))
    model_name, fuser, fuser_config = FUSERS[args.model]

    dataset, masks, n_classes = load_dataset(args.task, modality_miss=args.modality_miss)

    fixed_hps = {"model": fuser,
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "patience": 50,
                 "device": "cuda"}
    manager_hps = list(trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device', 'n_epochs',
                                                 'patience']

    run_experiment(args,
                   model_name=model_name,
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**trainer_config, **fuser_config},
                   model_hps=list(fuser_config.keys()),
                   manager_hps=manager_hps)
