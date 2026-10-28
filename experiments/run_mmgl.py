"""
Filename: run_mmgl.py

Author: Hakima Laribi

Description: Evaluates the MMGL baseline. On MIMIC-IV, its graph is also sparsified with top-k.

Example:
    python -m experiments.run_mmgl --task 30dr
"""
from experiments.pipeline import TB_TASKS, load_dataset, parse_args, run_experiment
from hps.hps_config import mmgl_fuser_config, mmgl_trainer_config
from hps.models_hps import mmgl_fuser_hps
from src.models.mmgl import MMGLFuser
from src.trainers.mmgl_training import MMGLTrainer

if __name__ == '__main__':
    args = parse_args('MMGL')

    dataset, masks, n_classes = load_dataset(args.task, modality_miss=args.modality_miss)

    fixed_hps = {"model": MMGLFuser,
                 "temporal_analysis": dataset.temporal_modalities,
                 "batch_size": 1000,
                 "n_epochs": 200,
                 "patience": 50,
                 "device": "cuda",
                 "top_percent": 0.005 if args.task == '30dr' else 1. if args.task in TB_TASKS else 0.05}
    manager_hps = list(mmgl_trainer_config.keys()) + ['model', 'temporal_analysis', 'batch_size', 'device',
                                                      'n_epochs', 'patience']

    run_experiment(args,
                   model_name='MMGL',
                   dataset=dataset,
                   masks=masks,
                   n_classes=n_classes,
                   fixed_hps=fixed_hps,
                   hps_ranges={**mmgl_trainer_config, **mmgl_fuser_config},
                   model_hps=mmgl_fuser_hps,
                   manager_hps=manager_hps,
                   manager_constructor=MMGLTrainer)
