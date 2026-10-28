"""
Filename: hps_config.py

Author: Hakima Laribi

Description: Ray Tune search spaces for trainer and model-architecture hyperparameters.
Each config dict maps parameter names to tune distributions/choices.
"""

from ray import tune


# Trainer configs: optimization, regularization, class imbalance weighting.

trainer_config = {
    "alpha": tune.uniform(0, 0.001),
    "beta": tune.uniform(0, 0.001),
    "lr": tune.uniform(0.0001, 0.01),
    "pos_weight": tune.uniform(0.1, 0.9),
}

digom_trainer_config = {
    "alpha": tune.choice([0.0]),
    "beta": tune.choice([0.0]),
    "lr": tune.uniform(0.0001, 0.01),
    "pos_weight": tune.uniform(0.1, 0.9),
    "lambda_1": tune.uniform(0.01, 0.5),
    "rho": tune.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
}

mmgl_trainer_config = {
    "alpha": tune.choice([0.0]),
    "beta": tune.choice([0.0]),
    "lr": tune.uniform(0.0001, 0.01),
    "pos_weight": tune.uniform(0.1, 0.9),
    "theta_smooth": tune.uniform(0, 1),
    "theta_degree": tune.uniform(0, 0.5),
    "theta_sparsity": tune.uniform(0, 0.001),
}

maxcorrmgnn_trainer_config = {
    "alpha": tune.choice([0.0]),
    "beta": tune.choice([0.0]),
    "lr": tune.uniform(0.0001, 0.01),
    "pos_weight": tune.uniform(0.1, 0.9),
    "lambda_1": tune.uniform(0.01, 0.5),
    "weight_decay": tune.uniform(0.0001, 0.01),
    "weight_decay_hgr": tune.uniform(0.0001, 0.01),
    "lr_hgr": tune.uniform(0.0001, 0.01),
}


# Fuser configs: model architecture (hidden sizes, layer counts, dropout, etc.).

early_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "fuser_n_layers": tune.randint(1, 4),
    "fuser_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
}

late_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "unimodal_n_layers": tune.randint(1, 4),
    "unimodal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
}

joint_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "fuser_n_layers": tune.randint(1, 4),
    "fuser_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "unimodal_n_layers": tune.randint(1, 4),
    "unimodal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
}

sgnn_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "fuser_n_layers": tune.randint(1, 3),
    "fuser_hidden_size": tune.choice([64, 128, 256, 512]),
    "gnn_n_layers": tune.randint(1, 2),
    "gnn_hidden_size": tune.choice([64, 128, 256, 512]),
}

stgnn_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([16, 32, 64, 128]),
    "fuser_n_layers": tune.randint(1, 3),
    "fuser_hidden_size": tune.choice([64, 128, 256, 512]),
    "gnn_n_layers": tune.randint(1, 2),
    "gnn_hidden_size": tune.choice([64, 128, 256, 512]),
}

mmgl_fuser_config = {
    "dropout": tune.uniform(0, 0.2),
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "gnn_hidden_size": tune.randint(1, 3),
    "num_att_heads": tune.randint(1, 10),
    "transformer_hidden_size": tune.choice([16, 32, 64, 128]),
    "n_transformer_layers": tune.randint(0, 1),
}

maxcorrmgnn_fuser_config = {
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "gnn_hidden_size": tune.choice([64, 128, 256]),
    "mlp_proj_size": tune.choice([128, 256, 512]),
}

digom_fuser_config = {
    "temporal_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "num_att_heads": tune.choice([1]),
    "att_hidden_size": tune.choice([64, 128, 256]),
    "gnn_hidden_size": tune.choice([64, 128, 256]),
    "gnn_n_layers": tune.randint(1, 3),
    "temperature": tune.choice([5, 10, 100]),
    "tau": tune.choice([0.01, 0.1, 1.0]),
}
