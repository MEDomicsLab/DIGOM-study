"""
Filename: models_hps.py

Author: Hakima Laribi

Description: Per-model constructor-argument whitelists. Each list names the keys, from the
flat hyperparameter dict used by Tuner and Evaluator (src/managers/tuning.py, evaluation.py),
to forward to the matching fuser model's `__init__`. Search spaces for these hyperparameters
are defined separately in hps/hps_config.py.
"""

LateFuser_hps = [
    'input_sizes', 'temporal_analysis', 'dropout', 'device', 'temporal_hidden_size',
    'unimodal_n_layers', 'unimodal_hidden_size',
]

EarlyFuser_hps = [
    'input_sizes', 'temporal_analysis', 'dropout', 'device', 'temporal_hidden_size',
    'fuser_n_layers', 'fuser_hidden_size',
]

JointFuser_hps = [
    'input_sizes', 'temporal_analysis', 'dropout', 'device', 'temporal_hidden_size',
    'fuser_n_layers', 'fuser_hidden_size', 'unimodal_n_layers', 'unimodal_hidden_size',
]

STGNNFuser_hps = [
    'input_sizes', 'gnn_hidden_size', 'temporal_hidden_size', 'max_seq_len',
    'dropout', 'gnn_n_layers', 'gnn_type', 'aggregator_type', 'num_heads',
]

mmgl_fuser_hps = [
    'input_sizes', 'dropout', 'gnn_hidden_size', 'temporal_hidden_size',
    'num_att_heads', 'transformer_hidden_size', 'n_transformer_layers', 'top_percent',
]

maxcorrmgnn_fuser_hps = [
    'input_sizes', 'gnn_hidden_size', 'temporal_hidden_size', 'mlp_proj_size', 'k',
]

digom_fuser_hps = [
    'num_att_heads', 'att_hidden_size', 'gnn_hidden_size', 'temporal_hidden_size',
    'gnn_n_layers', 'temperature', 'ablation', 'tau',
]
