## DIGOM : Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare
This repository stores the code implemented to generate the results of the paper: *DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare* : [arXiv:2610.32451](https://arxiv.org/abs/2610.32451).

## 1. Data
- **MIMIC-IV** (tasks `30dr`, `mortality_48h`, `disch_alive_48h`): the cohorts must be generated from the MIMIC-IV, MIMIC-CXR-JPG and MIMIC-IV-Note databases, as described in [src/data/README.md](src/data/README.md).
- **NIH-TB** (tasks `tuberculosis`, `tuberculosis_died`): download the preprocessed dataset [TB_data.pickle.zip](https://github.com/IBM/MMMG/blob/main/TB_data.pickle.zip) and unzip it in [csvs/datasets](csvs/datasets):
```
$ unzip TB_data.pickle.zip -d csvs/datasets/
```

## 2. How to use the package?
First, install the requirements under **Python 3.12** as following, then DGL (its CUDA builds are not on PyPI):
```
$ pip install -r requirements.txt
$ pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```
All experiments are run from the root of the repository, with `--task` one of `mortality_48h`, `disch_alive_48h`, `30dr`, `tuberculosis` or `tuberculosis_died`. Each model is evaluated with a nested 5-fold cross-validation, with 100 hyperparameter optimization trials per fold (`--n_trials`). The results (best hyperparameters, trained models, metrics and predictions of each fold, and a `summary.json` across folds) are saved in `records/<model>_<task>`, with `<model>` one of `DIGOM`, `EarlyFusion`, `JointFusion`, `LateFusion`, `STGNN`, `SGNN`, `MMGL`, `MaxCorrMGNN` and `MaxCorrMGNN_topk`.

**Note:** on `30dr`, the graph-based models require about 48 GB of GPU memory.

### 2.1. Training of DIGOM
```
$ python -m experiments.run_digom --task mortality_48h
```

### 2.2. Training of the baselines
```
$ python -m experiments.run_baselines --model early --task mortality_48h      # Early Fusion
$ python -m experiments.run_baselines --model joint --task mortality_48h      # Joint Fusion
$ python -m experiments.run_baselines --model late --task mortality_48h       # Late Fusion
$ python -m experiments.run_stgnn --task mortality_48h                        # STGNN (MIMIC-IV tasks only)
$ python -m experiments.run_sgnn --task mortality_48h                         # SGNN
$ python -m experiments.run_mmgl --task mortality_48h                         # MMGL
$ python -m experiments.run_maxcorrmgnn --task mortality_48h                  # MaxCorrMGNN (top-k on MIMIC-IV)
$ python -m experiments.run_maxcorrmgnn --task tuberculosis --top_k           # MaxCorrMGNN top-k on NIH-TB
```

### 2.3. Ablation of the interaction types
To train DIGOM with a single interaction type, or without any interaction, use `--ablation` with `intra_patient_cross_modal`, `cross_patient_same_modal`, `cross_patient_cross_modal` or `no_interaction`:
```
$ python -m experiments.run_digom --task mortality_48h --ablation cross_patient_cross_modal
```
The results are saved in `records/DIGOM_<task>_<ablation>`.

### 2.4. Robustness to missing modalities
The models trained above are evaluated, without retraining, after masking one modality in the test set. Pass the directory of the trained models with `--best_hps_dir`, add `--load_model`, and give the masked modality with `--modality_miss`:
```
$ python -m experiments.run_digom --task mortality_48h --best_hps_dir records/DIGOM_mortality_48h --load_model --modality_miss cxr
```
The same options apply to every model script. The modalities are:
- `30dr`: `demo`, `cxr`, `dx`, `rad`, `proc`, `lab`, `med`, `disch`
- `mortality_48h`, `disch_alive_48h`: `demo`, `cxr`, `rad`, `proc`, `lab`, `chart`
- `tuberculosis`, `tuberculosis_died`: `demographic_feat`, `ct_feat`, `snp_feat`, `cont_feat`, `regimen_feat`, `clinical_feat`

The results are saved in `records/<model>_<task>_missing_<modality>`.

## Project Tree
```
├── csvs
│   ├── datasets                   <- Cohorts of each task (to be generated or downloaded, see Section 1)
│   └── utils                      <- Mappings and intermediate files used to generate the MIMIC-IV cohorts
├── experiments                    <- Scripts to replicate our experiments
├── hps                            <- Ranges of the hyperparameter values
├── src                            <- All project modules
│   ├── data                       <- Generation of the MIMIC-IV cohorts (see src/data/README.md) and datasets
│   ├── managers
│   │   ├── evaluation.py          <- Nested cross-validation of a model
│   │   └── tuning.py              <- Hyperparameter optimization
│   ├── models
│   │   ├── digom.py               <- DIGOM
│   │   ├── baselines.py           <- Early, Joint and Late Fusion
│   │   ├── stgnn.py               <- STGNN
│   │   ├── sgnn.py                <- SGNN
│   │   ├── mmgl.py                <- MMGL
│   │   └── maxcorrmgnn.py         <- MaxCorrMGNN
│   ├── trainers                   <- Training loops of the models
│   └── utils                      <- Helper functions
├── LICENSE
├── README.md
└── requirements.txt
```
