"""
Filename: maxcorrmgnn_training.py

Author: Hakima Laribi

Description: Defines MaxCorrMGNNTrainer. A warm-up phase optimizes only the soft-HGR loss, then the
multiplex graph is built from the projections and the soft-HGR and cross-entropy losses are optimized jointly.

Code adapted to our framework from:
https://github.com/IBM/MMMG/blob/main/Graph_Classification/utils/HGR_modules.py
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import dgl
import torch
from torch import Tensor
from torch.nn import functional as F

from src.trainers.training import Trainer


class MaxCorrMGNNTrainer(Trainer):
    """Trainer of MaxCorrMGNN. See module docstring for details."""

    def __init__(self,
                 model: Callable,
                 temporal_analysis: List[bool],
                 model_params: Dict[str, Any],
                 alpha: float = 0.0001,
                 beta: float = 0.0001,
                 lr: float = 0.001,
                 pos_weight: float = 0.5,
                 batch_size: int = 1000,
                 n_epochs: int = 400,
                 patience: int = 50,
                 device: Literal['cpu', 'cuda'] = 'cuda',
                 path_to_experiment: str = 'records',
                 save_model: bool = True,
                 verbose: bool = True,
                 batch_size_inference: bool = 10000,
                 lambda_1: float = 0.1,
                 weight_decay: float = 0.001,
                 lr_hgr: float = 0.001,
                 weight_decay_hgr: float = 0.001,
                 n_classes=None,
                 load_model=None
                 ):
        """
        Args:
            batch_size_inference: batch size used for graph-based inference
            lambda_1: tradeoff weight between the HGR and cross-entropy losses in the main phase
            weight_decay: weight decay for the main-phase (MaxCorrMGNN) optimizer
            lr_hgr: learning rate for the warmup-phase (HGR) optimizer
            weight_decay_hgr: weight decay for the warmup-phase (HGR) optimizer
            See Trainer.__init__ for the remaining args.
        """
        super().__init__(model, temporal_analysis, model_params, alpha, beta, lr, pos_weight, batch_size, n_epochs,
                         patience, device, path_to_experiment, save_model, verbose, batch_size_inference,
                          n_classes=n_classes)
        if load_model is not None:
            self._model.gnn_predictor()  # initialize the GNN predictor part of the model to load the state dict
            self._model.load_state_dict(torch.load(load_model, map_location=device, weights_only=True))

        self._lr_hgr = lr_hgr
        self.weight_decay_hgr = weight_decay_hgr
        self.weight_decay = weight_decay
        self.lambda_1 = lambda_1
        self.prepare_valid_data = self.prepare_valid_data_for_adaptive_learning
        self.warmup_epochs = 50  # number of epochs to warmup HGR only

        # the elastic-net penalty is disabled here; the HGR loss is added separately
        self._alpha, self._beta = 0, 0

        # optimizers are created lazily in train_one_epoch, once per phase (HGR warmup, then MaxCorrMGNN)
        self._hgr_optimizer = None
        self._maxcorrmgnn_optimizer = None
        self.h_mod_train = None
        self.log_train = None

    def train_one_epoch(self, dataloader, x_train, w_train, y_train, lengths_train, epoch):
        """
        Runs one training epoch. Before `warmup_epochs`, optimizes only the soft HGR loss across
        modality projections (predictions are random placeholders, no graph is built). From
        `warmup_epochs` onward, builds a multiplex graph from the HGR projections and optimizes
        a weighted combination of the HGR loss and the prediction (cross-entropy) loss.
        Args:
            dataloader: training DataLoader
            x_train: per-modality training input tensors (unused; kept for API parity)
            w_train: per-sample training weights
            y_train: training labels
            lengths_train: per-modality original sequence lengths
            epoch: current epoch index (0-based)
        """
        # reset GNN/sparsity/combination params and switch optimizer at each phase transition
        if (epoch == 0) | (epoch == self.warmup_epochs):
            if epoch == 0:
                self._model.max_corr_sparsify()
                self._model.gnn_predictor()
                self._hgr_optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr_hgr,
                                                        weight_decay=self.weight_decay_hgr)
            else:
                self._model.max_corr_sparsify()
                self._model.gnn_predictor()
                self._maxcorrmgnn_optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr,
                                                          weight_decay=self.weight_decay)
                self._hgr_optimizer.state.clear()  # free memory
                torch.cuda.empty_cache()
                del self._hgr_optimizer

        if epoch < self.warmup_epochs:
            lambda_1 = 1.0
            optimizer = self._hgr_optimizer
            self._patience_counter = 0  # no early stopping during warmup
        else:
            lambda_1 = self.lambda_1
            optimizer = self._maxcorrmgnn_optimizer

        logits_shape = y_train.shape[0] if self.n_classes is None else (y_train.shape[0], self.n_classes)
        self.log_train = torch.zeros(logits_shape, device=self._device)
        self.idx_train = []

        self._model.train()
        for b_i, train_data in enumerate(dataloader):
            x, y, idx = train_data
            y = y.to(self._device)
            lengths = [l_mod[idx] if l_mod is not None else None for l_mod in lengths_train]

            h_mod, h_hgr = self._model.hgr_forward(x, lengths)
            sgl = self.soft_hgr_loss(h_mod, y.shape[0])

            if lambda_1 == 1:  # HGR warmup: learn only good projections, no graph or GNN used
                loss = sgl
                if self.n_classes is not None:
                    logits = 10 * torch.rand(y.shape[0], self.n_classes, device=self._device)
                else:
                    logits = 10 * torch.rand(y.shape[0], device=self._device)
            else:
                # build the multiplex graph from this batch's HGR projections only
                g1, g2 = self._model.create_multigraph(h_hgr, anchor_nodes=None)
                h_gnn = self._model.gnn_forward(h_hgr, h_hgr, g1, g2)
                logits = self._model.modality_prediction_fusion(h_gnn)

                cel = self.loss(logits, y, w_train[idx])
                loss = lambda_1 * sgl + (1 - lambda_1) * cel
                del g1, g2
                torch.cuda.empty_cache()

            self.log_train[idx] = logits.squeeze().detach()
            if self.h_mod_train is None:
                self.h_mod_train = {key: torch.zeros((y_train.shape[0], h_mod[key].shape[1]), device='cpu')
                                    for key in h_mod.keys()}
            for key in self.h_mod_train.keys():
                self.h_mod_train[key][idx] = h_mod[key].detach().to('cpu')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate_train_valid_epoch(self, x_train, y_train, w_train, lengths_train, g_train, x_valid, lengths_valid,
                                   g_valid, y_valid, w_valid, dataset, epoch):
        """
        Computes training and validation HGR and (once past warmup) prediction losses and
        metrics for one epoch. Training predictions/representations are reused from the
        logging done in `train_one_epoch`, since recomputing them would be redundant.
        Args:
            x_train: per-modality training input tensors
            y_train: training labels
            w_train: per-sample training weights
            lengths_train: per-modality training sequence lengths
            g_train: unused; kept for API parity
            x_valid: per-modality validation input tensors
            lengths_valid: per-modality validation sequence lengths
            g_valid: unused; kept for API parity
            y_valid: validation labels
            w_valid: per-sample validation weights
            dataset: unused; kept for API parity
            epoch: current epoch index (0-based)
        Returns:
            train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid
        """
        self._model.eval()
        with torch.no_grad():
            log_train, h_mod_train = self.log_train, self.h_mod_train
            log_valid, h_mod_valid, _ = self._model(x_valid,
                                                    lengths_valid,
                                                    idx_anchors=[i for i in range(len(x_train[0]))],
                                                    graph_loader_batch_size=self.batch_size_inference,
                                                    graph_learning=epoch >= self.warmup_epochs
                                                    )
            log_valid = log_valid[-len(y_valid):] if log_valid is not None else None
            h_mod_valid = {key: value[-len(y_valid):] for key, value in h_mod_valid.items()}

        lambda_1 = 1.0 if epoch < self.warmup_epochs else self.lambda_1
        train_loss = lambda_1 * self.soft_hgr_loss(h_mod_train, y_train.shape[0]).item()
        valid_loss = lambda_1 * self.soft_hgr_loss(h_mod_valid, y_valid.shape[0]).item()

        if epoch >= self.warmup_epochs:  # graph learning enabled, add the prediction loss and metrics
            if self.n_classes is None:
                p_train, p_valid = torch.sigmoid(log_train), torch.sigmoid(log_valid)
            else:
                p_train, p_valid = F.softmax(log_train, dim=1), F.softmax(log_valid, dim=1)

            train_loss += (1 - lambda_1) * self.loss(log_train, y_train.to(self._device), w_train).item()
            valid_loss += (1 - lambda_1) * self.loss(log_valid, y_valid.to(self._device), w_valid).item()

            train_scores = self.compute_metrics_per_epoch(p_train, y_train)
            valid_scores = self.compute_metrics_per_epoch(p_valid, y_valid)
        else:
            train_scores, valid_scores = {'AUROC': 0, 'AUPRC': 0}, {'AUROC': 0, 'AUPRC': 0}
            p_train = torch.rand(y_train.shape[0], device=self._device)
            p_valid = torch.rand(y_valid.shape[0], device=self._device)

        self.h_mod_train = None
        self.log_train = None

        return train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid

    def predict_proba(self,
                      x: List[Union[torch.Tensor, List[torch.Tensor]]],
                      lengths: Optional[List] = None,
                      idx_anchors: Optional[List] = None) -> Tuple[Tensor, dgl.DGLGraph]:
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths; computed from `x` if None
            idx_anchors: indices of training nodes to anchor inference to
        Returns:
            probas, g: predicted class probabilities for `x` through the HGR + multiplex GNN
            pipeline, and the graph built for them
        """
        self._model.eval()
        with torch.no_grad():
            if lengths is None:
                x, lengths = self.pad_temporal_sequences(x, self._zero_padding)

            logits, _, g = self._model(x,
                                       lengths,
                                       idx_anchors=idx_anchors,
                                       graph_loader_batch_size=self.batch_size_inference,
                                       graph_learning=True)
            probas = torch.sigmoid(logits) if self.n_classes is None else F.softmax(logits, dim=1)

            return probas, g
