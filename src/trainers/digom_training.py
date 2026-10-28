"""
Filename: digom_training.py

Author: Hakima Laribi

Description: Defines DIGOMTrainer. During the first warmup_epochs, only the soft-HGR and density
losses are optimized, then the cross-entropy loss is added: (1 - lambda_2) CE + lambda_2 ((1 - lambda_1) sHGR +
lambda_1 density), with lambda_1 = 0.001.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from src.trainers.training import Trainer


class DIGOMTrainer(Trainer):
    """Trainer of DIGOM. See module docstring for details."""

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
                 load_model: str = None,
                 lambda_1: float = 0.1,
                 rho: float = 0.01,
                 warmup_epochs: int = 5,
                 min_epochs: int = None,
                 n_classes: int = None,
                 ):
        """
        Args:
            lambda_1: weight lambda_2 of the paper between the cross-entropy loss and the sHGR/density losses
            rho: target graph density rho of the density loss
            warmup_epochs: number of warm-up epochs L_warmup
            See Trainer.__init__ for the remaining args.
        """
        super().__init__(model, temporal_analysis, model_params, alpha, beta, lr, pos_weight, batch_size, n_epochs,
                         patience, device, path_to_experiment, save_model, verbose, batch_size_inference, min_epochs,
                         load_model, n_classes=n_classes)

        self.prepare_valid_data = self.prepare_valid_data_for_adaptive_learning
        # the elastic-net penalty is disabled here; the HGR and density losses are added separately
        self._alpha, self._beta = 0, 0
        self.lambda_1 = round(lambda_1, 6)
        self.rho = rho
        self.warmup_epochs = warmup_epochs
        self.log_train = None

    def train_one_epoch(self, dataloader, x_train, w_train, y_train, lengths_train, epoch):
        """
        Runs one training epoch. Before `warmup_epochs`, optimizes only the soft HGR loss and the
        graph density regularization. From `warmup_epochs` onward, optimizes a weighted
        combination of the prediction (cross-entropy) loss and those two losses.
        Args:
            dataloader: training DataLoader
            x_train: per-modality training input tensors (unused; kept for API parity)
            w_train: per-sample training weights
            y_train: training labels
            lengths_train: per-modality original sequence lengths
            epoch: current epoch index (0-based)
        """
        logits_shape = y_train.shape[0] if self.n_classes is None else (y_train.shape[0], self.n_classes)
        self.log_train = torch.zeros(logits_shape, device=self._device)
        self._model.train()
        for i, train_data in enumerate(dataloader):
            x, y, idx = train_data
            y = y.to(self._device)
            lengths = [l_mod[idx] if l_mod is not None else None for l_mod in lengths_train]

            logits, (cat_logits, pred_q) = self._model(x, lengths, graph_learning=epoch >= self.warmup_epochs)

            hgr_loss = self.soft_hgr_loss({j: pred_q[j] for j in range(len(pred_q))}, batch_size=y.shape[0])
            prob_log, adjs = cat_logits
            if adjs[0].shape[0] > 0:
                density_loss = ((adjs.sum())/ adjs.numel() - self.rho) ** 2
            else:  # no edges (e.g. under a 'no_interaction' ablation), skip the density penalty
                density_loss = 0

            if epoch >= self.warmup_epochs:
                loss = (1 - self.lambda_1) * self.loss(logits, y, w_train[idx]) + self.lambda_1 * (
                            0.999 * hgr_loss + 0.001 * density_loss)
            else:
                loss = 0.999 * hgr_loss + 0.001 * density_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self.log_train[idx] = logits.squeeze().detach()

    def evaluate_train_valid_epoch(self, x_train, y_train, w_train, lengths_train, g_train, x_valid, lengths_valid,
                                   g_valid, y_valid, w_valid, dataset, epoch):
        """
        Computes training and validation prediction losses and metrics for one epoch. Training
        predictions are reused from the logging done in `train_one_epoch`, since recomputing
        them would be redundant.
        Args:
            x_train: per-modality training input tensors
            y_train: training labels
            w_train: per-sample training weights
            lengths_train: unused; kept for API parity
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
            log_train = self.log_train
            log_valid, _ = self._model(x_valid,
                                       lengths_valid,
                                       idx_anchors=[i for i in range(len(x_train[0]))],
                                       graph_learning=epoch >= self.warmup_epochs,
                                       )
            log_valid = log_valid[-len(y_valid):]

        train_loss = self.loss(log_train, y_train.to(self._device), w_train).item()
        valid_loss = self.loss(log_valid, y_valid.to(self._device), w_valid).item()

        if epoch >= self.warmup_epochs:  # graph learning enabled, add the metrics
            if self.n_classes is None:
                p_train, p_valid = torch.sigmoid(log_train), torch.sigmoid(log_valid)
            else:
                p_train, p_valid = F.softmax(log_train, dim=1), F.softmax(log_valid, dim=1)

            train_scores = self.compute_metrics_per_epoch(p_train, y_train)
            valid_scores = self.compute_metrics_per_epoch(p_valid, y_valid)
        else:
            train_scores, valid_scores = {'AUROC': 0, 'AUPRC': 0}, {'AUROC': 0, 'AUPRC': 0}
            p_train = torch.rand(y_train.shape[0], device=self._device)
            p_valid = torch.rand(y_valid.shape[0], device=self._device)

        self.log_train = None

        return train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid

    def predict_proba(self,
                      x: List[Union[torch.Tensor, List[torch.Tensor]]],
                      lengths: Optional[List] = None,
                      idx_anchors: Optional[List] = None) -> Tuple:
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths; computed from `x` if None
            idx_anchors: indices of training nodes to anchor inference to
        Returns:
            probas, None: predicted class probabilities for `x` through the DIGOM fuser (the
            graph is not returned by this model, unlike the other trainers' predict_proba)
        """
        self._model.eval()
        with torch.no_grad():
            if lengths is None:
                x, lengths = self.pad_temporal_sequences(x, self._zero_padding)

            logits, _ = self._model(x,
                                    lengths,
                                    idx_anchors=idx_anchors)

            probas = torch.sigmoid(logits) if self.n_classes is None else F.softmax(logits, dim=1)

            return probas, None
