"""
Filename: training.py

Author: Hakima Laribi

Description: Defines the Trainer class, which runs the training loop (with early stopping),
validation, inference and plotting for the fusion models, supporting both regular tensor
data and graph-structured (DGL) data.
"""
import os
from datetime import datetime
from os import path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.preprocessing.dataset import MMDataset, TorchDataset
from src.data.preprocessing.graph_dataset import GraphDatasetMixin
from src.models.stgnn import STGNNFuser


class Trainer:
    """
    Trains a fusion model: builds it from `model_params`, runs the epoch loop with
    early stopping on the sum of validation AUROC and AUPRC, and exposes prediction and
    plotting utilities.
    """

    def __init__(self,
                 model: Callable,
                 temporal_analysis: List[bool],
                 model_params: Dict[str, object],
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
                 min_epochs: int = None,
                 load_model: str = None,
                 n_classes: int = None
                 ):
        """
        Args:
            model: fuser model class to instantiate
            temporal_analysis: per-modality flag, True if the modality has temporal sequences
            model_params: keyword arguments forwarded to `model`'s constructor
            alpha: L1 penalty weight
            beta: L2 penalty weight
            lr: learning rate
            pos_weight: weight assigned to positive samples in the loss (binary case only)
            batch_size: training batch size
            n_epochs: maximum number of training epochs
            patience: number of epochs without validation improvement before early stopping
            device: device to train on, falls back to 'cpu' if cuda is unavailable
            path_to_experiment: directory where the model checkpoint and plots are saved
            save_model: if False, the checkpoint is deleted once training ends
            verbose: if True, prints progress every 5 epochs and the final best metrics
            batch_size_inference: batch size used for graph-based inference
            min_epochs: minimum number of epochs before early stopping can trigger
            load_model: path to a checkpoint to warm-start the model from
            n_classes: number of classes; None for binary classification
        """
        device = device if torch.cuda.is_available() else 'cpu'

        self._model = model(temporal_analysis=temporal_analysis, **model_params, device=device, n_classes=n_classes)
        if load_model is not None:
            self._model.load_state_dict(torch.load(load_model, map_location=device, weights_only=True))
        self._zero_padding = False if isinstance(self._model, STGNNFuser) else True

        self._temporal_analysis = temporal_analysis
        self._graph_data = False
        self._alpha = alpha
        self._beta = beta
        self._lr = lr
        self._pos_weight = pos_weight
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._patience = patience
        self._device = device
        self.batch_size_inference = batch_size_inference
        self.n_classes = n_classes
        self.min_epochs = min_epochs

        self._path_to_experiment = path_to_experiment if save_model else '.'
        self._model_name = f'{self._model.__class__.__name__}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.pth'
        self._save_model = save_model
        self._verbose = verbose

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        if n_classes is None:
            self._pred_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self._pred_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        self._threshold = None
        self._patience_counter = 0

    @property
    def threshold(self) -> float:
        """
        Returns:
            the decision threshold used to binarize predicted probabilities
        """
        return self._threshold

    def weight_per_sample(self, y: torch.Tensor) -> torch.Tensor:
        """
        Assign a weight to each sample according to its class weight. w_sample = class_weight / nb_samples_class
        Args:
            y: binary ground-truth labels
        Returns:
            a tensor of per-sample weights, same shape as y
        """
        # Find indexes of positive and negative samples
        positive_mask = y == 1
        negative_mask = y == 0

        # Assign per-sample weight
        weights = torch.zeros_like(y, dtype=torch.float)
        weights[positive_mask] = self._pos_weight / positive_mask.sum().item()
        weights[negative_mask] = (1 - self._pos_weight) / negative_mask.sum().item()

        return weights

    def prepare_valid_data(self,
                           x_train,
                           x_valid,
                           g_valid,
                           y_valid,
                           w_valid):
        """
        Prepares validation data for one epoch. For graph data, samples the k-hop neighbors
        of the validation nodes from the combined train+valid graph so the model can message-
        pass over them; otherwise just pads the validation sequences.
        Args:
            x_train: per-modality training input tensors (already padded)
            x_valid: per-modality validation input tensors (not yet padded)
            g_valid: validation graph, or None for non-graph data
            y_valid: validation labels
            w_valid: per-sample validation weights
        Returns:
            x_valid, lengths_valid, y_valid, w_valid, g_valid, ready for one evaluation epoch
        """
        if self._graph_data:
            # Pad the temporal sequences of the training and validation nodes if needed
            x_train_valid = [x_train_mod + x_valid_mod if isinstance(x_valid_mod, list)
                             else torch.cat([x_train_mod, x_valid_mod], dim=0)
                             for x_train_mod, x_valid_mod in zip(x_train, x_valid)]
            x_train_valid, len_train_valid = self.pad_temporal_sequences(x_train_valid, self._zero_padding)

            # Use a dataloader to sample all the validation nodes at once, and only their neighbors from training nodes
            graph_sampler = MultiLayerNeighborSampler([-1] * self._model.gnn_n_layers)  # -1 means all neighbors
            valid_loader = DGLDataLoader(g_valid,
                                         g_valid.nodes()[-len(y_valid):],
                                         graph_sampler,
                                         batch_size=len(y_valid),  # all validation nodes in one batch
                                         shuffle=False,
                                         drop_last=False)
            # Perform one single iteration to get the neighbor nodes and the validation nodes
            input_nodes, output_nodes, g_valid = next(iter(valid_loader))

            # Create the tensor of validation data containing validation nodes data and their neighbors data
            x_valid = [x_modality.to(self._device)[input_nodes] for x_modality in x_train_valid]
            valid_idx = (output_nodes - len(x_train[0])).tolist()  # adjust indices to valid set
            # reorder targets and weights according to how valid data are ordered in output nodes
            y_valid = y_valid[valid_idx]
            w_valid = w_valid[valid_idx]
            lengths_valid = [l_mod[input_nodes.tolist()] if l_mod is not None else None for l_mod in len_train_valid]

        else:
            # Pad the temporal sequences of the validation set if needed
            x_valid, lengths_valid = self.pad_temporal_sequences(x_valid, self._zero_padding)

        return x_valid, lengths_valid, y_valid, w_valid, g_valid

    def prepare_valid_data_for_adaptive_learning(self,
                                                 x_train,
                                                 x_valid,
                                                 g_valid,
                                                 y_valid,
                                                 w_valid):
        """
        Concatenates and pads the training and validation sequences of a graph dataset.
        Args:
            x_train: per-modality training input tensors
            x_valid: per-modality validation input tensors
            g_valid: validation graph
            y_valid: validation labels
            w_valid: per-sample validation weights
        Returns:
            x_train_valid, len_train_valid, y_valid, w_valid, g_valid: the concatenated and
            padded train+valid inputs and lengths, alongside the unchanged remaining inputs
        """
        x_train_valid = [x_train_mod + x_valid_mod if isinstance(x_valid_mod, list)
                         else torch.cat([x_train_mod, x_valid_mod], dim=0)
                         for x_train_mod, x_valid_mod in zip(x_train, x_valid)]
        x_train_valid, len_train_valid = self.pad_temporal_sequences(x_train_valid, self._zero_padding)
        return x_train_valid, len_train_valid, y_valid, w_valid, g_valid

    def train_one_epoch(self, dataloader, x_train, w_train, y_train, lengths_train, epoch):
        """
        Runs one training epoch: forward, loss, backward and optimizer step per batch.
        Args:
            dataloader: training DataLoader (or DGLDataLoader for graph data)
            x_train: per-modality training input tensors
            w_train: per-sample training weights
            y_train: training labels
            lengths_train: per-modality original sequence lengths
            epoch: current epoch index (0-based)
        """
        self._model.train()
        for train_data in dataloader:
            if not self._graph_data:
                x, y, idx = train_data
                weights = w_train[idx].to(self._device)

            else:
                # output nodes are the nodes sampled for the current batch, input nodes are all their k-hop neighbors
                input_nodes, output_nodes, graphs = train_data  # graphs are DGL blocks with neighbors at each hop
                x = [x_modality[input_nodes] for x_modality in x_train]
                idx = input_nodes.tolist()
                y = y_train[output_nodes.tolist()]
                weights = w_train[output_nodes.tolist()].to(self._device)

            # x is moved to device inside the forward method (done separately for each modality)
            y = y.to(self._device)
            lengths = [l_mod[idx] if l_mod is not None else None for l_mod in lengths_train]

            self._optimizer.zero_grad()

            logits = self._model(x, lengths) if not self._graph_data else self._model(x, lengths, graphs)
            loss = self.loss(logits, y, weights)
            loss.backward()
            self._optimizer.step()

            # clear gradients again to free memory before the next batch
            self._optimizer.zero_grad()

            del x, y, weights, logits, loss
            torch.cuda.empty_cache()

    def evaluate_train_valid_epoch(self, x_train, y_train, w_train, lengths_train, g_train, x_valid, lengths_valid,
                                   g_valid, y_valid, w_valid, dataset, epoch):
        """
        Computes training and validation predictions, losses and metrics for one epoch.
        Args:
            x_train: per-modality training input tensors
            y_train: training labels
            w_train: per-sample training weights
            lengths_train: per-modality training sequence lengths
            g_train: training graph, or None for non-graph data
            x_valid: per-modality validation input tensors
            lengths_valid: per-modality validation sequence lengths
            g_valid: validation graph, or None for non-graph data
            y_valid: validation labels
            w_valid: per-sample validation weights
            dataset: dataset being trained on, used to locate the validation nodes in `g_valid`
            epoch: current epoch index (0-based)
        Returns:
            train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid
        """
        self._model.eval()
        with torch.no_grad():
            if self._graph_data:
                log_train = self._model(x_train, lengths_train, g_train)
                log_valid = self._model(x_valid, lengths_valid, g_valid)[-len(dataset.valid_indexes):]
            else:
                log_train = self._model(x_train, lengths_train)
                log_valid = self._model(x_valid, lengths_valid)

        if self.n_classes is None:
            p_train, p_valid = torch.sigmoid(log_train), torch.sigmoid(log_valid)
        else:
            p_train, p_valid = F.softmax(log_train, dim=1), F.softmax(log_valid, dim=1)

        train_loss = self.loss(log_train, y_train.to(self._device), w_train.to(self._device)).item()
        valid_loss = self.loss(log_valid, y_valid.to(self._device), w_valid.to(self._device)).item()
        train_scores = self.compute_metrics_per_epoch(p_train, y_train)
        valid_scores = self.compute_metrics_per_epoch(p_valid, y_valid)

        del log_train, log_valid
        torch.cuda.empty_cache()

        return train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid

    def fit(self,
            dataset: Union[MMDataset, GraphDatasetMixin]):
        """
        Trains the model on `dataset` until `n_epochs` is reached or early stopping triggers,
        then reloads the best checkpoint (see `terminate_training`).
        Args:
            dataset: training/validation dataset, graph-structured or not
        """
        self._graph_data = True if isinstance(dataset, GraphDatasetMixin) else False

        x_train, y_train = dataset[dataset.train_indexes]
        x_valid, y_valid = dataset[dataset.valid_indexes]

        w_train, w_valid = self.weight_per_sample(y_train), self.weight_per_sample(y_valid)

        g_train = dataset.train_graph.to(self._device) if self._graph_data else None
        g_valid = dataset.valid_graph.to(self._device) if self._graph_data else None

        valid_data = self.prepare_valid_data(x_train, x_valid, g_valid, y_valid, w_valid)
        x_valid, lengths_valid, y_valid, w_valid, g_valid = valid_data

        x_train, lengths_train = self.pad_temporal_sequences(x_train, self._zero_padding)
        x_train = [x_modality.to(self._device) for x_modality in x_train]
        x_valid = [x_modality for x_modality in x_valid]

        dataloader = self.create_train_dataloader(x_train, y_train, dataset, g_train)

        best_val_metric = {'AUROC': 0, 'AUPRC': 0}
        best_train_metric = {'AUROC': 0, 'AUPRC': 0}

        self._patience_counter = 0

        train_losses, valid_losses = [], []
        train_metrics, valid_metrics = [], []

        for epoch in range(self._n_epochs):
            self.train_one_epoch(dataloader, x_train, w_train, y_train, lengths_train, epoch)

            scores = self.evaluate_train_valid_epoch(x_train, y_train, w_train, lengths_train, g_train,
                                                     x_valid, lengths_valid, g_valid, y_valid, w_valid,
                                                     dataset, epoch)
            train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid = scores

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_metrics.append(train_scores)
            valid_metrics.append(valid_scores)

            # print progress every 5 epochs
            if (((epoch + 1) % 5 == 0) or ((epoch + 1) == self._n_epochs)) & self._verbose:
                self.print_progress(epoch, train_loss, train_scores, valid_loss, valid_scores)

            # update best validation metric if improved, else count toward early stopping
            best_val_metric, best_train_metric, early_stop = self.early_stop(best_val_metric,
                                                                             valid_scores,
                                                                             best_train_metric,
                                                                             train_scores,
                                                                             epoch + 1)

            if early_stop | (epoch == self._n_epochs - 1):
                if self._verbose:
                    if early_stop:
                        print(f"Early Stopping occurred at epoch {epoch + 1}. ")

                    print(f"Best validation metrics: AUROC: {best_val_metric['AUROC']:.4f}, "
                          f"AUPRC: {best_val_metric['AUPRC']:.4f}"
                          f" with best training metrics: AUROC: {best_train_metric['AUROC']:.4f}, "
                          f"AUPRC: {best_train_metric['AUPRC']:.4f}")
                self.terminate_training(train_losses, valid_losses, train_metrics, valid_metrics)
                break

        del x_train, g_train, y_train, p_train, x_valid, g_valid, y_valid, p_valid, w_train, w_valid, dataloader
        torch.cuda.empty_cache()

    def terminate_training(self, train_losses, valid_losses, train_metrics, valid_metrics):
        """
        Reloads the best checkpoint, discarding it if `save_model` is False, else plots curves.
        Args:
            train_losses: per-epoch training losses
            valid_losses: per-epoch validation losses
            train_metrics: per-epoch training AUROC/AUPRC dicts
            valid_metrics: per-epoch validation AUROC/AUPRC dicts
        """
        self._model.load_state_dict(
            torch.load(path.join(self._path_to_experiment, self._model_name), weights_only=True))

        if not self._save_model:
            # delete the checkpoint (generally during hyperparameter tuning)
            os.remove(path.join(self._path_to_experiment, self._model_name))

        else:
            self.plot_training_curves(train_losses=train_losses,
                                      valid_losses=valid_losses,
                                      train_metrics=train_metrics,
                                      valid_metrics=valid_metrics)

    def loss(self, pred, y, sample_weights, model=None):
        """
        Weighted prediction loss plus an elastic-net (L1/L2) penalty on `model`'s weights.
        Args:
            pred: predicted logits
            y: ground-truth labels
            sample_weights: per-sample weights, or None for equal weighting
            model: model whose weights are penalized; defaults to self._model
        Returns:
            the scalar weighted loss, including the elastic-net penalty
        """
        model = model if model is not None else self._model
        pred = pred.squeeze()

        l1_penalty, l2_penalty = torch.tensor(0., device=self._device), torch.tensor(0., device=self._device)
        if (self._alpha > 0) | (self._beta > 0):
            for _, w in model.named_parameters():
                l1_penalty = l1_penalty + w.abs().sum().to(self._device)
                l2_penalty = l2_penalty + w.pow(2).sum().to(self._device)

        y = y.float() if self.n_classes is None else y
        loss = self._pred_loss_fn(pred.to(self._device), y.to(self._device))  # (N,) tensor, no reduction

        if isinstance(self._alpha, float):
            self._alpha = torch.tensor(self._alpha).to(self._device)
            self._beta = torch.tensor(self._beta).to(self._device)

        if (sample_weights is None) | (self.n_classes is not None):
            sample_weights = torch.full_like(pred[:, 0], 1 / pred.shape[0])  # equal weight per sample

        sample_weights = sample_weights.to(self._device)

        p_loss = (loss * sample_weights).sum() + self._alpha * l1_penalty + self._beta * l2_penalty

        return p_loss

    def soft_hgr_loss(self, outputs_batch, batch_size):
        """
        Soft HGR correlation loss, summed pairwise over modality projections in `outputs_batch`.
        Shared by trainers (MaxCorrMGNNTrainer, DIGOMTrainer) whose models align modality projections
        via a Hirschfeld-Gebelein-Renyi (HGR) style loss.

        Adapted to our framework from : https://github.com/IBM/MMMG/blob/main/Graph_Classification/utils/HGR_modules.py

        Args:
            outputs_batch: dict mapping modality name to its batch of HGR projections
            batch_size: number of samples in the batch, used to normalize the expectation term

        Returns the negative of the summed pairwise soft HGR correlations (to minimize), averaged
        over the M * (M - 1) modality pairs.
        """
        covs_trace = 0  # trace term for covariance
        exp_term = 0  # expectation term

        M = len(outputs_batch)

        for k, v in enumerate(outputs_batch):
            outputs_batch[v] = outputs_batch[v].to(self._device)
            f_m = torch.mean(outputs_batch[v], dim=0).reshape(1, -1)
            N = outputs_batch[v].shape[1]

            f_ms = outputs_batch[v] - f_m.expand_as(outputs_batch[v])
            f_cov = torch.cov(f_ms.transpose(0, 1)) + 0.001 * torch.eye(N).to(f_ms.device)  # reg term for stability

            for k1, v1 in enumerate(outputs_batch):
                outputs_batch[v1] = outputs_batch[v1].to(self._device)
                if not (k1 <= k):  # avoid double counting each pair
                    g_m = torch.mean(outputs_batch[v1], dim=0).reshape(1, -1)
                    g_ms = outputs_batch[v1] - g_m.expand_as(outputs_batch[v1])
                    g_cov = torch.cov(g_ms.transpose(0, 1)) + 0.001 * torch.eye(N).to(g_ms.device)

                    covs_trace += 0.5 * torch.trace(f_cov.mm(g_cov))

                    # diag + sum is equivalent to trace here but much faster
                    exp_norm_term = torch.diag((f_ms).mm(g_ms.transpose(0, 1)))
                    exp_term += torch.sum(exp_norm_term) * (1 / (batch_size - 1))

        del outputs_batch
        torch.cuda.empty_cache()

        hgr = ((-exp_term + covs_trace) / (M * (M - 1)))
        return hgr

    def early_stop(self, best_val_metric, valid_score, best_train_score, train_score, epoch):
        """
        Early stopping mechanism that checks if the validation metrics did not improve for a certain number of epochs
        Args:
            best_val_metric: best validation AUROC/AUPRC seen so far
            valid_score: current epoch's validation AUROC/AUPRC
            best_train_score: training AUROC/AUPRC at the best validation epoch so far
            train_score: current epoch's training AUROC/AUPRC
            epoch: current epoch number (1-based)
        Returns:
            best_val_metric, best_train_score, early_stop: updated best scores and whether to
            stop training now
        """
        # we want a good AUPRC and AUROC, so we sum them into a single metric
        if valid_score['AUROC'] + valid_score['AUPRC'] >= best_val_metric['AUROC'] + best_val_metric['AUPRC']:
            best_val_metric = valid_score
            best_train_score = train_score
            torch.save(self._model.state_dict(), path.join(self._path_to_experiment, self._model_name))
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        if self._patience_counter >= self._patience:
            if self.min_epochs is not None:
                if self.min_epochs <= epoch:
                    return best_val_metric, best_train_score, True
            else:
                return best_val_metric, best_train_score, True

        return best_val_metric, best_train_score, False

    def predict_proba(self,
                      x: List[Union[torch.Tensor, List[torch.Tensor]]],
                      g: Optional[dgl.DGLGraph] = None,
                      lengths: Optional[List] = None) -> Tuple[Tensor, dgl.DGLGraph]:
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            g: graph to message-pass over, or None for non-graph data
            lengths: per-modality original sequence lengths; computed from `x` if None
        Returns:
            probas, g: predicted class probabilities for `x`, and the graph used (if any)
        """
        self._model.eval()
        with torch.no_grad():
            if lengths is None:
                x, lengths = self.pad_temporal_sequences(x, self._zero_padding)
            if g is None:
                logits = self._model(x, lengths)
            else:
                logits = self._model(x, lengths, g)

            del x, lengths
            probas = torch.sigmoid(logits) if self.n_classes is None else F.softmax(logits, dim=1)
            return probas, g

    def predict_proba_with_graphs(self,
                                  nodes_ids: Tensor):
        """
        Returns a `predict` closure that runs inference over `nodes_ids` through a DGL batch
        loader, since inference always uses the training graph (too large to fit in one batch).
        Args:
            nodes_ids: ids of the nodes to predict on
        Returns:
            a `predict(x, g, lengths=None)` closure returning (probas, g) for `nodes_ids`
        """
        graph_sampler = MultiLayerNeighborSampler([-1] * self._model.gnn_n_layers)  # -1 means all neighbors

        def predict(x: List[Union[torch.Tensor, List[torch.Tensor]]], g: dgl.DGLGraph,
                    lengths: Optional[List] = None) -> Tuple[Tensor, dgl.DGLGraph]:
            """
            Args:
                x: per-modality input tensors for the sampled nodes
                g: graph (or DGL blocks) to message-pass over
                lengths: per-modality original sequence lengths
            Returns:
                probas, g: predicted class probabilities for `nodes_ids`, and the graph used
            """
            dataloader = DGLDataLoader(g.to(self._device),
                                       nodes_ids.to(self._device),
                                       graph_sampler,
                                       batch_size=self.batch_size_inference,
                                       shuffle=False,
                                       drop_last=False)

            # Keep positions of nodes_ids
            id_to_index = {nid.item(): i for i, nid in enumerate(nodes_ids)}
            if self.n_classes is None:
                probas = torch.empty(len(nodes_ids), device=self._device)
            else:
                probas = torch.empty((len(nodes_ids), self.n_classes), device=self._device)
            for batch in dataloader:
                input_nodes, output_nodes, graphs = batch
                x_batch = [[xi[i] for i in input_nodes.tolist()] for xi in x]
                batch_probas, g = self.predict_proba(x_batch, graphs, lengths)

                # Reorder probabilities to match the original node ids
                indices = [id_to_index[nid] for nid in output_nodes.tolist()]
                probas[indices] = batch_probas

            return probas, g

        return predict

    def create_train_dataloader(self, x_train, y_train, dataset, g_train):
        """
        Builds the training DataLoader, a graph-sampling DGLDataLoader if `g_train` is set.
        Args:
            x_train: per-modality training input tensors
            y_train: training labels
            dataset: training dataset, used for its ids and temporal_analysis flag
            g_train: training graph, or None for non-graph data
        Returns:
            a DataLoader (or DGLDataLoader) over the training data
        """
        torch_dataset = TorchDataset(x=x_train,
                                     y=y_train,
                                     ids=dataset.train_ids,
                                     temporal_analysis=(sum(self._temporal_analysis) > 0))

        if not self._graph_data:
            dataloader = DataLoader(dataset=torch_dataset,
                                    shuffle=True,
                                    batch_size=self._batch_size,
                                    collate_fn=self.collate_fn if dataset.temporal_analysis else None)
        else:
            graph_sampler = MultiLayerNeighborSampler([-1] * self._model.gnn_n_layers)  # -1 means all neighbors
            dataloader = DGLDataLoader(g_train,
                                       g_train.nodes()[[i for i in range(len(x_train[0]))]],
                                       graph_sampler,
                                       batch_size=self._batch_size,
                                       shuffle=True,
                                       drop_last=False)

        return dataloader

    @staticmethod
    def pad_with_last_value(x: List[torch.Tensor]) -> torch.Tensor:
        """
        Pads a list of tensors with the last value of each tensor.
        Args:
            x: list of variable-length tensors to pad to the same length
        Returns:
            a single tensor stacking the padded sequences
        """
        if not x:
            return torch.tensor([])

        max_len = max([s.size(0) for s in x])
        padded_sequences = []

        for seq in x:
            current_len = seq.size(0)
            padding_needed = max_len - current_len

            if padding_needed > 0:
                last_value = seq[-1].unsqueeze(0)
                padding = last_value.repeat(padding_needed, *([1] * (len(seq.shape) - 1)))
                padded_seq = torch.cat([seq, padding], dim=0)
                padded_sequences.append(padded_seq)
            else:
                padded_sequences.append(seq)

        return torch.stack(padded_sequences, dim=0)

    def pad_temporal_sequences(self,
                               x: List[torch.Tensor],
                               zero_padding: bool = True) -> tuple[list[Tensor], list[Tensor | None]]:
        """
        Pads the temporal sequences in x according to the maximum sequence length
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            zero_padding: if True, zero-pad sequences; if False, pad with each sequence's last value
        Returns:
            padded sequence
            original lengths of the sequences
        """
        lengths = []
        for i, x_modality in enumerate(x):
            if self._temporal_analysis[i]:
                lengths.append(torch.tensor([seq.size(0) for seq in x_modality], device='cpu'))
                if zero_padding:
                    x[i] = pad_sequence(x_modality, batch_first=True)
                else:
                    x[i] = self.pad_with_last_value(x_modality)
            else:
                lengths.append(None)
        return x, lengths

    def compute_metrics_per_epoch(self, proba, y):
        """
        Computes AUROC and AUPRC, binarizing labels first in the multiclass (one-vs-rest) case.
        Args:
            proba: predicted class probabilities
            y: ground-truth labels
        Returns:
            a dict with keys 'AUROC' and 'AUPRC'
        """
        if self.n_classes is None:
            auroc = roc_auc_score(y.cpu().numpy(), proba.cpu().numpy())
            auprc = average_precision_score(y.cpu().numpy(), proba.cpu().numpy())
        else:
            y_bin = label_binarize(y.cpu().numpy(), classes=np.arange(self.n_classes))
            auroc = roc_auc_score(y_bin, proba.cpu().numpy(), multi_class='ovr', average='weighted')
            auprc = average_precision_score(y_bin, proba.cpu().numpy(), average='weighted')
        return {'AUROC': auroc, 'AUPRC': auprc}

    @staticmethod
    def print_progress(epoch, train_loss, train_scores, valid_loss, valid_scores):
        """
        Prints training and validation loss/AUROC/AUPRC for the current epoch.
        Args:
            epoch: current epoch index (0-based)
            train_loss: current epoch's training loss
            train_scores: current epoch's training AUROC/AUPRC dict
            valid_loss: current epoch's validation loss
            valid_scores: current epoch's validation AUROC/AUPRC dict
        """
        print(f"Epoch {epoch + 1} ====== "
              f"Train Loss: {train_loss:.4f} | AUROC: {train_scores['AUROC']:.4f} | AUPRC: {train_scores['AUPRC']:.4f}"
              f" ===== "
              f"Valid Loss: {valid_loss:.4f} | AUROC: {valid_scores['AUROC']:.4f} | AUPRC: {valid_scores['AUPRC']:.4f}")

    @staticmethod
    def collate_fn(data):
        """
        Reshapes a batch into x = [x1, ..., xn], one list of tensors per modality.
        Args:
            data: list of (x, y, idx) samples, each x itself a list of per-modality tensors
        Returns:
            x_cohort, y, idx: x_cohort is [x1, ..., xn] with xi the list of tensors for
            modality i across the batch; y is stacked into a single tensor
        """
        x, y, idx = zip(*data)
        y = torch.stack(y)
        x_cohort = [list(modality) for modality in zip(*x)]
        return x_cohort, y, list(idx)

    @staticmethod
    def find_optimal_threshold(p_train, y_train) -> float:
        """
        Finds the optimal threshold from the ROC curve that separates the negative and positive
        classes by optimizing Youden's J statistic: J = TruePositiveRate - FalsePositiveRate.

        Args:
            p_train: predicted probabilities to belong to the positive class
            y_train: ground truth labels
        Returns:
            a float representing the optimal threshold
        """
        y_train = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        p_train = p_train.cpu().numpy() if isinstance(p_train, torch.Tensor) else p_train
        fpr, tpr, thresholds = roc_curve(y_train, p_train, pos_label=1)

        J = tpr - fpr
        threshold = thresholds[np.argmax(J)]

        return threshold

    def plot_training_curves(self, train_losses, valid_losses, train_metrics, valid_metrics):
        """
        Plots and saves training/validation loss, AUROC and AUPRC curves as one PNG.
        Args:
            train_losses: per-epoch training losses
            valid_losses: per-epoch validation losses
            train_metrics: per-epoch training AUROC/AUPRC dicts
            valid_metrics: per-epoch validation AUROC/AUPRC dicts
        """
        plt.style.use('ggplot')

        epochs = list(range(1, len(train_losses) + 1))
        train_aurocs = [m["AUROC"] for m in train_metrics]
        valid_aurocs = [m["AUROC"] for m in valid_metrics]
        train_auprcs = [m["AUPRC"] for m in train_metrics]
        valid_auprcs = [m["AUPRC"] for m in valid_metrics]

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, marker='o', label='Train Loss')
        plt.plot(epochs, valid_losses, marker='o', label='Valid Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_aurocs, marker='o', label='Train AUROC')
        plt.plot(epochs, valid_aurocs, marker='o', label='Valid AUROC')
        plt.title('AUROC')
        plt.xlabel('Epoch')
        plt.ylabel('AUROC')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_auprcs, marker='o', label='Train AUPRC')
        plt.plot(epochs, valid_auprcs, marker='o', label='Valid AUPRC')
        plt.title('AUPRC')
        plt.xlabel('Epoch')
        plt.ylabel('AUPRC')
        plt.legend()

        plt.tight_layout()
        plt.savefig(path.join(self._path_to_experiment, "training_curves.png"))
        plt.close()
