"""
Filename: mmgl_training.py

Author: Hakima Laribi

Description: Defines MMGLTrainer, a Trainer subclass for the MMGL fuser. Training is split
each epoch into two phases: the Modality-Aware Representation Learning (MARL) module and the
graph learner are optimized first, then the graph is rebuilt from their updated representations
and the GNN module is optimized on it.

Code adapted to our framework from : https://github.com/SsGood/MMGL
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
from torch import Tensor
from torch.nn import functional as F

from src.trainers.training import Trainer
from src.utils.utils import normalize_adj


class MMGLTrainer(Trainer):
    """
    Multi-Modal Graph Learning model class.

    We tried to replicate the code in : https://github.com/SsGood/MMGL/tree/main while respecting the methodology
    provided in the paper: 10.1109/TMI.2022.3159264
    """

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
                 theta_smooth: float = 0.001,
                 theta_degree: float = 0.001,
                 theta_sparsity: float = 0.001,
                 load_model: str = None,
                 n_classes: int = None,
                 ):
        """
        Args:
            theta_smooth: weight of the graph smoothness penalty
            theta_degree: weight of the degree regularization penalty
            theta_sparsity: weight of the graph sparsity penalty
            See Trainer.__init__ for the remaining args.
        """
        super().__init__(model, temporal_analysis, model_params, alpha, beta, lr, pos_weight, batch_size, n_epochs,
                         patience, device, path_to_experiment, save_model, verbose, load_model=load_model, n_classes=n_classes)
        self.theta_smooth = theta_smooth
        self.theta_degree = theta_degree
        self.theta_sparsity = theta_sparsity
        self.prepare_valid_data = self.prepare_valid_data_for_adaptive_learning

        # the elastic-net penalty is disabled here; the graph construction loss is added separately
        self._alpha, self._beta = 0, 0

        self._marl_optimizer = torch.optim.Adam(self._model.marl.parameters(), lr=lr)
        self.gc_optimizer = torch.optim.Adam(self._model.graph_learner.parameters(), lr=lr)
        self._gnn_optimizer = torch.optim.Adam(self._model.gnn.parameters(), lr=lr)

    def graph_construction_loss(self, feat, adj):
        """
        Graph regularization loss (smoothness, degree and sparsity penalties) encouraging the
        learned adjacency `adj` to be well-behaved given node features `feat`.
        Args:
            feat: node features used to compute the smoothness penalty
            adj: learned (normalized) adjacency matrix
        Returns:
            the scalar graph construction loss
        """
        # Graph Laplacian L = D - A where D is the degree matrix, which is diagonal, and A is the adjacency matrix
        device = self._device
        feat = feat.to(device)
        L = torch.diagflat(torch.sum(adj, -1)) - adj
        vec_one = torch.ones(adj.size(-1), requires_grad=False).to(device)

        smoothess_penalty = torch.trace(torch.mm(feat.T, torch.mm(L, feat))) / int(np.prod(adj.shape))
        degree_penalty = torch.mm(vec_one.unsqueeze(0),
                                  torch.log(torch.mm(adj, vec_one.unsqueeze(-1)) + 1e-5)).squeeze() / adj.shape[-1]
        sparsity_penalty = torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))

        graph_loss = self.theta_smooth * smoothess_penalty - self.theta_degree * degree_penalty + self.theta_sparsity * sparsity_penalty

        return graph_loss.to(self._device)

    def train_one_epoch(self, dataloader, x_train, w_train, y_train, lengths_train, epoch):
        """
        Runs one training epoch in two phases: optimizes the MARL module and graph learner,
        rebuilds the graph from their updated representations, then optimizes the GNN on it.
        Args:
            dataloader: training DataLoader
            x_train: per-modality training input tensors (unused; kept for API parity)
            w_train: per-sample training weights
            y_train: training labels
            lengths_train: per-modality original sequence lengths
            epoch: current epoch index (0-based)
        """
        self._model.marl.train()
        self._model.graph_learner.train()

        # 1- train the MARL module and the adaptive graph learner for one epoch (all batches)
        logits_all, adjacency, h_all, idx_reordered = self.marl_agl_one_epoch(dataloader, lengths_train)
        graph_loss = self.graph_construction_loss(h_all, adjacency)
        pred_loss = self.loss(logits_all, y_train[idx_reordered], w_train[idx_reordered], self._model.marl)
        total_loss = pred_loss + graph_loss

        self._marl_optimizer.zero_grad()
        self.gc_optimizer.zero_grad()
        total_loss.backward()
        self._marl_optimizer.step()
        self.gc_optimizer.step()

        # redo a forward pass to get hidden representations after this optimization step
        logits_all, adjacency, h_all, idx_reordered = self.marl_agl_one_epoch(dataloader, lengths_train)

        # rebuild the DGL graph from the (binary) adjacency matrix, edge weights are not used
        dst, src = adjacency.nonzero(as_tuple=True)  # indices of nodes with edges (non-zero entries in adj)
        src, dst = src.detach(), dst.detach()
        g_train = dgl.graph((src, dst), num_nodes=adjacency.shape[0]).to(self._device)
        g_train = dgl.remove_self_loop(g_train)
        g_train = dgl.add_self_loop(g_train)

        # 2- train the GNN module for one epoch (all batches)
        self._model.gnn.train()
        logits, idx_reordered = self.gnn_one_epoch(g_train, h_all, idx_reordered)

        # graph loss uses the MARL hidden representations (after update); prediction loss uses the GNN output
        graph_loss = self.graph_construction_loss(h_all, adjacency)
        pred_loss = self.loss(logits, y_train[idx_reordered], w_train[idx_reordered], self._model.gnn)
        total_loss = pred_loss + graph_loss

        self._marl_optimizer.zero_grad()
        self.gc_optimizer.zero_grad()
        self._gnn_optimizer.zero_grad()
        total_loss.backward()
        self._marl_optimizer.step()
        self.gc_optimizer.step()
        self._gnn_optimizer.step()


    def marl_agl_one_epoch(self, dataloader, lengths_train):
        """
        Runs the MARL module over all batches to collect hidden representations and logits,
        then derives a normalized adjacency matrix from the graph learner on those representations.
        Args:
            dataloader: training DataLoader
            lengths_train: per-modality original sequence lengths
        Returns:
            logits_all, adjacency, h_all, idx_train: concatenated MARL logits and hidden
            representations, the normalized adjacency matrix built from them, and the node
            order (idx_train) matching logits_all/h_all's rows
        """
        h_train = []
        logits_train = []
        idx_train = []
        for train_data in dataloader:
            x, y, idx = train_data
            lengths = [l_mod[idx] if l_mod is not None else None for l_mod in lengths_train]
            logits, h_mlpsh_sp = self._model.marl(x, lengths)

            h_train.append(h_mlpsh_sp)
            logits_train.append(logits)
            idx_train += idx  # keep track of the order of nodes

        h_all = torch.cat(h_train, dim=0)
        logits_all = torch.cat(logits_train, dim=0)

        adjacency = self._model.graph_learner.compute_adjacency_matrix(h_all)
        # normalize and add self-loops; needed for the graph construction loss, not for the DGL graph
        adjacency = normalize_adj(adjacency + torch.eye(adjacency.size(0)).to(adjacency.device))

        return logits_all, adjacency, h_all, idx_train

    def gnn_one_epoch(self, g_train, h_all, idx_reordered):
        """
        Runs the GNN module over `g_train` in batches, sampling all neighbors at each hop.
        Args:
            g_train: graph built from the MARL/graph-learner adjacency matrix
            h_all: node hidden representations from the MARL module, indexed by `idx_reordered`
            idx_reordered: node order matching the rows of `h_all`
        Returns:
            logits, idx_all: concatenated GNN logits and the matching node order
        """
        graph_sampler = MultiLayerNeighborSampler([-1] * 2)  # -1 means all neighbors, 2 layers are used in the GNN
        dataloader = DGLDataLoader(g_train,
                                   g_train.nodes()[[i for i in range(h_all.size(0))]],
                                   graph_sampler,
                                   batch_size=self._batch_size,
                                   shuffle=True,
                                   drop_last=False)
        logits_all = []
        idx_all = []
        idx_reordered = torch.tensor(idx_reordered).squeeze()
        for train_data in dataloader:
            # output nodes are the nodes sampled for the current batch, input nodes are all their k-hop neighbors
            input_nodes, output_nodes, graphs = train_data  # graphs are DGL blocks with neighbors at each hop

            x = h_all[input_nodes]
            idx = idx_reordered[output_nodes.tolist()]
            idx_all += idx.tolist()

            logits = self._model.gnn(graphs, x)
            logits_all.append(logits)

        return torch.cat(logits_all, dim=0), idx_all

    def evaluate_train_valid_epoch(self, x_train, y_train, w_train, lengths_train, g_train, x_valid, lengths_valid,
                                   g_valid, y_valid, w_valid, dataset, epoch):
        """
        Computes training and validation predictions, losses (MARL + GNN) and metrics for one epoch.
        Args:
            x_train: per-modality training input tensors
            y_train: training labels
            w_train: per-sample training weights
            lengths_train: per-modality training sequence lengths
            g_train: unused (graph is rebuilt internally by the model); kept for API parity
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
            log_train, adj_train, g_train, h_sh_sp_train = self._model(x_train, lengths_train)

            log_valid, adj_valid, g_valid, h_sh_sp_valid = self._model(x_valid,
                                                                       lengths_valid,
                                                                       idx_anchors=[i for i in range(len(x_train[0]))],
                                                                       graph_loader_batch_size=self.batch_size_inference)
            log_valid = log_valid[-len(w_valid):]

        if self.n_classes is None:
            p_train, p_valid = torch.sigmoid(log_train), torch.sigmoid(log_valid)
        else:
            p_train, p_valid = F.softmax(log_train, dim=1), F.softmax(log_valid, dim=1)

        train_loss = self.loss(log_train, y_train.to(self._device), w_train, self._model.marl).item()
        train_loss += self.loss(log_train, y_train.to(self._device), w_train, self._model.gnn).item()

        valid_loss = self.loss(log_valid, y_valid.to(self._device), w_valid, self._model.marl).item()
        valid_loss += self.loss(log_valid, y_valid.to(self._device), w_valid, self._model.gnn).item()

        train_scores = self.compute_metrics_per_epoch(p_train, y_train)
        valid_scores = self.compute_metrics_per_epoch(p_valid, y_valid)

        return train_loss, valid_loss, train_scores, valid_scores, p_train, p_valid

    def predict_proba(self,
                      x: List[Union[torch.Tensor, List[torch.Tensor]]],
                      lengths: Optional[List] = None,
                      idx_anchors: Optional[List] = None,
                      g_anchor: Optional[List] = None) -> Tuple[Tensor, dgl.DGLGraph]:
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths; computed from `x` if None
            idx_anchors: indices of training nodes to anchor inference to
            g_anchor: anchor graph to reuse, forwarded to the model; built internally if None
        Returns:
            probas, g: predicted class probabilities for `x` through the MARL + graph learner
            + GNN pipeline, and the graph built for them
        """
        self._model.eval()
        with torch.no_grad():
            if lengths is None:
                x, lengths = self.pad_temporal_sequences(x, self._zero_padding)

            logits, adj, g, h = self._model(x, lengths, idx_anchors, g_anchor, self.batch_size_inference)
            probas = torch.sigmoid(logits) if self.n_classes is None else F.softmax(logits, dim=1)

            return probas, g