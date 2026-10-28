"""
Filename: digom.py

Author: Hakima Laribi

Description: DIGOM (Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare).
The patient-adaptive sparse graph learning module samples, for every ordered pair of modalities, a discrete
adjacency matrix between patients (Bernoulli sampling with Gumbel-Softmax and a straight-through estimator),
giving a multiplex graph of M^2 disentangled views. The multimodal multi-interaction message passing module then
propagates over these views with GIN layers, and the enriched modality representations are concatenated for
prediction.
"""
from typing import List, Literal, Optional, Union

import dgl
import torch
from dgl.nn.pytorch.conv import GINConv
from torch import nn
from torch.nn import functional as F

from src.models.abstract_fusers import PostTemporalFuser


class DIGOMFuser(PostTemporalFuser):
    """DIGOM model."""

    def __init__(self,
                 input_sizes,
                 temporal_analysis,
                 device,
                 temporal_hidden_size,
                 num_att_heads,
                 att_hidden_size,
                 gnn_hidden_size,
                 gnn_n_layers,
                 temperature,
                 tau,
                 ablation: Literal['no_interaction', 'intra_patient_cross_modal', 'cross_patient_same_modal',
                                   'cross_patient_cross_modal'] = None,
                 n_classes=None,):
        """
        Args:
            num_att_heads: number of heads of the projections (1 in the paper)
            att_hidden_size: size d of the latent projections H (w_k, f_phi) and Z (w_q, f_theta)
            gnn_hidden_size: hidden size of the message passing layers
            gnn_n_layers: number of message passing layers L
            temperature: inverse of the similarity temperature tau of the paper (scores are multiplied by it)
            tau: Gumbel-Softmax temperature tau_gs
            ablation: if set, keeps a single interaction type ('intra_patient_cross_modal',
                'cross_patient_same_modal' or 'cross_patient_cross_modal') or none ('no_interaction')
            n_classes: number of classes; None for binary classification
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, 0, device, temporal_hidden_size, n_classes)
        self.num_att_heads = num_att_heads
        self._att_hidden_size = att_hidden_size
        self._gnn_hidden_size = gnn_hidden_size
        self._gnn_n_layers = gnn_n_layers
        self.M = len(input_sizes)
        self.temperature = temperature
        self.tau = tau
        self.ablation = ablation
        self.n_con = {}

        # Modality projections
        self.w_q, self.w_k, self.w_k2 = [], [], []
        for i in range(self.M):
            fq = nn.Sequential(torch.nn.Linear(self._embedded_sizes[i], 32, bias=True),
                               torch.nn.LeakyReLU(),
                               torch.nn.Linear(32, 32, bias=True),
                               torch.nn.LeakyReLU(),
                               torch.nn.Linear(32, att_hidden_size * num_att_heads, bias=True),
                               ).to(self._device)
            fk = nn.Sequential(torch.nn.Linear(self._embedded_sizes[i], 32, bias=True),
                               torch.nn.LeakyReLU(),
                               torch.nn.Linear(32, 32, bias=True),
                               torch.nn.LeakyReLU(),
                               torch.nn.Linear(32, att_hidden_size * num_att_heads, bias=True),
                               ).to(self._device)

            self.w_q.append(fq)
            self.w_k.append(fk)

        self.w_q = torch.nn.ModuleList(self.w_q)
        self.w_k = torch.nn.ModuleList(self.w_k)

        self.pred_q = nn.Linear(att_hidden_size * num_att_heads * self.M, 1).to(self._device)

        # GNN layers
        self.gnn_convs = []
        self.gnn_bns = []
        self.gin_lins = []
        for i in range(self._gnn_n_layers):
            input_size = self._att_hidden_size if i == 0 else self._gnn_hidden_size
            self.gin_lins.append(nn.ModuleList([nn.Linear(input_size, self._gnn_hidden_size)
                                                for _ in range(self.M)]).to(self._device))
            self.gnn_convs.append(GINConv(nn.Identity(), 'sum', init_eps=-1).to(self._device))
            self.gnn_bns.append(nn.BatchNorm1d(self._gnn_hidden_size).to(self._device))
        self.gnn_convs, self.gnn_bns = torch.nn.ModuleList(self.gnn_convs), torch.nn.ModuleList(self.gnn_bns)
        self.gin_lins = torch.nn.ModuleList(self.gin_lins)

        self.fuse_views = nn.Linear(att_hidden_size + gnn_hidden_size, gnn_hidden_size).to(self._device)
        self.bn = nn.BatchNorm1d(self._gnn_hidden_size).to(self._device)

        # Joint fusion
        self.fuse_modal = nn.Linear(gnn_hidden_size * self.M, 1 if n_classes is None else n_classes).to(self._device)

    def create_cross_patients_graphs(self, x, anchor_nodes, graph_learning):
        """
        Builds, for every ordered pair of modalities (m2 -> m1), a bipartite graph of edges
        sampled via Bernoulli distribution between patients, optionally
        overridden by `self.ablation`. Only anchor nodes (not testing nodes) may send messages.
        Args:
            x: per-modality input tensors
            anchor_nodes: indices of patients allowed to propagate messages; all if None
            graph_learning: if False, skip assembling the DGL heterograph (only compute the
                edge-probability/adjacency tensors used for the training losses), used during warmup
        Returns:
            graphs, (prob_edge, adjs): the DGL heterograph of sampled edges (empty list if
            `graph_learning` is False), the per-pair edge log-probabilities (only kept during
            training), and the per-pair sampled adjacency matrices considered for the density
            regularization (only kept during training)
        """
        # Get dimensions
        bs = x[0].shape[0]  # batch size
        n_mod = len(x)  # number of modalities¸, x is already processed temporally

        # Create anchor mask, Zero the connections between each pair of non-anchor nodes, at each modality block
        anchor_nodes = anchor_nodes if anchor_nodes is not None else [i for i in range(bs)]
        mask_anchor = torch.eye(bs).to(self._device)  # always keep self-loops

        # nodes are ordered as : anchor nodes, non-anchor nodes (rows as well as columns)
        mask_anchor[:, :len(anchor_nodes)] = 1  # propagation (columns) from anchor nodes only

        # list of graphs
        graphs_dict = {}
        graphs = []

        # Create a bipartite graph for each pair of modalities
        prob_edge = []
        adjs = []
        for m1 in range(n_mod):
            # Project queries
            q = self.w_q[m1](x[m1].to(self._device)).view(bs, self.num_att_heads, self._att_hidden_size)
            q = q.permute(1, 0, 2)
            q_norm = F.normalize(q, dim=2)
            for m2 in range(n_mod):
                # Project keys
                k = self.w_k[m2](x[m2].to(self._device)).view(bs, self.num_att_heads, self._att_hidden_size)

                # Normalized dot product
                k = k.permute(1, 0, 2)  # size (n_heads, bs, att_hidden_size)
                k_norm = F.normalize(k.transpose(1, 2), dim=1)
                mat_mul = torch.matmul(q_norm, k_norm)

                # Use sigmoid instead of softmax to not have small probability value
                m = mat_mul.sum(dim=0)
                att_scores = m * self.temperature  # size (bs, bs), sum over all heads

                log_prob_edge = att_scores  # log(p)
                log_prob_no_edge = -att_scores  # log(1-p) supposedly, not the real value
                if self.training:
                    prob_edge.append(att_scores.to('cpu'))

                # Add 1-p probabilities for the category "no edge"
                cat_logits = torch.stack([log_prob_no_edge, log_prob_edge],
                                         dim=-1).to(self._device)  # size (bs, bs, 2)
                del k, log_prob_edge, log_prob_no_edge,
                torch.cuda.empty_cache()

                # Gumbel Softmax to sample edges
                if self.ablation != 'no_interaction':
                    adj_blocks = []
                    block_size = 8000
                    for i in range(0, cat_logits.size(0), block_size):
                        block = cat_logits[i:i + block_size].to(self._device)
                        # Guumbel softmax + straight through estimator
                        adj_block = F.gumbel_softmax(block, tau=self.tau, hard=True)[:, :, 1].squeeze()
                        adj_blocks.append(adj_block)

                    adj = torch.cat(adj_blocks, dim=0)
                else:
                    adj = torch.zeros_like(att_scores)

                if self.ablation == 'intra_patient_cross_modal':
                    adj = adj * torch.eye(bs).to(self._device)
                elif self.ablation == 'cross_patient_same_modal':
                    adj = adj if m1 == m2 else torch.zeros_like(adj)
                elif self.ablation == 'cross_patient_cross_modal':
                    adj = adj * (torch.ones_like(adj) - torch.eye(bs).to(
                        self._device)) if m1 != m2 else torch.zeros_like(adj)
                else:
                    adj = adj

                # Mask non-anchor nodes, cannot propagate, only receive messages
                adj = adj * mask_anchor

                # Get number of connections per patient modality
                if not self.training:
                    self.n_con[('m2', f'propagates_{m1 * n_mod + m2}', 'm1')] = (
                    adj.sum(dim=1).to('cpu'),  # number of connections of each individual
                    # Average overall connectivity
                    (torch.softmax(cat_logits, dim=-1)[:, :, 0].squeeze() * adj).mean(dim=1).to('cpu'),
                    # Average probability among selected connections
                    ((torch.softmax(cat_logits, dim=-1)[:, :, 0].squeeze() * adj).sum(dim=1) / adj.sum(dim=1)).to(
                        'cpu'))

                if self.training:
                    # for these, we don't need to control density
                    if (self.ablation != 'intra_patient_cross_modal'):  # max density 1/bs with bs >> we dont need optimization
                        if ((self.ablation != 'cross_patient_same_modal') | (m1 == m2)):
                            # for this ablation, only care about same-modal graphs for density
                            adjs.append(adj)

                # Create graph
                if graph_learning:
                    dst, src = adj.nonzero(as_tuple=True)
                    dst, src = dst.to('cpu'), src.to('cpu')
                    graphs_dict[('m2', f'propagates_{m1 * n_mod + m2}', 'm1')] = (src, dst)
                    del dst, src
                    torch.cuda.empty_cache()

                del adj
                torch.cuda.empty_cache()

        if graph_learning:
            graphs = dgl.heterograph(graphs_dict, num_nodes_dict={'m2': bs, 'm1': bs})

        if self.training:
            prob_edge = torch.stack(prob_edge, dim=0) if len(prob_edge) > 0 else torch.tensor([])
            adjs = torch.stack(adjs, dim=0) if len(adjs) > 0 else torch.tensor([])
        return graphs, (prob_edge, adjs)

    def gnn_forward(self, x, g, input_nodes=None):
        """
        Runs `self._gnn_n_layers` GIN-style layers: at each layer, every modality aggregates
        (degree-normalized sum) messages from all modalities' bipartite graphs, including itself.
        Args:
            x: per-modality, temporally-aggregated input tensors
            g: DGL heterograph with one ('m2', 'propagates_<i>', 'm1') edge type per ordered
                pair of modalities, as built by create_cross_patients_graphs
            input_nodes: unused; kept for API parity
        Returns:
            per-modality node embeddings after the last GNN layer, shape (M, bs, gnn_hidden_size)
        """
        bs = x[0].shape[0]  # batch size

        h_all = []
        qs = [self.w_q[m](x[m]).view(bs, self.num_att_heads, self._att_hidden_size).mean(1)
              for m in range(self.M)]
        ks = [self.w_q[m](x[m]).view(bs, self.num_att_heads, self._att_hidden_size).mean(1)
              for m in range(self.M)]

        # GNN layers
        for n_layer in range(self._gnn_n_layers):
            conv = self.gnn_convs[n_layer]
            bn = self.gnn_bns[n_layer]
            h_layer = []  # store embeddings for this layer
            n_mod = 0
            for m1 in range(self.M):
                h_temp_list = []
                deg_list = []
                # Get projection of modality m1
                q = qs[m1] if n_layer == 0 else h_all[-1][m1]
                for m2 in range(self.M):
                    graph = g[('m2', f'propagates_{n_mod}', 'm1')].to(self._device)  # [n_mod].to(self._device)
                    k = ks[m2] if n_layer == 0 else h_all[-1][m2]

                    # graph convolution is just a sum of neighbor features
                    h = conv.double()(graph, (k.double(), q.double())).float()
                    h_temp_list.append(h)
                    deg_list.append(graph.in_degrees(etype=('m2', f'propagates_{n_mod}', 'm1')))
                    n_mod += 1

                # sum all embeddings obtained from each bipartite graph
                h_temp = torch.stack(h_temp_list, dim=0).sum(dim=0)

                # Weight the aggregated embedding received from neighbors by the destination node degree
                deg_tmp = torch.stack(deg_list, dim=0).sum(dim=0) + 1  # +1 to avoid division by zero
                weight_agg = 1 / deg_tmp.unsqueeze(1) * h_temp  # mean aggregation

                # GIN style convolution to aggregate node embeddings
                fused = self.gin_lins[n_layer][m1](q + weight_agg)

                if n_layer != self._gnn_n_layers - 1:
                    fused = F.relu(bn(fused))
                h_layer.append(fused)
            h_all.append(h_layer)

        # Convert to tensor if needed
        h_all = torch.stack([torch.stack(layer, dim=0) for layer in h_all], dim=0)
        return h_all[-1]

    def fuse_and_predict(self, h):
        """
        Args:
            h: per-modality node embeddings, shape (n_mod, bs, gnn_hidden_size)
        Returns:
            predictions obtained by concatenating modality embeddings and applying a linear layer
        """
        h = h.permute(1, 0, 2)  # size (bs, n_mod, gnn_hidden_size)
        h = h.reshape(h.shape[0], -1)  # size (bs, n_mod * gnn_hidden_size)
        return self.fuse_modal(h).squeeze()

    def forward(self,
                x: List[Union[torch.Tensor, List[torch.Tensor]]],
                lengths: Optional[List[int]] = None,
                idx_anchors=None,
                graph_learning=True):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
            idx_anchors: indices of patients allowed to propagate messages; all if None
            graph_learning: if False, skip graph construction and the GNN, returning random
                placeholder logits (used during an HGR-only warmup phase)
        Returns:
            logits, (cat_logits, list_pred): predicted logits, the (edge log-probabilities,
            sampled adjacency matrices) pair from create_cross_patients_graphs, and the raw
            per-modality query projections (consumed as HGR-style predictions by the trainer)
        """
        bs = len(x[0])
        x_temporal = [self.transform_modality(x_modality, i, lengths) for i, x_modality in enumerate(x)]

        list_pred = [self.w_q[m](x_temporal[m]) for m in range(self.M)]
        graphs, cat_logits = self.create_cross_patients_graphs(x_temporal, idx_anchors, graph_learning)

        if graph_learning:
            h_gnn = self.gnn_forward(x_temporal, graphs)
            logits = self.fuse_and_predict(h_gnn)
        else:
            if self.n_classes is not None:
                logits = 10 * torch.rand(bs, self.n_classes, device=self._device)
            else:
                logits = 10 * torch.rand(bs, device=self._device)

        return logits, (cat_logits, list_pred)
