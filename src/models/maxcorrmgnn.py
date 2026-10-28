"""
Filename: maxcorrmgnn.py

Author: Hakima Laribi

Description: MaxCorrMGNN baseline (Maximal Correlation Multi-layered Graph Neural Network). Modalities are
projected with soft-HGR maximal correlation, then a multiplex graph is learned with intra-planar (cross-patient
same-modal) and inter-planar (cross-modal) connections, optionally sparsified with top-k (k).

Code adapted to our framework from: https://github.com/IBM/MMMG
"""
import collections
import gc
from typing import List, Optional, Union

import dgl
import dgl.function as fn
import torch
from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
from dgl.nn.pytorch.conv import GINConv
from filelock import FileLock
from torch import nn
from torch.nn import functional as F

from src.models.abstract_fusers import PostTemporalFuser


class MaxCorrMGNNFuser(PostTemporalFuser):
    """MaxCorrMGNN model. See module docstring for details."""

    def __init__(self,
                 input_sizes,
                 temporal_analysis,
                 device,
                 temporal_hidden_size,
                 mlp_proj_size,
                 gnn_hidden_size,
                 k=0.05,
                 n_classes=None):
        """
        Args:
            mlp_proj_size: output size of the per-modality MLP projection (HGR latent space)
            gnn_hidden_size: hidden size of the multiplex GNN layers
            k: fraction of top-k correlated node pairs kept per block when building the graph
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, 0, device, temporal_hidden_size, n_classes)

        # Set attributes
        self.mlp_proj_size = mlp_proj_size  # Dimension of the MLP latent space (last layer)
        self.num_classes = 1 if self.n_classes is None else self.n_classes  # binary classification
        self.M = len(input_sizes)
        self.gnn_hidden_size = gnn_hidden_size  # hidden size of the GNN
        self.k = k

        # modality wise MLPs for modality feature projection
        self.lin_proj1 = [nn.Linear(embed_size, 32, bias=True) for embed_size in self._embedded_sizes]
        self.lin_proj2 = [nn.Linear(32, 32, bias=True) for _ in enumerate(input_sizes)]
        self.lin_proj3 = [nn.Linear(32, self.mlp_proj_size, bias=True) for _ in enumerate(input_sizes)]
        self.linears1 = nn.ModuleList(self.lin_proj1).to(self._device)
        self.linears2 = nn.ModuleList(self.lin_proj2).to(self._device)
        self.linears3 = nn.ModuleList(self.lin_proj3).to(self._device)
        self.bn1 = [nn.BatchNorm1d(self.mlp_proj_size).to(self._device) for _ in enumerate(input_sizes)]

        # Init sparsity matrix and weight coefficient
        self.sparse_m, self.sparse, self.c = None, None, None

        # Init multiplex GNN layers
        self.convCA1, self.linCA1, self.convAC1, self.linAC1 = None, None, None, None
        self.convCA2, self.linCA2, self.convAC2, self.linAC2 = None, None, None, None
        self.lin_readout, self.bn_gnn2, self.bn_gnn1 = None, None, None

        self.cross_modal = None
        self.A_edges, self.C_edges = None, None

    def hgr_forward(self, x, lengths):
        """
        Temporally aggregates each modality (if needed), then projects it with a 3-layer MLP
        into the shared HGR latent space of size `mlp_proj_size`.
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            proj_feat, features: dict of per-modality projections, and the same projections
            stacked into a single (N*M, mlp_proj_size) tensor
        """
        x_temporal = [self.transform_modality(x_modality, i, lengths) for i, x_modality in enumerate(x)]

        # Project with a 3-layers MLP in a latent of size in_feat
        proj_feat = collections.defaultdict(list)
        features = torch.zeros(1, self.mlp_proj_size)
        for i, x_modality in enumerate(x_temporal):
            # projects all features to common space using individual projections
            h = self.lin_proj1[i](x_modality)
            h = F.leaky_relu(h)
            h = self.lin_proj2[i](h)
            h = F.leaky_relu(h)
            h = self.lin_proj3[i](h).to('cpu')

            proj_feat[i] = h  # dictionary
            features = torch.cat([features, h], dim=0)  # tensor, shape : N*M x in_feat
        del x_temporal, x, h
        torch.cuda.empty_cache()
        return proj_feat, features[1:, :]

    def max_corr_sparsify(self):
        """
        (Re)initializes the modality-combination weights `self.c` and the per-pair sparsity
        thresholds `self.sparse` used by `create_multigraph`.
        """
        # matrix of weight of shape 1xM, used to combine each modality-level predictions
        self.c = torch.nn.Parameter(torch.Tensor(1, self.M)).to(self._device)  # across node copies
        torch.nn.init.xavier_uniform_(self.c)

        # matrix S of threshold for each pair of modalities, should be M*(M+1)/2 because S is symmetric
        self.sparse = (torch.nn.Parameter(torch.Tensor(self.M * (self.M - 1), 1))).to(self._device)
        torch.nn.init.xavier_uniform_(self.sparse)

    def gnn_predictor(self):
        """Builds the two-hop multiplex GNN (GINConv) layers and the final classifier."""
        self.linAC1 = nn.Linear(self.mlp_proj_size, self.gnn_hidden_size).to(self._device)
        self.convAC1 = GINConv(self.linAC1, 'mean').to(self._device)

        self.linCA1 = nn.Linear(self.mlp_proj_size, self.gnn_hidden_size).to(self._device)
        self.convCA1 = GINConv(self.linCA1, 'mean').to(self._device)

        self.linAC2 = nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size).to(self._device)
        self.convAC2 = GINConv(self.linAC2, 'mean').to(self._device)

        self.linCA2 = nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size).to(self._device)
        self.convCA2 = GINConv(self.linCA2, 'mean').to(self._device)

        # batch norm for GNN
        self.bn_gnn1 = nn.BatchNorm1d(self.gnn_hidden_size).to(self._device)
        self.bn_gnn2 = nn.BatchNorm1d(2 * self.gnn_hidden_size).to(self._device)

        # for classifier
        self.lin_readout = nn.Linear(2 * self.gnn_hidden_size, self.num_classes, bias=False).to(self._device)

    def gnn_forward(self, h_g1, h_g2, g1, g2):
        """
        Runs the two-hop multiplex GNN: one GINConv stack over g1 (A->C transitions) and one
        over g2 (C->A transitions), then combines and classifies their outputs.
        Args:
            h_g1: node features for g1
            h_g2: node features for g2
            g1: graph (or list of 2 per-layer DGL blocks) for the A->C transitions
            g2: graph (or list of 2 per-layer DGL blocks) for the C->A transitions
        Returns:
            classification logits
        """
        # Get graphs or blocks
        h_g1, h_g2 = h_g1.to(self._device), h_g2.to(self._device)
        if isinstance(g1, list):
            g1_0, g1_1 = g1[0].to(self._device), g1[1].to(self._device)
            g2_0, g2_1 = g2[0].to(self._device), g2[1].to(self._device)
            g1_0_nodes, g1_1_nodes = g1_0.number_of_dst_nodes(), g1_1.number_of_dst_nodes()
            g2_0_nodes, g2_1_nodes = g2_0.number_of_dst_nodes(), g2_1.number_of_dst_nodes()

        else:
            g1_0, g1_1 = g1.to(self._device), g1.to(self._device)
            g2_0, g2_1 = g2.to(self._device), g2.to(self._device)
            g1_0_nodes, g1_1_nodes = g1_0.number_of_nodes(), g1_1.number_of_nodes()
            g2_0_nodes, g2_1_nodes = g2_0.number_of_nodes(), g2_1.number_of_nodes()

        # Add edge weighted importance during forward pass
        h_AC1 = F.relu(self.convAC1.double()(g1_0, h_g1.double(), edge_weight=g1_0.edata['normalized'].double()))
        h_AC1 = h_AC1.reshape(g1_0_nodes, -1)
        h_CA1 = F.relu(self.convCA1.double()(g2_0, h_g2.double(), edge_weight=g2_0.edata['normalized'].double()))
        h_CA1 = h_CA1.reshape(g2_0_nodes, -1)

        # h_AC1 and h_CA1 can't be concatenated yet (unlike h_AC2/h_CA2 below): with DGL blocks
        # from a dataloader, their intermediate nodes aren't necessarily aligned at this hop
        h_AC1 = self.bn_gnn1.double()(h_AC1)
        h_CA1 = self.bn_gnn1.double()(h_CA1)

        h_AC2 = F.relu(self.convAC2.double()(g1_1, h_AC1, edge_weight=g1_1.edata['normalized'].double()))
        h_AC2 = h_AC2.reshape(g1_1_nodes, -1)
        h_CA2 = F.relu(self.convCA2.double()(g2_1, h_CA1, edge_weight=g2_1.edata['normalized'].double()))
        h_CA2 = h_CA2.reshape(g2_1_nodes, -1)

        # aggregate across transition types
        h = (torch.cat((h_AC2, h_CA2), dim=1))
        h = self.bn_gnn2(h.float())

        # project classes
        h = self.lin_readout(h)
        del h_AC1, h_CA1, h_AC2, h_CA2
        torch.cuda.empty_cache()
        return h.squeeze()

    def modality_prediction_fusion(self, h):
        """
        Aggregates per-modality predictions into one per-patient prediction, weighting each
        modality by a learned softmax weight `self.c` (fewer parameters than a full linear
        aggregator over all N*M rows).
        Args:
            h: per-modality-and-patient predictions, shape (M * nnodes, ...)
        Returns:
            per-patient predictions, shape (nnodes, ...)
        """
        bs = h.shape[0]  # current batch size (N*M,)
        nnodes = int(bs / self.M)  # number of patients in the batch

        # aggregate across modalities using a weighted sum, this reduces the number of learnable parameters
        w = F.softmax(self.c, dim=1).flatten()  # shape [M,]
        # expand each modality weight across N nodes
        weights = torch.repeat_interleave(w, nnodes).to(self._device)  # shape [M*N,]
        if self.n_classes is not None:
            weights = weights.unsqueeze(1)
        h = h * weights  # broadcast to logits
        # then sum modalities per node
        h = h.view(self.M, nnodes, -1).sum(dim=0)  # shape (N,)
        return h

    def create_multigraph(self, features, anchor_nodes):
        """
        Builds the two supra-adjacency matrices A (intra-modal) and C (cross-modal, with a
        self-loop diagonal) by thresholding and top-k-sparsifying the block-wise correlation
        between every pair of modalities, then derives the two 2-walk graphs g1 (A then C) and
        g2 (C then A) used by `gnn_forward`.
        Args:
            features: HGR projections for all modalities, shape (M * nnodes, mlp_proj_size)
            anchor_nodes: indices of nodes allowed to propagate messages; all nodes if None
        Returns:
            g1, g2: the two 2-walk multiplex graphs
        """
        bs = features.shape[0]  # current batch size (N*M)
        nnodes = int(bs / self.M)  # number of patients in the batch
        anchor_nodes = anchor_nodes if anchor_nodes is not None else [i for i in range(nnodes)]

        # lists for sparse representation
        A_rows, A_cols, A_vals = [], [], []
        C_rows, C_cols, C_vals = [], [], []

        features = features.to(self._device)
        # Zero the connections between each pair of non-anchor nodes, at each modality block
        mask_anchor = torch.eye(nnodes).to(features.device)  # always keep self-loops
        # nodes are ordered as : anchor nodes, non-anchor nodes (rows as well as columns)
        mask_anchor[:len(anchor_nodes), :] = 1  # propagation (rows) from anchor nodes only
        thresholds = F.sigmoid(self.sparse).to(self._device)
        cross_modal = False
        count = 0

        for i in range(self.M):
            for j in range(i, self.M):
                count += 1
                features_i = features[i * nnodes:(i + 1) * nnodes, :]
                features_j = features[j * nnodes:(j + 1) * nnodes, :]

                # compute correlations
                a_W_block = self.compute_block_correlation(features_i, features_j)

                mask_block = thresholds[count]
                a_W_block = F.relu(a_W_block - mask_block)

                topk = int(len(anchor_nodes) * self.k)

                # top-k per column
                _, indices_topk = torch.topk(
                    a_W_block,
                    k=topk,
                    dim=0,
                    largest=True
                )
                topk_mask = torch.zeros_like(a_W_block, dtype=torch.bool)
                topk_mask.scatter_(dim=0, index=indices_topk, value=True)

                # enforce anchor propagation
                mask = mask_anchor * topk_mask
                a_W_block = a_W_block * mask
                a_W_block_T = a_W_block.transpose(0, 1) * mask

                # collect nonzeros
                if i == j:
                    # self-block -> goes to A
                    rows, cols = torch.nonzero(a_W_block, as_tuple=True)
                    vals = a_W_block[rows, cols]
                    A_rows.append(rows + i * nnodes)
                    A_cols.append(cols + j * nnodes)
                    A_vals.append(vals)
                else:
                    if (a_W_block != 0).any():
                        cross_modal = True

                        # forward block
                        rows, cols = torch.nonzero(a_W_block, as_tuple=True)
                        vals = a_W_block[rows, cols]
                        C_rows.append(rows + i * nnodes)
                        C_cols.append(cols + j * nnodes)
                        C_vals.append(vals)

                        # transpose block
                        rows, cols = torch.nonzero(a_W_block_T, as_tuple=True)
                        vals = a_W_block_T[rows, cols]
                        C_rows.append(rows + j * nnodes)
                        C_cols.append(cols + i * nnodes)
                        C_vals.append(vals)

        del features, mask_anchor, mask, topk_mask, a_W_block, a_W_block_T
        torch.cuda.empty_cache()

        curr_device = self._device if self.training else 'cpu'

        # === Build A ===
        if A_rows:
            A_indices = torch.stack([torch.cat(A_rows), torch.cat(A_cols)], dim=0).to(curr_device)
            A_values = torch.cat(A_vals).to(curr_device)
            A = torch.sparse_coo_tensor(A_indices, A_values, (bs, bs)).coalesce()
            del A_indices, A_values
        else:
            A = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.tensor([]), (bs, bs)).to(curr_device)

        # === Build C (with diagonal identity) ===
        diag_idx = torch.arange(bs, device=curr_device)
        diag_indices = torch.stack([diag_idx, diag_idx], dim=0)
        diag_values = torch.ones(bs, device=curr_device)

        if C_rows:
            C_indices = torch.stack([torch.cat(C_rows), torch.cat(C_cols)], dim=0).to(curr_device)
            C_values = torch.cat(C_vals).to(curr_device)

            # add diagonals
            all_indices = torch.cat([C_indices, diag_indices], dim=1)
            all_values = torch.cat([C_values, diag_values])
            C = torch.sparse_coo_tensor(all_indices, all_values, (bs, bs)).coalesce().to(curr_device)
            del all_indices, all_values, C_indices, C_values
        else:
            # only diagonals
            C = torch.sparse_coo_tensor(diag_indices, diag_values, (bs, bs)).coalesce().to(curr_device)

        del A_rows, A_cols, A_vals, C_rows, C_cols, C_vals, diag_idx, diag_indices, diag_values
        torch.cuda.empty_cache()

        g1 = self.build_2walks_graph(A, C, bs) if cross_modal else self.build_2walks_graph(A, C, bs, mat_mul=A)
        g2 = self.build_2walks_graph(C, A, bs) if cross_modal else g1
        self.cross_modal = cross_modal
        self.A_edges = A._nnz() / nnodes
        self.C_edges = C._nnz() / nnodes - 1 # remove self-loops

        del A, C
        torch.cuda.empty_cache()
        if not self.training:
            self.n1_edges = int(g1.num_edges() / g1.num_nodes())
            self.n2_edges = int(g2.num_edges() / g2.num_nodes())
        return g1, g2

    def build_2walks_graph(self, mat1, mat2, bs, mat_mul=None):
        """
        Builds a DGL graph from the sparse product `mat1 @ mat2` (a 2-hop walk), with edges
        normalized by `1 / sqrt(degree_i * degree_j)` for a stable forward pass.
        Args:
            mat1: left sparse COO matrix of the 2-hop walk
            mat2: right sparse COO matrix of the 2-hop walk
            bs: total number of nodes (N * M), used so disconnected nodes aren't dropped
            mat_mul: precomputed `mat1 @ mat2` product; computed here if None
        Returns:
            the resulting DGL graph, with per-edge normalization in `g1.edata['normalized']`
        """
        if mat_mul is None:
            mat1 = mat1.to(self._device).to_sparse_csr()
            mat2 = mat2.to(self._device).to_sparse_csr()
            try:
                mat_mul = torch.sparse.mm(mat1, mat2).to('cpu').to_sparse_coo()
                del mat1, mat2
                torch.cuda.empty_cache()
                gc.collect()
            except:
                print('Sparse mm failed, using cpu')
                with FileLock("/tmp/my_heavy_op.lock"):
                    # heavy memory operation, lock to avoid multiple ray workers to run this simultaneously
                    mat1, mat2 = mat1.to('cpu'), mat2.to('cpu')
                    mat_mul = torch.sparse.mm(mat1.to_sparse_csr(), mat2).to_sparse_coo()
                    del mat1, mat2
                    torch.cuda.empty_cache()
                    gc.collect()

        # adding num_nodes to avoid running into the situation where disconnected nodes are dropped completely
        edge_r, edge_c = mat_mul.indices()
        edge_r, edge_c = edge_r.to('cpu'), edge_c.to('cpu')
        g1 = dgl.graph((edge_r, edge_c), num_nodes=bs)

        # normalise adjacency matrices for stable forward pass
        degs = g1.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g1.ndata['norm'] = norm.unsqueeze(1)
        g1.apply_edges(fn.u_mul_v('norm', 'norm', 'normalized'))  # computes weight for each edge aij = 1 / sqrt(di*dj)

        del mat_mul, edge_r, edge_c, degs, norm
        torch.cuda.empty_cache()

        return g1

    @staticmethod
    def compute_block_correlation(x, y):
        """
        Computes the correlation between each row of x and each row of y.
        Args:
            x: tensor, shape (n_samples, n_features)
            y: tensor, shape (n_samples, n_features)
        Returns:
            an (n_samples, n_samples) matrix of absolute correlations, with the identity matrix
            subtracted off (removes self-correlation when x and y are the same block, i.e. i == j)
        """
        # Center each row
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)

        # Normalize each row (rowwise std)
        x_norm = x_centered / x_centered.norm(dim=1, keepdim=True)
        y_norm = y_centered / y_centered.norm(dim=1, keepdim=True)

        # Correlation = cosine similarity of normalized centered rows
        corr = x_norm @ y_norm.T
        return torch.abs(corr) - torch.eye(x.shape[0], device=x.device)

    def forward(self,
                x: List[Union[torch.Tensor, List[torch.Tensor]]],
                lengths: Optional[List[int]] = None,
                graph_loader_batch_size=None,
                idx_anchors=None,
                graph_learning=True):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
            graph_loader_batch_size: if set, runs the GNN in batches via a DGLDataLoader instead
                of on the full graph at once (for graphs too large to fit in memory)
            idx_anchors: indices of nodes that can send messages; if None, all nodes can
            graph_learning: if False, skips graph construction and the GNN, returning
                placeholder (None) logits and graph (used during the HGR-only warmup phase)
        Returns:
            logits, h_mod, g1: predicted logits for the non-anchor nodes (None if
            `graph_learning` is False), the per-modality HGR projections, and the multiplex
            graph g1 (only g1 is returned, to compute edges-per-node downstream)
        """
        # 1- GRU and HGR forward
        h_mod, h_hgr = self.hgr_forward(x, lengths)

        logits, g1 = None, None

        if graph_learning:
            if idx_anchors is not None:
                idx_non_anchors = [i for i in range(h_mod[0].shape[0]) if i not in idx_anchors]
            else:
                idx_non_anchors = [i for i in range(h_mod[0].shape[0])]

            # 2- Create multiplex graph (M plans, with N nodes at each plan, with intra and cross planar connections)
            g1, g2 = self.create_multigraph(h_hgr, idx_anchors)

            # 3- Create graph data loader
            if graph_loader_batch_size is not None:
                # Get relative indices of the non-anchor nodes in the multiplex graph
                nid_non_anchors = []
                for i in range(self.M):  # for each modality block
                    nid_non_anchors.extend([n + i * h_mod[0].shape[0] for n in idx_non_anchors])

                graph_sampler = MultiLayerNeighborSampler([-1] * 2)  # -1 means all neighbors
                dataloader = DGLDataLoader(g1,
                                           g1.nodes()[nid_non_anchors],
                                           graph_sampler,
                                           batch_size=graph_loader_batch_size,
                                           shuffle=False,
                                           drop_last=False)

                # sanity check
                if nid_non_anchors != g1.nodes()[nid_non_anchors].tolist():
                    raise ValueError(f'node ids in the graph and the list of non-anchor nodes do not match')

                # Get the logits for each batch and concatenate them
                if self.n_classes is None:
                    h_gnn = torch.zeros(len(nid_non_anchors), device=self._device)
                else:
                    h_gnn = torch.zeros((len(nid_non_anchors), self.n_classes), device=self._device)

                id_to_index = {nid.item(): i for i, nid in enumerate(g1.nodes()[nid_non_anchors])}

                for data in dataloader:
                    input_nodes_g1, output_nodes_g1, g1_blocks = data

                    # Use output nodes to sample the g2 graph with connections to that nodes
                    input_nodes_g2, output_nodes_g2, g2_blocks = graph_sampler.sample(g2, output_nodes_g1)

                    # sanity checks
                    if not torch.eq(output_nodes_g1, output_nodes_g2).all():
                        raise ValueError(f'targets nodes in g1 and g2 must match')

                    else:
                        output_nodes = output_nodes_g1

                    # Get the indices of the output nodes in the set
                    indices = [id_to_index[nid] for nid in output_nodes.tolist()]

                    # Apply the GNN to get the logits for the batch
                    h_gnn[indices] = self.gnn_forward(h_hgr[input_nodes_g1.to(h_hgr.device)],
                                                      h_hgr[input_nodes_g2.to(h_hgr.device)],
                                                      g1_blocks,
                                                      g2_blocks)
                del dataloader, graph_sampler, input_nodes_g1, output_nodes_g1, g1_blocks, input_nodes_g2, output_nodes_g2, g2_blocks

            else:
                # Use Multiplex graph to compute predictions, no need for idx non-anchors here
                h_gnn = self.gnn_forward(h_hgr, h_hgr, g1, g2)

            # Fuse modalities
            logits = self.modality_prediction_fusion(h_gnn)
            del h_gnn, g2

        del h_hgr
        torch.cuda.empty_cache()
        return logits, h_mod, g1  # return only g1 to compute number of edges per node
