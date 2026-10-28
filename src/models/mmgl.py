"""
Filename: mmgl.py

Author: Hakima Laribi

Description: MMGL baseline (Multimodal Graph Learning). MARL fuses the modalities of each patient with
attention, GraphLearner learns a patient population graph from the fused representations (optionally sparsified
with top-k), and GraphSAGE predicts the outcome on it.

Code adapted to our framework from: https://github.com/SsGood/MMGL
"""
import copy
from typing import List, Literal, Optional, Union

import dgl
import torch
from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
from dgl.nn.pytorch.conv import SAGEConv
from torch import nn
from torch.nn import functional as F

from src.models.abstract_fusers import PostTemporalFuser
from src.models.layers import FeedForwardLayer, HeterogeneousTransformer, HomogeneousTransformer, OutputLayer


class MARL(PostTemporalFuser):
    """
    Modality-Aware Representation Learning module. Learns patient representations from multiple modalities. Uses
    Transformers to learn the relationships between modalities. Use an MLP and a FC to predict the patient labels from
    the hidden representations (H_sh) and the attention weights (H_sp) of the last transformer block.

    We tried to replicate the code in : https://github.com/SsGood/MMGL/blob/main/MMGL_inductive/network.py#L25
    while respecting the methodology provided in the paper: 10.1109/TMI.2022.3159264
    """

    def __init__(self,
                 input_sizes: List[int],
                 temporal_analysis: List[bool],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 num_att_heads: int = 8,
                 transformer_hidden_size: int = 128,
                 n_transformer_layers: int = 1,
                 n_classes: int=None):
        """
        Args:
            num_att_heads: number of attention heads in each transformer block
            transformer_hidden_size: per-head hidden size of the transformer blocks
            n_transformer_layers: number of homogeneous transformer (+ feed-forward) blocks
                stacked after the initial heterogeneous one
            n_classes: number of classes; None for binary classification
            See PostTemporalFuser.__init__ for the remaining args.
        """
        # Initialize the MARL model parameters and structures here
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size, n_classes)

        # Set attributes
        self._device = device
        self._dropout = dropout
        self.num_att_heads = num_att_heads
        self.transformer_hidden_size = transformer_hidden_size
        self.n_transformer_layers = n_transformer_layers

        self._hetero_dims_transformer = HeterogeneousTransformer(num_att_heads,
                                                                 device,
                                                                 self.compute_self_attention,
                                                                 len(self._embedded_sizes),
                                                                 [input_shape for input_shape in self._embedded_sizes],
                                                                 transformer_hidden_size,
                                                                 dropout)
        # usually we add FFN after the transformer block but not done in their code
        self._homo_dims_transformer = nn.ModuleList([HomogeneousTransformer(num_att_heads,
                                                                            device,
                                                                            self.compute_self_attention,
                                                                            len(self._embedded_sizes),
                                                                            transformer_hidden_size * num_att_heads,
                                                                            transformer_hidden_size,
                                                                            dropout)
                                                     for _ in range(n_transformer_layers)])

        # Add a feedforward for each homo_dim_transformer
        self._feed_forward = nn.ModuleList([FeedForwardLayer(transformer_hidden_size * num_att_heads,
                                                             transformer_hidden_size * num_att_heads,
                                                             dropout) for _ in range(n_transformer_layers)])

        # Final Layer for prediction with MLP using h_sp (before the graph construction)
        self.mlp_sp = OutputLayer(transformer_hidden_size * num_att_heads * len(input_sizes),
                                  transformer_hidden_size * num_att_heads,
                                  len(input_sizes),
                                  dropout,
                                  1 if self.n_classes is None else self.n_classes)

    def compute_self_attention(self, q, k, v, tau):
        """
        Scaled dot-product self-attention, passed to HeterogeneousTransformer/HomogeneousTransformer.
        Args:
            q: query tensor, shape (batch_size, num_att_heads, num_modalities, hidden_size)
            k: key tensor, same shape as q
            v: value tensor, same shape as q
            tau: scaling factor applied to the attention scores (typically sqrt(hidden_size))
        Returns:
            att_output, att_weights: the weighted sum of values, and the normalized attention
            weights, both shape (batch_size, num_att_heads, num_modalities, ...)
        """
        # Compute attention scores, shape: (batch_size, num_att_heads, num_modalities, num_modalities)
        att_scores = torch.matmul(q, k.transpose(2, 3)) / tau

        # Apply softmax to get attention weights normalized
        att_weights = F.softmax(att_scores, dim=-1)  # we want all columns of each modality to sum to 1

        # attention dropout (done in the original transformer but not in the paper)
        att_weights = F.dropout(att_weights, p=self._dropout, training=self.training)

        # Weighted sum of values, shape : (bs, n_heads, num_modalities, hidden_size)
        att_output = torch.matmul(att_weights, v)

        return att_output, att_weights

    def forward(self, x: List[Union[torch.Tensor, List[torch.Tensor]]], lengths: Optional[List[int]]):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            logits, h_mlpsh_sp: predicted logits, and the concatenation H_sp || H_sh (used for
            graph construction downstream)
        """
        # Validate that the List x is of the same size as temporal_analysis
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        # Transform each modality using the corresponding embedder (Temporal or Identity)
        h_temporal = [self.transform_modality(x_modality, i, lengths) for i, x_modality in enumerate(x)]

        bs = h_temporal[0].shape[0]  # batch size
        output, att_weights = self._hetero_dims_transformer(h_temporal) #output: (bs, n_mod, att_heads * hidden_size)
        # Aggregate attention weights across heads
        att_weights = att_weights.mean(dim=1)  # shape: (bs, n_modalities, n_modalities)

        for i in range(self.n_transformer_layers):
            # second transformer block, attention, FC, dropout, addition with V, layer norm
            output, att_weights = self._homo_dims_transformer[i](output)
            # Aggregate attention across heads
            att_weights = att_weights.mean(dim=1)
            # Feed Forward of the classic transformer block
            output = self._feed_forward[i](output)

        output = output.view(bs, -1)  # shape : (batch_size, heads * hidden_size * num_modalities), this is H_sh
        attn_embedding = att_weights.view(bs, -1)  # shape (batch_size, num_modalities*num_modalities) this is H_sp
        logits, h_mlpsh_sp = self.mlp_sp(output, attn_embedding)  # we only use the attention of the last transformer
        # h_mlpsh_sp is the concatenation H_sp || H_sh referred to in the paper (used for graph construction)
        return logits, h_mlpsh_sp


class GraphLearner(nn.Module):
    """Builds a similarity graph from node representations via a learned cosine-similarity metric."""

    def __init__(self, input_size, device, top_percent):
        """
        Args:
            input_size: size of the input node representations
            device: device to run the model on
            top_percent: fraction of highest-similarity edges to keep per node (e.g. 0.01 for top 1%)
        """
        super().__init__()
        # Create the Linear projection for graph construction
        self._train_device = device
        self._device = device
        self.w_gc = nn.Linear(input_size, input_size).to(self._device)
        self.threshold = 0.9 # using the default value of the paper
        self.top_percent = top_percent

    def compute_adjacency_matrix(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x1: node representations to compute similarities from (rows of the adjacency matrix)
            x2: node representations to compute similarities to (columns); defaults to x1
        Returns:
            a [x1.size(0), x2.size(0)] adjacency matrix: cosine similarities above both the
            per-row top_percent quantile and `self.threshold`, zero elsewhere
        """
        x2 = x2 if x2 is not None else x1
        x1, x2 = x1, x2

        # 1. Apply learnable transformation
        x1 = self.w_gc(x1.to(self._device))
        x2 = self.w_gc(x2.to(self._device))

        # 2. Normalize node features to unit length (for cosine similarity)
        x1_norm = F.normalize(x1, p=2, dim=1)  # shape [bs, feat_dim]
        x2_norm = F.normalize(x2, p=2, dim=1)  # shape [bs, feat_dim]

        # 3. Compute cosine similarity between all pairs
        cos_sim = torch.mm(x1_norm, x2_norm.T)  # shape [bs1, bs2]

        # Find threshold for each row (99th percentile → top 1%)
        thresholds = torch.quantile(cos_sim, 1 - self.top_percent, dim=1, keepdim=True)

        # Build mask
        th = torch.max(thresholds.squeeze(), torch.tensor([self.threshold]*thresholds.shape[0]).to(self._device))
        mask = (cos_sim >= th.unsqueeze(dim=1)).detach().float()
        markoff_value = 0
        output = cos_sim * mask + markoff_value * (1 - mask)

        return output


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE node classifier, run in double precision for reproducibility."""

    def __init__(self, in_feats, h_feats, dropout, device, n_classes=None):
        """
        Args:
            in_feats: size of the input node features
            h_feats: hidden size of the first SAGEConv layer
            dropout: dropout probability applied between the two SAGEConv layers
            device: device to run the model on
            n_classes: number of classes; None for binary classification
        """
        super(GraphSAGE, self).__init__()
        # necessary to use double precision to avoid precision errors that cause non-reproducibility
        self._train_device = device
        self._device = device
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean').to(self._device).double()
        self.conv2 = SAGEConv(h_feats, 1 if n_classes is None else n_classes, 'mean').to(self._device)
        self.dropout = dropout

    def forward(self, g, x):
        """
        Args:
            g: DGL graph (or list of 2 per-layer DGL blocks) to message-pass over
            x: input node features
        Returns:
            logits, moved back to `self._train_device`
        """
        x = x.to(self._device)
        g = g.to(self._device) if not isinstance(g, list) else [gi.to(self._device) for gi in g]
        h = F.relu(self.conv1.double()(g[0]if isinstance(g, list) else g, x.double()))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2.double()(g[1] if isinstance(g, list) else g, h.double()).float()  # logits
        return h.squeeze().to(self._train_device).float()


class MMGLFuser(nn.Module):
    """Group MARL, GraphLearner and GraphSAGE together into the full MMGL fuser."""

    def __init__(self,
                input_sizes,
                temporal_analysis,
                dropout,
                device,
                temporal_hidden_size,
                num_att_heads,
                transformer_hidden_size,
                n_transformer_layers,
                gnn_hidden_size,
                top_percent,
                 n_classes=None
                ):
        """
        Args:
            input_sizes: per-modality input feature size
            temporal_analysis: per-modality flag, True if the modality has temporal sequences
            dropout: dropout probability used throughout MARL
            device: device to run the model on
            temporal_hidden_size: hidden size of the GRU layers used for temporal modalities
            num_att_heads: number of attention heads in MARL's transformer blocks
            transformer_hidden_size: per-head hidden size of MARL's transformer blocks
            n_transformer_layers: number of homogeneous transformer blocks in MARL
            gnn_hidden_size: divides the GraphLearner output size to get the GNN hidden size
            top_percent: fraction of highest-similarity edges to keep per node in GraphLearner
            n_classes: number of classes; None for binary classification
        """
        super(MMGLFuser, self).__init__()
        self._device = device
        self.n_classes = n_classes
        # Create the Modality-Aware Representation Learning module
        self.marl = MARL(input_sizes,
                         temporal_analysis,
                         dropout,
                         device,
                         temporal_hidden_size,
                         num_att_heads,
                         transformer_hidden_size,
                         n_transformer_layers,
                         n_classes=n_classes).to(self._device)

        # Create the graph constructor
        in_shape = transformer_hidden_size * num_att_heads + len(input_sizes)**2 # H_sh + H_sp
        self.graph_learner = GraphLearner(in_shape, device, top_percent)

        # Create the GNN for node classification
        self.gnn = GraphSAGE(in_shape, int(in_shape / gnn_hidden_size), dropout, device, n_classes=n_classes)

    def forward(self, x, lengths, idx_anchors=None, g_anchors=None, graph_loader_batch_size=None):
        """
        Forward pass of the MMGL model. Used for inference only (evaluation and prediction); in
        training, MARL and the GNN are optimized separately since they use different optimizers.
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
            idx_anchors: indices of nodes that can send messages; if None, all nodes can (as in
                the training phase)
            g_anchors: base DGL graph of the anchor nodes; if not None, only edges from anchor
                to non-anchor nodes are added to it instead of rebuilding the full graph
            graph_loader_batch_size: if set, runs the GNN in batches via a DGLDataLoader instead
                of on the full graph at once (for graphs too large to fit in memory)
        Returns:
            gnn_logits, None, g, h_mlpsh_sp: predicted logits for the non-anchor nodes, a
            placeholder (kept for API parity with other trainers), the graph used, and the
            MARL hidden representations (H_sp || H_sh)
        """
        # 1. Get the logits and the hidden representation from MARL
        logits, h_mlpsh_sp = self.marl(x, lengths)
        if idx_anchors is not None:
            idx_non_anchors = [i for i in range(h_mlpsh_sp.shape[0]) if i not in idx_anchors]
            logits = logits[-len(idx_non_anchors):]

        # 2. Compute the adjacency matrix using the graph learner
        if idx_anchors is None:
            idx_non_anchors = [i for i in range(h_mlpsh_sp.shape[0])]
            adj_matrix = self.graph_learner.compute_adjacency_matrix(h_mlpsh_sp)
        else:
            idx_non_anchors = [i for i in range(h_mlpsh_sp.shape[0]) if i not in idx_anchors]
            if g_anchors is not None:
                # We have the base graph of connections between anchor nodes, add only edges from anchor to non-anchor
                adj_matrix = self.graph_learner.compute_adjacency_matrix(h_mlpsh_sp[idx_non_anchors],
                                                                         h_mlpsh_sp[idx_anchors])
            else:
                # compute all edges between anchors nodes, in addition to edges from anchors to non-anchors
                adj_matrix = self.graph_learner.compute_adjacency_matrix(h_mlpsh_sp, h_mlpsh_sp[idx_anchors])

        # 3. Create the DGL graph from the adjacency matrix
        dst, src = adj_matrix.nonzero(as_tuple=True) # edge weights are not used in the GNN (binary adjacency matrix)
        if (g_anchors is not None) & (idx_anchors is not None):
            g = copy.deepcopy(g_anchors)
            # shift the dst indices to use them as new node ids
            dst = torch.tensor([i + len(idx_anchors) for i in dst], dtype=torch.int64).to(self._device)
            # Add the new nodes to the graph
            g.add_nodes(len(idx_non_anchors))
            g.add_edges(src, dst)

        else:
            g = dgl.graph((src, dst), num_nodes=adj_matrix.shape[0])
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # 4. Create DGLDataloader if the graph is too large (non training phases)
        if graph_loader_batch_size is not None:
            graph_sampler = MultiLayerNeighborSampler([-1] * 2)  # -1 means all neighbors
            dataloader = DGLDataLoader(g,
                                       g.nodes()[-len(idx_non_anchors):],
                                       graph_sampler,
                                       batch_size=graph_loader_batch_size,
                                       shuffle=False,
                                       drop_last=False)

            # Get the logits for each batch and concatenate them
            if self.n_classes is None:
                gnn_logits = torch.zeros(len(idx_non_anchors), device=self._device)
            else:
                gnn_logits = torch.zeros((len(idx_non_anchors), self.n_classes), device=self._device)
            id_to_index = {nid.item(): i for i, nid in enumerate(g.nodes()[-len(idx_non_anchors):])}
            for data in dataloader:
                input_nodes, output_nodes, g_blocks = data
                # Get the indices of the output nodes in the set (if we remove anchor nodes)
                indices = [id_to_index[nid] for nid in output_nodes.tolist()]
                # Apply the GNN to get the logits for the batch
                gnn_logits[indices] = self.gnn(g_blocks, h_mlpsh_sp[input_nodes.to(h_mlpsh_sp.device)])

        else:
            # 5. Apply the GNN to get the final logits
            gnn_logits = self.gnn(g, h_mlpsh_sp)

        return gnn_logits, None, g, h_mlpsh_sp
