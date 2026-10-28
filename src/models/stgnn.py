"""
Filename: stgnn.py

Author: Hakima Laribi

Description: STGNN baseline (Multimodal Spatiotemporal Graph Neural Network). GraphGRU is a GRU whose
gates are computed by GNN message passing over a patient population graph, so each timestep's hidden state uses the
node's neighbors. STGNNFuser applies a GraphGRU to temporal modalities and a GNN to static ones, then fuses them with
an MLP.

Code adapted to our framework from: https://github.com/tsy935/readmit-stgnn
"""
from typing import List, Literal, Optional, Union

import dgl
import torch
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from dgl.nn.pytorch.conv import GATConv, SAGEConv
from torch import nn
from torch.nn import ModuleList
from torch.nn import functional as F

from src.models.abstract_fusers import Fuser


def _build_graph_conv_stack(conv_cls, n_layers: int, input_size: int, output_size: int, device, **kwargs) -> ModuleList:
    """
    Creates a stack of `n_layers` graph convolution layers (GAT or SAGE). Shared by GraphGRU
    and STGNNFuser, which both build their GNN stacks the same way.
    Args:
        conv_cls: graph convolution layer class to instantiate (GATConv or SAGEConv)
        n_layers: number of layers to stack
        input_size: size of the input features
        output_size: size of the output features of every layer
        device: device to move the resulting layers to
    Returns:
        an nn.ModuleList of graph convolution layers
    """
    gnn_layers = [conv_cls(input_size, output_size, **kwargs)]
    for _ in range(n_layers - 1):
        gnn_layers.append(conv_cls(output_size, output_size, **kwargs))
    return ModuleList(gnn_layers).to(device)


class GraphGRU(nn.Module):
    """GRU cell whose gates are computed by GNN message passing instead of linear layers."""

    def __init__(self,
                 input_size: int,
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 gnn_n_layers: int = 1,
                 gnn_hidden_size: int = 2,
                 gnn_type: Literal['gat', 'sage'] = 'sage',
                 temporal_hidden_size: int = 256,
                 max_seq_len: int = None,
                 **kwargs,  # attributes for the graph conv layer (e.g., num_heads for GATConv)
                 ):
        """
        Args:
            input_size: input feature size of the modality
            dropout: dropout probability applied between intermediate GNN layers
            device: device to run the model on
            gnn_n_layers: number of GNN layers per gate
            gnn_hidden_size: hidden size of the GNN layers (unused here; kept for API parity)
            gnn_type: GNN layer type used for message passing ('gat' or 'sage')
            temporal_hidden_size: hidden size of the GRU state
            max_seq_len: maximum sequence length to unroll; longer sequences are truncated
        """
        super().__init__()
        self._device = device
        self._dropout = dropout
        self.input_size = input_size
        self.gnn_conv = SAGEConv if gnn_type == 'sage' else GATConv
        self.gnn_n_layers = gnn_n_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self._max_seq_len = max_seq_len

        # separate GNN stacks for the update/reset gates and for the candidate state
        self.g_conv_gate = self.create_graph_conv_layer(input_size, temporal_hidden_size, **kwargs)
        self.g_conv_candidate = self.create_graph_conv_layer(input_size, temporal_hidden_size, **kwargs)

        # one linear layer computes both the update and reset gates (split in half below),
        # since they only differ in their own parameters, same as in a standard GRU
        self.gate_linear = nn.Linear(temporal_hidden_size * 2, temporal_hidden_size * 2)
        self.candidate_linear = nn.Linear(temporal_hidden_size * 2, temporal_hidden_size)

    def create_graph_conv_layer(self, input_size: int, output_size: int, **kwargs) -> nn.ModuleList:
        """
        Creates a stack of `self.gnn_n_layers` graph convolution layers (GAT or SAGE).
        Args:
            input_size: size of the input features
            output_size: size of the output features of every layer
        Returns:
            an nn.ModuleList of graph convolution layers
        """
        return _build_graph_conv_stack(self.gnn_conv, self.gnn_n_layers, input_size, output_size, self._device,
                                       **kwargs)

    def gnn_forward_pass(self, gnn_layers: ModuleList, graph, z: torch.Tensor) -> torch.Tensor:
        """
        Propagates messages through the GNN layers.
        Args:
            gnn_layers: stack of GNN layers to run z through sequentially
            graph: DGL graph (or list of per-layer DGL blocks) to message-pass over
            z: input node features
        Returns:
            updated node features after message passing
        """
        h = z
        for n_layer in range(self.gnn_n_layers):
            # different bipartite graph per layer when using DGL block batching
            g_layer = graph[n_layer] if isinstance(graph, list) else graph
            h = gnn_layers.double()[n_layer](g_layer, h.double())
            h = F.relu(h)
            if n_layer != self.gnn_n_layers - 1:
                h = F.dropout(h, p=self._dropout)

            if isinstance(gnn_layers[n_layer], GATConv):
                h = h.mean(dim=1)  # aggregate over multiple attention heads

        return h.float()

    def forward(self,
                g: Union[DGLGraph, DGLBlock],
                x: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Unrolls the graph-convolutional GRU over the sequence, one GNN-based update per timestep.

        Args:
            g: DGL graph (or list of per-layer DGL blocks) to message-pass over
            x: node sequences, shape (num_nodes [+ neighbors], seq_len, input_size)
            lengths: actual sequence length of each node, shape (num_nodes [+ neighbors],)
        Returns:
            the final hidden state, shape (num_nodes, temporal_hidden_size)
        """
        # truncate to the longest sequence actually present in this batch
        if isinstance(g, list):
            x = x[:, :lengths[g[-1].dstnodes().detach().cpu()].max(), :]
            input_shape = g[-1].dstnodes().shape[0]
        else:
            x = x[:, :lengths.max(), :]
            input_shape = x.shape[0]

        seq_len = min(self._max_seq_len, x.shape[1]) if self._max_seq_len is not None else x.shape[1]

        h_t = torch.zeros((input_shape, self.temporal_hidden_size)).to(self._device)

        for t in range(seq_len):
            x_t = x[:, t, :].squeeze(1)

            if (torch.cuda.device_count() > 1) & ('cuda' in self._device):
                # split the graph convolutions onto a second GPU when one is available
                x_t = x_t.to('cuda:1')
                g = [gi.to('cuda:1') for gi in g] if isinstance(g, list) else g.to('cuda:1')
                self.g_conv_gate = self.g_conv_gate.to('cuda:1')
                self.g_conv_candidate = self.g_conv_candidate.to('cuda:1')

            x_gate = self.gnn_forward_pass(self.g_conv_gate, g, x_t).to(self._device)

            z_t = torch.cat([x_gate, h_t], dim=1)
            h_gate = torch.sigmoid(self.gate_linear(z_t))
            u_t, r_t = torch.split(h_gate, self.temporal_hidden_size, dim=1)

            x_candidate = self.gnn_forward_pass(self.g_conv_candidate, g, x_t).to(self._device)

            h_candidate = self.candidate_linear(torch.cat([x_candidate, r_t * h_t], dim=1))
            c_t = torch.tanh(h_candidate)

            h_t = u_t * h_t + (1 - u_t) * c_t

        return h_t


class STGNNFuser(Fuser):
    """Applies a GraphGRU to temporal modalities and a plain GNN to static ones, then fuses them."""

    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 gnn_n_layers: int = 1,
                 gnn_hidden_size: int = 2,
                 fuser_n_layers: int = 1,
                 fuser_hidden_size: int = 2,
                 gnn_type: Literal['gat', 'sage'] = 'sage',
                 max_seq_len: int = None,
                 n_classes: int = None,
                 **kwargs,  # attributes for the graph conv layer (e.g., num_heads for GATConv)
                 ):
        """
        Args:
            gnn_n_layers: number of GNN layers per static modality and per GraphGRU gate
            gnn_hidden_size: hidden size of the static-modality GNN layers
            fuser_n_layers: number of hidden layers in the fusion MLP
            fuser_hidden_size: hidden size of the fusion MLP
            gnn_type: GNN layer type used for message passing ('gat' or 'sage')
            max_seq_len: maximum sequence length to unroll in GraphGRU; longer ones are truncated
            n_classes: number of classes; None for binary classification
            See Fuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size)

        self.gnn_conv = SAGEConv if gnn_type == 'sage' else GATConv
        self.gnn_n_layers = gnn_n_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_type = gnn_type
        self.fuser_n_layers = fuser_n_layers
        self.fuser_hidden_size = fuser_hidden_size
        self.n_classes = n_classes

        # a plain GNN for static modalities, a GraphGRU for temporal ones
        self._graph_embedder = [
            self.create_graph_conv_layer(input_sizes[i], self.gnn_hidden_size, **kwargs)
            if not temporal_analysis[i]
            else GraphGRU(input_size=input_sizes[i],
                          dropout=self._dropout,
                          device=self._device,
                          temporal_hidden_size=temporal_hidden_size,
                          gnn_n_layers=self.gnn_n_layers,
                          gnn_hidden_size=self.gnn_hidden_size,
                          gnn_type=gnn_type,
                          max_seq_len=max_seq_len,
                          **kwargs)
            for i in range(len(input_sizes))]
        self._graph_embedder = nn.ModuleList(self._graph_embedder).to(self._device)

        mlp_input_shape = sum([gnn_hidden_size if not temporal_analysis[i]
                               else self._temporal_hidden_size
                               for i in range(len(input_sizes))])
        self._mlp_fuser = self.create_mlp_fuser(n_inputs=mlp_input_shape,
                                                mlp_hidden_size=fuser_hidden_size,
                                                mlp_layers=self.fuser_n_layers).to(self._device)

    def create_graph_conv_layer(self, input_size: int, output_size: int, **kwargs) -> nn.ModuleList:
        """
        Creates a stack of `self.gnn_n_layers` graph convolution layers (GAT or SAGE).
        Args:
            input_size: size of the input features
            output_size: size of the output features of every layer
        Returns:
            an nn.ModuleList of graph convolution layers
        """
        return _build_graph_conv_stack(self.gnn_conv, self.gnn_n_layers, input_size, output_size, self._device,
                                       **kwargs)

    def forward(self,
                x: List[Union[torch.Tensor, List[torch.Tensor]]],
                lengths: Optional[List[int]] = None,
                g: Optional[dgl.DGLGraph] = None) -> torch.Tensor:
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
            g: DGL graph (or list of per-layer DGL blocks) to message-pass over
        Returns:
            the fused logits
        """
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        output = []
        for i, x_modality in enumerate(x):
            x_modality = (torch.stack(x_modality) if isinstance(x_modality, list) else x_modality).to(self._device)

            if not self._temporal_analysis[i]:
                h = x_modality
                for n_layer, gnn_layer in enumerate(self._graph_embedder[i]):
                    if not isinstance(g, list):
                        h = gnn_layer.double()(g.to(self._device), h.double())
                    else:
                        h = gnn_layer.double()(g[n_layer].to(self._device), h.double())

                    h = F.relu(h)

                    if n_layer != self.gnn_n_layers - 1:  # dropout only for intermediate layers
                        h = F.dropout(h, p=self._dropout)

                    if self.gnn_type == 'gat':
                        h = h.mean(dim=1)  # aggregate over multiple attention heads
            else:
                h = self._graph_embedder[i](g, x_modality, lengths[i])

            output.append(h.float())

        fused_input = torch.cat(output, dim=1)
        transformed_input = self._mlp_fuser(fused_input).view(-1)

        return transformed_input