"""
Filename: sgnn.py

Author: Hakima Laribi

Description: SGNN, the non-temporal adaptation of STGNN. Each modality is propagated over a patient
population graph built from demographic similarities, then the modality representations are fused with an MLP.
"""
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import dgl
import torch
from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv
from torch import nn
from torch.nn import functional as F

from src.models.abstract_fusers import PostTemporalFuser


class GNNFuser(PostTemporalFuser, ABC):
    """
    Abstract base for GNN-based fusers: embeds each modality through its own stack of GNN
    layers (the layer type is set by subclasses via build_conv_layer), then concatenates and
    fuses the embeddings with an MLP.
    """

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
                 gnn_type: str = None,
                 n_classes: int = None
                 ):
        """
        Args:
            gnn_n_layers: number of GNN layers per modality
            gnn_hidden_size: hidden (and output) size of every GNN layer
            fuser_n_layers: number of hidden layers in the fusion MLP
            fuser_hidden_size: hidden size of the fusion MLP
            gnn_type: GNN layer type identifier set by subclasses
            n_classes: number of classes; None for binary classification
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size, n_classes)

        self.gnn_n_layers = gnn_n_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_type = gnn_type

        # Create a list of gnns, a GNN per modality
        self._gnn_embedder = self.create_gnn_embedder(gnn_hidden_size, gnn_n_layers)

        # Create MLP that fuses all modalities
        self._mlp_fuser = self.create_mlp_fuser(n_inputs=len(self._embedded_sizes) * gnn_hidden_size,
                                                mlp_hidden_size=fuser_hidden_size,#int(sum(self._embedded_sizes) * gnn_hidden_size / fuser_hidden_size),
                                                mlp_layers=fuser_n_layers).to(self._device)


    def create_gnn_embedder(self, gnn_hidden_size: int = 2, gnn_layers: int = 1):
        """
        Builds one stack of `gnn_layers` graph convolution layers per modality.
        Args:
            gnn_hidden_size: hidden (and output) size of every GNN layer
            gnn_layers: number of GNN layers per modality
        Returns:
            an nn.ModuleList of per-modality nn.ModuleList of GNN layers
        """
        gnn_embedder = []
        for embed_size in self._embedded_sizes:
            layers = [self.build_conv_layer(embed_size, gnn_hidden_size)]
            layers += [self.build_conv_layer(gnn_hidden_size, gnn_hidden_size) for _ in
                       range(gnn_layers - 1)]
            gnn_embedder.append(nn.ModuleList(layers))

        return nn.ModuleList(gnn_embedder).to(self._device)

    @abstractmethod
    def build_conv_layer(self, input_size: int, output_size: int) -> nn.Module:
        """
        Args:
            input_size: size of the input features
            output_size: size of the output features
        Returns:
            a single graph convolution layer, to be defined by subclasses
        """
        pass

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
        # Validate that the List x is of the same size as temporal_analysis
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        # Transform each modality using the corresponding embedder
        output = []
        for i, x_modality in enumerate(x):
            h = self.transform_modality(x_modality, i, lengths)
            # Pass through the GNN embedder for each modality
            for n_layer, gnn_layer in enumerate(self._gnn_embedder[i]):
                h = gnn_layer(g.to(self._device), h) if not isinstance(g, list) else gnn_layer(g[n_layer].to(self._device), h)
                h = F.relu(h)

                if n_layer != self.gnn_n_layers - 1:  # dropout only for intermediate layers
                    h = F.dropout(h, p=self._dropout)

            output.append(h)

        # After processing all modalities, concatenate and pass through MLP to obtain logits
        fused_input = torch.cat(output, dim=1)
        transformed_input = self._mlp_fuser(fused_input).squeeze()

        # free memory
        del fused_input, output, x, gnn_layer, h, x_modality

        return transformed_input


class SGNNFuser(GNNFuser):
    """SGNN model, using GraphSAGE convolutions."""

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
                 aggregator: Literal['mean', 'gcn', 'pool', 'lstm'] = 'mean',
                 n_classes: int = None
                 ):
        """
        Args:
            aggregator: neighbor aggregation function used by each SAGEConv layer
            See GNNFuser.__init__ for the remaining args.
        """
        self.aggregator = aggregator

        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size, gnn_n_layers,
                         gnn_hidden_size, fuser_n_layers, fuser_hidden_size, 'sage', n_classes)

    def build_conv_layer(self, input_size: int, output_size: int) -> nn.Module:
        """
        Args:
            input_size: size of the input features
            output_size: size of the output features
        Returns:
            a SAGEConv layer using `self.aggregator`
        """
        return SAGEConv(input_size, output_size, self.aggregator)
