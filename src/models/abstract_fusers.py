"""
Filename: abstract_fusers.py

Author: Hakima Laribi

Description: Base classes for multimodal fusion models. Fuser defines the common constructor
contract (per-modality input sizes and temporal/static flags). PostTemporalFuser adds the
shared per-modality embedding step (a bidirectional GRU for temporal modalities, identity for
static ones) and MLP-building helpers used by concrete fusers to combine embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Fuser(nn.Module, ABC):
    """Abstract base class for fusion models: validates and stores shared modality config."""

    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 **kwargs
                 ):
        """
        Args:
            temporal_analysis: per-modality flag, True if the modality has temporal sequences
            input_sizes: per-modality input feature size
            dropout: dropout probability used by subclasses
            device: device to run the model on (GPU, CPU)
            temporal_hidden_size: hidden size of the GRU layers used for temporal modalities
        """
        super().__init__()

        if len(input_sizes) != len(temporal_analysis):
            raise ValueError(f'input_sizes {len(input_sizes)} and temporal_analysis {len(temporal_analysis)} must be '
                             f'of the same sizes (number of modalities)')

        self._temporal_analysis = temporal_analysis
        self._input_sizes = input_sizes
        self._dropout = dropout
        self._device = device
        self._temporal_hidden_size = temporal_hidden_size

    def create_mlp_fuser(self,
                         n_inputs: int,
                         mlp_hidden_size: int = 128,
                         mlp_layers: int = 1):
        """
        Builds an MLP mapping the fused representation of `n_inputs` modalities to logits
        (1, or n_classes if set). Requires `self.n_classes` to be set by the concrete
        subclass before being called.
        Args:
            n_inputs: number of modalities being fused
            mlp_hidden_size: hidden size of the first MLP layer
            mlp_layers: number of hidden layers, see create_mlp_hidden_layers
        Returns:
            an nn.Sequential mapping the fused modality representations to logits
        """
        layers = [nn.Linear(n_inputs, mlp_hidden_size), nn.ReLU(), nn.BatchNorm1d(mlp_hidden_size)]
        layers += self.create_mlp_hidden_layers(mlp_layers, mlp_hidden_size, output_size=64)
        layers.append(nn.Linear(64, 1 if self.n_classes is None else self.n_classes))

        return nn.Sequential(*layers)

    def create_mlp_hidden_layers(self,
                                 mlp_layers: int,
                                 mlp_hidden_size: int = 128,
                                 output_size: int = 64):
        """
        Builds `mlp_layers` hidden layers, narrowing to `output_size` on the last one.
        Args:
            mlp_layers: number of hidden layers to build
            mlp_hidden_size: hidden size of all but the last layer
            output_size: output size of the last layer
        Returns:
            a list of nn.Module layers (Linear, ReLU/BatchNorm1d, Dropout)
        """
        layers = []
        for i in range(mlp_layers):
            if i == mlp_layers - 1:
                layers.extend([nn.Linear(mlp_hidden_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()])
            else:
                layers.extend([nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU(), nn.Dropout(self._dropout)])

        return layers

    @abstractmethod
    def forward(self, x: List[Union[torch.Tensor, List[torch.Tensor]]], lengths: Optional[List[int]] = None):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            the fused prediction (logits), to be defined by subclasses
        """
        pass


class PostTemporalFuser(Fuser, ABC):
    """
    Abstract class for fusers that process modalities (via MLP, GNN or others) after temporal analysis (via GRU).
    """
    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 n_classes: int = None,
                 **kwargs
                 ):
        """
        Args:
            n_classes: number of classes; None for binary classification
            See Fuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size)
        self.n_classes = n_classes
        # one embedder per modality: a bidirectional GRU for temporal ones, identity for static ones
        self._embedder = nn.ModuleList([nn.Identity() if not is_temporal
                                        else nn.GRU(input_size=input_sizes[i],
                                                    hidden_size=temporal_hidden_size,
                                                    num_layers=1,
                                                    bidirectional=True, batch_first=True)
                                        for i, is_temporal in enumerate(temporal_analysis)]).to(self._device)

        self._embedded_sizes = [temporal_hidden_size * 2 if is_temporal else input_sizes[i]
                                for i, is_temporal in enumerate(temporal_analysis)]

    def transform_modality(self,
                           x_modality: Union[torch.Tensor, List[torch.Tensor]],
                           index_modality: int,
                           lengths: Optional[List[int]] = None):
        """
        Embeds one modality: for a temporal modality, packs the (padded) sequences and returns
        the concatenated last hidden state of the bidirectional GRU; for a static modality,
        returns it unchanged (identity).
        Args:
            x_modality: input tensor (or list of tensors) for this modality
            index_modality: index of this modality in temporal_analysis/input_sizes
            lengths: per-modality original sequence lengths, required if this modality is temporal
        Returns:
            the embedded modality, shape (N, D)
        """
        if self._temporal_analysis[index_modality]:
            if isinstance(x_modality, list):
                pad_input = torch.stack(x_modality).to(self._device)
            else:
                pad_input = x_modality.to(self._device)

            # pack padded sequences to avoid computation on zero-padded steps,
            # see: https://stackoverflow.com/posts/56211056/revisions
            packed_input = pack_padded_sequence(pad_input, lengths[index_modality],
                                                batch_first=True,
                                                enforce_sorted=False)

            h_per_seq, last_h = self._embedder[index_modality](packed_input)
            h_t = torch.cat((last_h[-2], last_h[-1]), dim=1)  # concat h_last of each GRU direction
            output = h_t  # of size (N, D)
            del pad_input, packed_input, h_per_seq, last_h

        else:
            if isinstance(x_modality, list):
                x_modality = torch.stack(x_modality).view(len(x_modality), -1)
            output = self._embedder[index_modality](x_modality.to(self._device))  # of size (N, D)
        del x_modality
        torch.cuda.empty_cache()
        return output

    @abstractmethod
    def forward(self, x: List[Union[torch.Tensor, List[torch.Tensor]]], lengths: Optional[List[int]] = None):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            the fused prediction (logits), to be defined by subclasses
        """
        pass