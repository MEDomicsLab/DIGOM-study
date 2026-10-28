"""
Filename: baselines.py

Author: Hakima Laribi

Description: Conventional fusion baselines. EarlyFuser concatenates the modalities and passes them
through an MLP; JointFuser projects each modality with its own MLP, then concatenates and fuses them with a second
MLP; LateFuser predicts from each modality independently and combines the predictions with a fully connected layer.
"""
from typing import List, Literal, Optional, Union

import torch
from torch import nn

from src.models.abstract_fusers import PostTemporalFuser


class LateFuser(PostTemporalFuser):
    """Classifies each modality independently, then aggregates the per-modality predictions."""

    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 unimodal_n_layers: int = 1,
                 unimodal_hidden_size: int = 2,
                 n_classes: int = None
                 ):
        """
        Args:
            unimodal_n_layers: number of hidden layers in each per-modality classifier MLP
            unimodal_hidden_size: hidden size of each per-modality classifier MLP
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size, n_classes)

        self._unimodal_mlp = self.create_unimodal_mlp(unimodal_hidden_size, unimodal_n_layers)

        # linear layer that aggregates the per-modality predictions into the final prediction
        if self.n_classes is None:
            self._pred_aggregator = nn.Linear(len(input_sizes), 1).to(self._device)
        else:
            self._pred_aggregator = nn.Linear(len(input_sizes) * self.n_classes, self.n_classes).to(self._device)

    def create_unimodal_mlp(self, mlp_hidden_size: int = 2, mlp_layers: int = 1):
        """
        Builds one classifier MLP (ending in a probability, sigmoid/softmax) per modality.
        Args:
            mlp_hidden_size: hidden size of the first layer of each MLP
            mlp_layers: number of hidden layers in each MLP, see create_mlp_hidden_layers
        Returns:
            an nn.ModuleList with one classifier MLP per modality
        """
        mlp_embedder = []
        for embed_size in self._embedded_sizes:
            layers = [nn.Linear(embed_size, mlp_hidden_size), nn.ReLU(), nn.BatchNorm1d(mlp_hidden_size)]
            layers += self.create_mlp_hidden_layers(mlp_layers, mlp_hidden_size, output_size=64)

            if self.n_classes is None:
                layers.append(nn.Linear(64, 1))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(64, self.n_classes))
                layers.append(nn.Softmax(dim=1))

            mlp_embedder.append(nn.Sequential(*layers))

        return nn.ModuleList(mlp_embedder).to(self._device)

    def forward(self,
                  x: List[Union[torch.Tensor, List[torch.Tensor]]],
                  lengths: Optional[List[int]] = None):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            the fused logits, obtained from a linear combination of the per-modality
            probabilities (callers apply sigmoid/softmax on top, as with the other fusers)
        """
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        output = []
        for i, x_modality in enumerate(x):
            h = self.transform_modality(x_modality, i, lengths)
            output.append(self._unimodal_mlp[i](h))

        # aggregate the per-modality probabilities into the final prediction
        fused_proba = torch.cat(output, dim=1)
        multimodal_proba = self._pred_aggregator(fused_proba).squeeze()

        del fused_proba, output, x
        torch.cuda.empty_cache()

        return multimodal_proba


class JointFuser(PostTemporalFuser):
    """Embeds each modality through its own MLP, then concatenates and fuses them with an MLP."""

    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 unimodal_n_layers: int = 1,
                 unimodal_hidden_size: int = 2,
                 fuser_n_layers: int = 1,
                 fuser_hidden_size: int = 2,
                 n_classes: int = None
                 ):
        """
        Args:
            unimodal_n_layers: number of hidden layers in each per-modality embedding MLP
            unimodal_hidden_size: hidden size of each per-modality embedding MLP
            fuser_n_layers: number of hidden layers in the fusion MLP
            fuser_hidden_size: hidden size of the fusion MLP
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis, input_sizes, dropout, device, temporal_hidden_size, n_classes)

        self._mlp_embedder = self.create_mlp_embedder(unimodal_hidden_size, unimodal_n_layers)
        self._mlp_fuser = self.create_mlp_fuser(n_inputs=sum(self._embedded_sizes),
                                                mlp_hidden_size=fuser_hidden_size,
                                                mlp_layers=fuser_n_layers).to(self._device)

    def create_mlp_embedder(self, mlp_hidden_size: int = 2, mlp_layers: int = 1):
        """
        Builds one embedding MLP per modality (no final activation, unlike LateFuser's).
        Args:
            mlp_hidden_size: hidden size of the first layer of each MLP
            mlp_layers: number of hidden layers in each MLP, see create_mlp_hidden_layers
        Returns:
            an nn.ModuleList with one embedding MLP per modality
        """
        mlp_embedder = []
        for embed_size in self._embedded_sizes:
            layers = [nn.Linear(embed_size, mlp_hidden_size), nn.ReLU(), nn.BatchNorm1d(mlp_hidden_size)]
            layers += self.create_mlp_hidden_layers(mlp_layers, mlp_hidden_size, output_size=embed_size)
            mlp_embedder.append(nn.Sequential(*layers))

        return nn.ModuleList(mlp_embedder).to(self._device)

    def forward(self,
                x: List[Union[torch.Tensor, List[torch.Tensor]]],
                lengths: Optional[List[int]] = None):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            the fused logits
        """
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        output = []
        for i, x_modality in enumerate(x):
            h = self.transform_modality(x_modality, i, lengths)
            output.append(self._mlp_embedder[i](h))

        fused_input = torch.cat(output, dim=1)
        transformed_input = self._mlp_fuser(fused_input).squeeze()

        del fused_input, output, x
        torch.cuda.empty_cache()

        return transformed_input


class EarlyFuser(PostTemporalFuser):
    """Concatenates the raw per-modality embeddings directly and fuses them with a single MLP."""

    def __init__(self,
                 temporal_analysis: List[bool],
                 input_sizes: List[int],
                 dropout: float,
                 device: Literal['cpu', 'cuda'],
                 temporal_hidden_size: int = 256,
                 fuser_n_layers: int = 1,
                 fuser_hidden_size: int = 2,
                 n_classes: int = None
                 ):
        """
        Args:
            fuser_n_layers: number of hidden layers in the fusion MLP
            fuser_hidden_size: hidden size of the fusion MLP
            See PostTemporalFuser.__init__ for the remaining args.
        """
        super().__init__(temporal_analysis,
                         input_sizes,
                         dropout,
                         device,
                         temporal_hidden_size,
                         n_classes)

        self._mlp = self.create_mlp_fuser(n_inputs=sum(self._embedded_sizes),
                                          mlp_hidden_size=fuser_hidden_size,
                                          mlp_layers=fuser_n_layers).to(self._device)

    def forward(self,
                x: List[Union[torch.Tensor, List[torch.Tensor]]],
                lengths: Optional[List[int]] = None):
        """
        Args:
            x: per-modality input tensors (or lists of tensors for temporal modalities)
            lengths: per-modality original sequence lengths, required for temporal modalities
        Returns:
            the fused logits
        """
        if len(x) != len(self._temporal_analysis):
            raise ValueError('Number of modalities does not match temporal analysis configuration')

        output = [self.transform_modality(x_modality, i, lengths) for i, x_modality in enumerate(x)]

        fused_input = torch.cat(output, dim=1)
        transformed_input = self._mlp(fused_input).squeeze()

        del fused_input, output, x
        torch.cuda.empty_cache()

        return transformed_input
