"""
Filename: graph_dataset.py

Author: Hakima Laribi

Description: Graph-based counterparts of MMDataset/MMTuberculosisDataset. GraphDatasetMixin
builds one similarity graph per split (train/valid/test): edges connect each sample to its
top_percent most similar neighbors (Gaussian kernel over the Euclidean distance between selected
"distance modalities"), with validation and test nodes only able to receive from train/valid.
"""
from abc import ABC
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Union

import dgl
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing.dataset import MMDataset, MMTuberculosisDataset


class GraphDatasetMixin(ABC):
    """Mixin class with methods for building similarity-graphs for multi-modal dataset."""

    @property
    def train_graph(self):
        """
        Returns:
            the training graph
        """
        return self._train_graph

    @property
    def valid_graph(self):
        """
        Returns:
            the validation graph
        """
        return self._valid_graph

    @property
    def test_graph(self):
        """
        Returns:
            the test graph
        """
        return self._test_graph

    def set_masks(self,
                  train_ids: List[int],
                  valid_ids: List[int],
                  test_ids: List[int],
                  top_percent: float = None,
                  distance_modalities: List[str] = None):
        """
        Builds the train/valid/test similarity graphs
        Args:
            train_ids: ids of the training samples
            valid_ids: ids of the validation samples
            test_ids: ids of the test samples
            top_percent: fraction of highest-similarity edges to keep per node
            distance_modalities: modalities whose features are used to compute similarity
        """
        super().set_masks(train_ids, valid_ids, test_ids)

        # Enumerate the nodes, train is first because always present, then valid because it is used to propagate to test
        self._node_ids = {sample_id: i for i, sample_id in enumerate(train_ids + valid_ids + test_ids)}

        # Build graph for each mask, this includes self-loops
        for mask_type in ['train', 'valid', 'test']:
            # Set the node ids according to the mask type and their initial order
            if mask_type == 'train':
                step = 0
            elif mask_type == 'valid':
                step = len(self._train_indexes)
            elif mask_type == 'test':
                step = len(self._train_indexes) + len(self._valid_indexes)
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")

            x1_edge, x2_edge = self.get_edge_features(mask_type=mask_type, distance_modalities=distance_modalities)

            # Compute the Euclidean distance between edge features of the nodes
            distances = self.compute_euclidian_distance(x1_edge, x2_edge)

            # Apply Gaussian kernel
            sigma = distances.std()
            distances.pow_(2).div_(2 * sigma ** 2).neg_().exp_()
            sim = distances

            # find the threshold for the top percent of similarities
            top_k = int(top_percent * sim.shape[1])  # top X percent of similiarities

            # Get top-k values and their indices
            top_values, top_indices = torch.topk(sim, k=top_k)
            del sim, distances  # free memory immediately

            n = x1_edge.shape[0]
            rows = torch.arange(n)

            # Vectorized edge building
            src = top_indices.flatten()  # (n * top_k,)
            dst = (rows + step).repeat_interleave(top_k)  # (n * top_k,)
            weight = top_values.flatten()  # (n * top_k,)

            # Self-loops: only where not already present
            self_nodes = rows + step
            is_self_loop = (top_indices == self_nodes.unsqueeze(1))  # (n, top_k)
            missing_self = ~is_self_loop.any(dim=1)  # (n,)

            if missing_self.any():
                self_src = self_nodes[missing_self]
                src = torch.cat([src, self_src])
                dst = torch.cat([dst, self_src])
                weight = torch.cat([weight, torch.ones(missing_self.sum())])

            edges = {
                'src': src.tolist(),
                'dst': dst.tolist(),
                'weight': weight.tolist()
            }
            self.build_dgl_graph(mask_type, edges)

    def build_dgl_graph(self,
                        mask_type: str,
                        edges: Dict[str, List[Union[float, int]]]):
        """
        Builds (or extends) the DGL graph for one split from the given edges.
        Args:
            mask_type: 'train', 'valid' or 'test'
            edges: {'src': [...], 'dst': [...], 'weight': [...]} edges to add
        """
        if mask_type == 'train':
            # Add edges and their weights to the training graph, nodes include only the training samples
            g = dgl.graph((edges['src'], edges['dst']))
            g.edata['weight'] = torch.tensor(edges['weight'], dtype=torch.float32)
            self._train_graph = g

        elif (mask_type == 'valid') | (mask_type == 'test'):
            # Get the base graph, for testing sample we add the validation nodes to the training graph and propagate
            g = deepcopy(self._train_graph) if mask_type == 'valid' else deepcopy(self._valid_graph)
            # Get the edge weights of the base graph
            base_weight = g.edata['weight']
            # Add edges between nodes of the current mask and all nodes of the base graph
            g.add_edges(edges['src'], edges['dst'])
            # Add the weights of the new edges
            e_weights = torch.tensor(edges['weight'], dtype=torch.float32)
            g.edata['weight'] = torch.cat([base_weight, e_weights], dim=0)
            setattr(self, f'_{mask_type}_graph', g)

    def get_edge_features(self,
                          mask_type: str,
                          distance_modalities: List[str] = None):
        """
        Gathers, for each requested modality, the features of the nodes of `mask_type` (x1) and
        of the nodes they're allowed to receive edges from (x2; see class docstring).
        Args:
            mask_type: 'train', 'valid' or 'test'
            distance_modalities: modalities whose features are used to compute similarity
        Returns:
            x1_edge, x2_edge: concatenated features across `distance_modalities`
        """
        x1_edge, x2_edge = [], []

        for modality in distance_modalities:
            if modality not in self._df_modalities.keys():
                raise ValueError(f"Modality {modality} not found in the dataset.")

            x_modality = self._x[modality]  # Tensor with modality features

            if isinstance(x_modality, list):
                x_modality = torch.stack(x_modality).view(len(x_modality), -1)

            if mask_type == 'train':
                x1_edge.append(x_modality[self._train_indexes])
                x2_edge.append(x_modality[self._train_indexes])

            elif mask_type == 'valid':
                x1_edge.append(x_modality[self._valid_indexes])
                # propagate from both training patients only
                x2_edge.append(x_modality[self._train_indexes])

            elif mask_type == 'test':
                x1_edge.append(x_modality[self._test_indexes])
                # propagate from both train and valid sets to test set
                x2_edge.append(x_modality[self._train_indexes + self._valid_indexes])

        x1_edge = torch.cat(x1_edge, dim=1)
        x2_edge = torch.cat(x2_edge, dim=1)

        return x1_edge, x2_edge

    @staticmethod
    def compute_euclidian_distance(x1, x2, chunk_size=10000):
        """
        Computes pairwise Euclidean distances row-by-row in chunks, to save memory.
        Args:
            x1: tensor, shape (n1, n_features)
            x2: tensor, shape (n2, n_features)
            chunk_size: number of rows of x1 processed at once
        Returns:
            pairwise distance matrix, shape (n1, n2)
        """
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        result = torch.empty(n1, n2, dtype=x1.dtype, device=x1.device)

        x2_norm = (x2 ** 2).sum(dim=1)  # precompute once

        for i in range(0, n1, chunk_size):
            x1_chunk = x1[i: i + chunk_size]
            x1_norm = (x1_chunk ** 2).sum(dim=1).view(-1, 1)
            d_sqrt = x1_norm + x2_norm.view(1, -1) - 2.0 * x1_chunk @ x2.T
            result[i: i + chunk_size] = torch.sqrt(torch.clamp(d_sqrt, min=0.0))

        return result


class GMMDataset(GraphDatasetMixin, MMDataset):
    """Graph-based counterpart of MMDataset, adding per-split similarity graphs."""

    def __init__(self,
                 df_modalities: Dict[str, pd.DataFrame],
                 target_column: pd.DataFrame,
                 temporal_analysis: bool,
                 temporal_modalities: List[bool],
                 to_tensor: bool = True,
                 id_column: str = 'hadm_id',
                 impute_cont_cols: Optional[Literal['mean', 'zero', 'minus_one']] = None,
                 modality_miss: Optional[Dict[str, np.array]] = None):
        """
        Args: see MMDataset.__init__.
        """
        super().__init__(df_modalities=df_modalities,
                         target_column=target_column,
                         temporal_analysis=temporal_analysis,
                         temporal_modalities=temporal_modalities,
                         to_tensor=to_tensor,
                         id_column=id_column,
                         impute_cont_cols=impute_cont_cols,
                         modality_miss=modality_miss)

        self._node_ids = {}
        self._train_graph, self._valid_graph, self._test_graph = None, None, None


class GMMTuberculosisDataset(GraphDatasetMixin, MMTuberculosisDataset):
    """Graph-based counterpart of MMTuberculosisDataset, adding per-split similarity graphs."""

    def __init__(self,
                 df_modalities: Dict[str, np.array],
                 target_column: np.array,
                 mask_imputation: Dict[str, np.array],
                 to_tensor: bool = True,
                 modality_miss=None):
        """
        Args: see MMTuberculosisDataset.__init__.
        """
        super().__init__(df_modalities=df_modalities,
                         mask_imputation=mask_imputation,
                         target_column=target_column,
                         to_tensor=to_tensor,
                         modality_miss=modality_miss)

        self._node_ids = {}
        self._train_graph, self._valid_graph, self._test_graph = None, None, None