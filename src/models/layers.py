"""
Filename: layers.py

Author: Hakima Laribi

Description: Reusable transformer building blocks for multimodal fusers: multi-head
self-attention over per-modality projections (heterogeneous or shared input sizes), a
position-wise feed-forward layer, and an output head combining an MLP with an attention
embedding.

code adapted to our framework from :  https://github.com/SsGood/MMGL/blob/main/MMGL_inductive/layers.py
"""
import torch
from torch import nn
from torch.nn import functional as F


class HeterogeneousTransformer(nn.Module):
    """
    Multi-head self-attention over modalities with heterogeneous input sizes: each modality has
    its own query/key/value projection, so unlike HomogeneousTransformer the projections cannot
    be computed in a single batched linear layer.
    """

    def __init__(self, num_att_heads, device, self_attention_fn, n_modalities, input_shape,
                 hidden_size, dropout):
        """
        Args:
            num_att_heads: number of attention heads
            device: device to run the model on
            self_attention_fn: callable(q, k, v, scale) -> (attention_output, attention_weights)
            n_modalities: number of modalities (one query/key/value projection per modality)
            input_shape: per-modality input feature size
            hidden_size: per-head hidden size
            dropout: dropout probability applied after the output projection
        """
        super(HeterogeneousTransformer, self).__init__()
        # Save attributes
        self._device = device
        self.n_att_heads = num_att_heads
        self.hidden_size = hidden_size
        self.self_attention_fn = self_attention_fn
        self.dropout = dropout

        # Create the linear layers for query, key, and value projections for each modality
        self.w_q, self.w_k, self.w_v = [], [], []
        for i in range(n_modalities):
            self.w_q.append(torch.nn.Linear(input_shape[i], hidden_size * num_att_heads, bias=False).to(self._device))
            self.w_k.append(torch.nn.Linear(input_shape[i], hidden_size * num_att_heads, bias=False).to(self._device))
            self.w_v.append(torch.nn.Linear(input_shape[i], hidden_size * num_att_heads, bias=False).to(self._device))

        self.w_q = torch.nn.ModuleList(self.w_q)
        self.w_k = torch.nn.ModuleList(self.w_k)
        self.w_v = torch.nn.ModuleList(self.w_v)

        # Create fully connected layer (feed-forward layer) to concatenate attention heads
        self.fc = torch.nn.Linear(self.hidden_size * num_att_heads, self.hidden_size * num_att_heads, bias=False).to(
            self._device)

        self.layer_norm = torch.nn.LayerNorm(self.hidden_size * num_att_heads).to(self._device)

    def forward(self, x):
        """
        Args:
            x: list of per-modality input tensors, one per modality, each shape (batch_size, input_shape[i])
        """
        # Get dimensions
        bs = x[0].shape[0]  # batch size
        n_mod = len(x)  # number of modalities

        # Initialize projections, size (batch_size, num_modalities, hidden_size * num_att_heads)
        q = torch.zeros(bs, n_mod, self.hidden_size * self.n_att_heads).to(self._device)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)

        # Compute the multi-head attention for each modality
        for i, x_modality in enumerate(x):
            # Linear projections
            q[:, i, :] = self.w_q[i](x_modality)  # size (batch_size, hidden_size * num_att_heads)
            k[:, i, :] = self.w_k[i](x_modality)
            v[:, i, :] = self.w_v[i](x_modality)

        # Reshape projections to (batch_size, num_att_heads, num_modalities, hidden_size)
        q = q.view(x[0].shape[0], len(x), self.n_att_heads, self.hidden_size).transpose(1, 2)
        k = k.view(x[0].shape[0], len(x), self.n_att_heads, self.hidden_size).transpose(1, 2)
        v = v.view(x[0].shape[0], len(x), self.n_att_heads, self.hidden_size).transpose(1, 2)

        # Compute self-attention : P and P*V
        att_output, att_weights = self.self_attention_fn(q, k, v, self.hidden_size ** 0.5)

        # Reshape outputs : (bs, num_modalities, num_att_heads * hidden_size)
        att_output = att_output.transpose(1, 2).contiguous().view(bs, n_mod, -1)
        v = v.transpose(1, 2).contiguous().view(bs, n_mod, -1)

        # The FC merging the heads keeps the same shape
        att_output = F.dropout(self.fc(att_output), p=self.dropout,
                               training=self.training)  # (bs, n_mod, head * hidden)

        # Addition of : (I + P) * V
        output = att_output + v

        # Layer normalization
        output = self.layer_norm(output)

        return output, att_weights


class HomogeneousTransformer(nn.Module):
    """
    Multi-head self-attention over modalities sharing the same input size: a single query/key/
    value projection is applied to all modalities at once, unlike HeterogeneousTransformer.
    """

    def __init__(self, num_att_heads, device, self_attention_fn, n_modalities, input_shape, hidden_size,
                 dropout):
        """
        Args:
            num_att_heads: number of attention heads
            device: device to run the model on
            self_attention_fn: callable(q, k, v, scale) -> (attention_output, attention_weights)
            n_modalities: number of modalities
            input_shape: shared input feature size across modalities
            hidden_size: per-head hidden size
            dropout: dropout probability applied after the output projection
        """
        super(HomogeneousTransformer, self).__init__()
        # Save attributes
        self._device = device
        self.n_att_heads = num_att_heads
        self.hidden_size = hidden_size
        self.self_attention_fn = self_attention_fn
        self.n_mod = n_modalities
        self.dropout = dropout

        # Create the linear layers for query, key, and value projections
        self.w_q = torch.nn.Linear(input_shape, hidden_size * num_att_heads, bias=False).to(self._device)
        self.w_k = torch.nn.Linear(input_shape, hidden_size * num_att_heads, bias=False).to(self._device)
        self.w_v = torch.nn.Linear(input_shape, hidden_size * num_att_heads, bias=False).to(self._device)

        # Create fully connected layer (feed-forward layer) to concatenate attention heads
        self.fc = torch.nn.Linear(self.hidden_size * num_att_heads, self.hidden_size * num_att_heads, bias=False).to(
            self._device)

        self.layer_norm = torch.nn.LayerNorm(self.hidden_size * num_att_heads).to(self._device)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (batch_size, n_modalities, input_shape)
        Returns:
            output, att_weights: the attention output (batch_size, n_modalities, hidden_size *
            num_att_heads), projected through `self.fc`, added to `v` (residual-style) and
            layer-normalized; and the raw attention weights from `self_attention_fn`
        """
        # Get dimensions
        bs = x.shape[0]  # batch size

        # Compute the multi-head attention
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape projections to (batch_size, num_att_heads, num_modalities, hidden_size)
        q = q.view(bs, self.n_mod, self.n_att_heads, self.hidden_size).transpose(1, 2)
        k = k.view(bs, self.n_mod, self.n_att_heads, self.hidden_size).transpose(1, 2)
        v = v.view(bs, self.n_mod, self.n_att_heads, self.hidden_size).transpose(1, 2)

        # Compute self-attention : P and P*V
        att_output, att_weights = self.self_attention_fn(q, k, v, self.hidden_size ** 0.5)

        # Reshape outputs : (bs, num_modalities, num_att_heads * hidden_size)
        att_output = att_output.transpose(1, 2).contiguous().view(bs, self.n_mod, -1)
        v = v.transpose(1, 2).contiguous().view(bs, self.n_mod, -1)

        # The FC merging the heads keeps the same shape
        att_output = F.dropout(self.fc(att_output), p=self.dropout,
                               training=self.training)  # (bs, n_mod, head * hidden)

        # Addition of : (I + P) * V
        output = att_output + v

        # Layer normalization
        output = self.layer_norm(output)

        return output, att_weights


class FeedForwardLayer(nn.Module):
    """Position-wise feed-forward sublayer (Linear-GELU-Linear) with a residual connection."""

    def __init__(self, input_shape, hidden_size, dropout=0.1):
        """
        Args:
            input_shape: input (and output) feature size
            hidden_size: hidden size of the intermediate layer
            dropout: dropout probability applied before the residual connection
        """
        super().__init__()
        self.w_1 = nn.Linear(input_shape, hidden_size)  # position-wise
        self.w_2 = nn.Linear(hidden_size, input_shape)  # position-wise
        self.layer_norm = nn.LayerNorm(input_shape, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: input tensor, shape (..., input_shape)
        Returns:
            x transformed by the feed-forward sublayer, residual connection and layer norm
        """
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))  # FFN in transformer block
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class OutputLayer(nn.Module):
    """MLP head over shared/private representations, concatenated and classified with a linear layer."""

    def __init__(self, d_in, d_hidden, modal_num, dropout, n_classes):
        """
        Args:
            d_in: input feature size of the MLP head
            d_hidden: hidden size of the MLP head
            modal_num: number of modalities; the classifier expects `modal_num ** 2` extra
                features from `attn_embedding` (e.g. a flattened attention/adjacency matrix)
            dropout: dropout probability in the MLP head
            n_classes: number of output classes/logits
        """
        super(OutputLayer, self).__init__()
        # MLP head used to predict F(sp)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_hidden + modal_num ** 2, n_classes)

    def forward(self, x, attn_embedding):
        """
        Args:
            x: shared representation, shape (..., d_in)
            attn_embedding: extra features concatenated to the MLP output, shape (..., modal_num ** 2)
        Returns:
            output, combined_x: classification logits, and the concatenated features fed to the
            classifier
        """
        x = self.mlp_head(x)  # MLP(H_sh)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x).squeeze()  # logits : FC(MLP(H_sh) || H_sp)
        return output, combined_x