# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


class AMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(
        self,
        query_type: str,
        query_nb: int,
        num_heads: int,
        size: int,
        dropout: float = 0.1,
    ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(AMultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.query_type = query_type
        self.query_nb = query_nb
        if query_type == "mean":
            self.pooling = nn.AvgPool1d(
                kernel_size=query_nb, stride=1, padding=query_nb // 2
            )
        elif query_type == "attention":
            self.att_layer = nn.Linear(num_heads * head_size, 1)

        # else:
        #     self.pooling = None
        # self.pooling = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # self.pooling = nn.Conv1d(size, size, 3, 1, 1)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        # q aggregation
        if self.query_type == "mean":
            q = self.pooling(q.transpose(1, 2).contiguous()).transpose(1, 2)
        elif self.query_type == "attention":
            q_len = q.shape[1]
            q_att_score = self.att_layer(q).reshape(-1, 1, q_len).repeat(1, q_len, 1)
            q_att_mask = q.new_ones((q_len, q_len), dtype=torch.bool)
            for i in range(q_len):
                start = max(0, i - self.query_nb // 2)
                end = min(q_len, i + self.query_nb // 2 + 1)
                q_att_mask[i, start:end] = False
            q_att_score = q_att_score.masked_fill(q_att_mask, float("-inf"))
            q_att_score = self.softmax(q_att_score)
            # batch x query_len x query_len , batch x query_len x feautre_len --> batch x query_len x feature_len
            q = torch.matmul(q_att_score, q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
class DeformableMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(
        self,
        query_type: str,
        query_nb: int,
        num_heads: int,
        size: int,
        dropout: float = 0.1,
        num_keys: int = 5,
    ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        :param num_keys: The number of observed keys
        """
        super(DeformableMultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.num_keys = num_keys
        self.sample_offsets = nn.Linear(size, num_heads * self.num_keys)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.query_type = query_type
        self.query_nb = query_nb
        if query_type == "mean":
            self.pooling = nn.AvgPool1d(
                kernel_size=query_nb, stride=1, padding=query_nb // 2
            )
        elif query_type == "attention":
            self.att_layer = nn.Linear(num_heads * head_size, 1)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        key_len = k.size(1)
        query_len = q.size(1)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # q aggregation
        if self.query_type == "mean":
            q = self.pooling(q.transpose(1, 2).contiguous()).transpose(1, 2)
        elif self.query_type == "attention":
            q_len = q.shape[1]
            q_att_score = self.att_layer(q).reshape(-1, 1, q_len).repeat(1, q_len, 1)
            q_att_mask = q.new_ones((q_len, q_len), dtype=torch.bool)
            for i in range(q_len):
                start = max(0, i - self.query_nb // 2)
                end = min(q_len, i + self.query_nb // 2 + 1)
                q_att_mask[i, start:end] = False
            q_att_score = q_att_score.masked_fill(q_att_mask, float("-inf"))
            q_att_score = self.softmax(q_att_score)
            # batch x query_len x query_len , batch x query_len x feautre_len --> batch x query_len x feature_len
            q = torch.matmul(q_att_score, q)

        # batch x num_heads x query_len x num_keys
        offsets = (
            self.sample_offsets(q)
            .reshape(batch_size, -1, num_heads, self.num_keys)
            .transpose(1, 2)
        )

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        location_point = torch.arange(
            0, query_len, device=offsets.device, dtype=torch.float32
        ).unsqueeze(-1)

        left = -self.num_keys // 2
        right = self.num_keys + left
        reference_point = torch.arange(
            left, right, device=offsets.device, dtype=torch.float32
        )
        sign_lens = mask.sum(-1).squeeze().float() - 1
        sampling_locations = reference_point + offsets + location_point
        sampling_locations = sampling_locations % sign_lens[:, None, None, None]
        # batch x num_heads x query_len x num_keys
        sampling_locations = sampling_locations / (key_len - 1) * 2 - 1
        y_location = sampling_locations.new_ones(sampling_locations.shape)
        # batch*num_heads x query_len x num_keys x 2
        sampling_locations = torch.stack([sampling_locations, y_location], -1).reshape(
            batch_size * num_heads, query_len, self.num_keys, 2
        )

        # batch*num_heads x head_size x 1 x key_len
        reshaped_k = k.reshape(batch_size * num_heads, 1, -1, self.head_size).permute(
            0, 3, 1, 2
        )
        reshaped_v = v.reshape(batch_size * num_heads, 1, -1, self.head_size).permute(
            0, 3, 1, 2
        )

        # batch x num_heads x query_len x num_keys x head_size
        smaple_k = (
            nn.functional.grid_sample(
                input=reshaped_k,
                grid=sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .reshape(batch_size, num_heads, self.head_size, query_len, -1)
            .permute(0, 1, 3, 4, 2)
        )
        smaple_v = (
            nn.functional.grid_sample(
                input=reshaped_v,
                grid=sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .reshape(batch_size, num_heads, self.head_size, query_len, -1)
            .permute(0, 1, 3, 4, 2)
        )

        # compute scores
        q = q / math.sqrt(self.head_size)
        q = q.unsqueeze(3).expand(
            batch_size, num_heads, query_len, self.num_keys, self.head_size
        )
        # batch x num_heads x query_len x num_keys
        scores = (q * smaple_k).sum(-1)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        # if mask is not None:
        #     scores = scores.masked_fill(~mask, float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]

        # batch x num_heads x query_len x head_size
        context = (attention.unsqueeze(-1) * smaple_v).sum(-2)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, num_heads * self.head_size
        )
        output = self.output_layer(context)

        return output


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class DeformableTransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self,
        query_type: str,
        query_nb: int,
        attentions_type: str,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        num_keys: int = 5,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(DeformableTransformerEncoderLayer, self).__init__()
        self.attention_type = attentions_type
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)

        if attentions_type == "weighted_local_global":
            self.df_src_src_att = DeformableMultiHeadedAttention(
                query_type,
                query_nb,
                num_heads,
                size,
                dropout=dropout,
                num_keys=num_keys,
            )
            self.src_src_att = AMultiHeadedAttention(
                query_type, query_nb, num_heads, size, dropout=dropout
            )
            self.weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.weight.data.fill_(0.5)
            self.fusion_layer_norm = nn.LayerNorm(size, eps=1e-6)
        elif attentions_type == "local":
            self.src_src_att = DeformableMultiHeadedAttention(
                query_type,
                query_nb,
                num_heads,
                size,
                dropout=dropout,
                num_keys=num_keys,
            )
        elif attentions_type == "global":
            self.src_src_att = AMultiHeadedAttention(
                query_type, query_nb, num_heads, size, dropout=dropout
            )

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        if self.attention_type == "weighted_local_global":
            h_g = self.src_src_att(x_norm, x_norm, x_norm, mask)
            h_d = self.df_src_src_att(x_norm, x_norm, x_norm, mask)
            h_g = self.dropout(h_g) + x
            h_d = self.dropout(h_d) + x
            h = self.fusion_layer_norm(h_g * self.weight + h_d * (1 - self.weight))
        elif self.attention_type == "local":
            h = self.src_src_att(x_norm, x_norm, x_norm, mask)
            h = self.dropout(h) + x
        elif self.attention_type == "global":
            h = self.src_src_att(x_norm, x_norm, x_norm, mask)
            h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o


class CrossTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(CrossTransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.corss_src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.cross_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        memory2: Tensor = None,
        src_mask: Tensor = None,
        src_mask2: Tensor = None,
        trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        tgt = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        tgt = self.dropout(tgt) + x

        # source-target attention
        tgt_norm = self.dec_layer_norm(tgt)
        tgt_2 = self.src_trg_att(memory, memory, tgt_norm, mask=src_mask)
        tgt = self.dropout(tgt_2) + tgt
        tgt_norm = self.cross_layer_norm(tgt)
        tgt_2 = self.corss_src_trg_att(memory2, memory2, tgt_norm, mask=src_mask2)
        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(tgt_2) + tgt)

        return o
