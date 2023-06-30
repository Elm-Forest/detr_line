# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MultiScaleFormer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=[2, 2, 2, 2],
                 num_decoder_layers=4, dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_block1 = TransformerEncoder(encoder_layer, num_encoder_layers[0], encoder_norm)
        self.encoder_block2 = TransformerEncoder(encoder_layer, num_encoder_layers[1], encoder_norm)
        self.encoder_block3 = TransformerEncoder(encoder_layer, num_encoder_layers[2], encoder_norm)
        self.encoder_block4 = TransformerEncoder(encoder_layer, num_encoder_layers[3], encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src[0].shape
        src_ = src[0].flatten(2).permute(2, 0, 1)
        pos_ = pos_embed[0].flatten(2).permute(2, 0, 1)
        pos_key = pos_.clonze()
        mask_ = mask[0].flatten(1)
        mask_key = mask_.clone()
        memory1 = self.encoder_block1(value=src_,
                                      query=src_,
                                      src_key_padding_mask=mask_,
                                      pos_key=pos_,
                                      pos_query=pos_key)

        src_ = src[1].flatten(2).permute(2, 0, 1)
        pos_ = pos_embed[1].flatten(2).permute(2, 0, 1)
        mask_ = mask[1].flatten(1)

        memory2 = self.encoder_block2(value=src_,
                                      query=memory1,
                                      src_key_padding_mask=mask_,
                                      pos_key=pos_,
                                      pos_query=pos_key)

        src_ = src[2].flatten(2).permute(2, 0, 1)
        pos_ = pos_embed[2].flatten(2).permute(2, 0, 1)
        mask_ = mask[2].flatten(1)

        memory3 = self.encoder_block3(value=src_,
                                      query=memory2,
                                      src_key_padding_mask=mask_,
                                      pos_key=pos_,
                                      pos_query=pos_key)

        src_ = src[3].flatten(2).permute(2, 0, 1)
        pos_ = pos_embed[3].flatten(2).permute(2, 0, 1)
        mask_ = mask[3].flatten(1)

        memory4 = self.encoder_block4(value=src_,
                                      query=memory3,
                                      src_key_padding_mask=mask_,
                                      pos_key=pos_,
                                      pos_query=pos_key)

        # src = src.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, [memory1, memory2, memory3, memory4],
                          memory_key_padding_mask=mask_key,
                          pos=pos_key,
                          query_pos=query_embed)

        return hs.transpose(1, 2), memory4.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, value=None, query=None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_key: Optional[Tensor] = None,
                pos_query: Optional[Tensor] = None):

        for layer in self.layers[:1]:
            query = layer(value=query, query=query,
                          src_mask=mask, src_key_padding_mask=None,
                          pos_key=pos_query, pos_query=pos_query)
        for layer in self.layers[1:]:
            query = layer(value=value, query=query,
                          src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                          pos_key=pos_key, pos_query=pos_query)
        if self.norm is not None:
            query = self.norm(query)

        return query


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[List] = None,
                pos: Optional[List] = None,
                query_pos: Optional[List] = None):
        output = tgt

        intermediate = []

        for layer, mem in zip(self.layers[:4], memory):
            output = layer(output, mem, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        for layer in self.layers[4:]:
            output = layer(output, memory[-1], tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, value=None, query=None,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos_key: Optional[Tensor] = None,
                     pos_query: Optional[Tensor] = None):
        src2 = self.self_attn(self.with_pos_embed(query, pos_query),
                              self.with_pos_embed(value, pos_key),
                              value=value,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        query = query + self.dropout1(src2)
        query = self.norm1(query)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(src2)
        query = self.norm2(query)
        return query

    def forward_pre(self, value=None, query=None,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos_key: Optional[Tensor] = None,
                    pos_query: Optional[Tensor] = None):
        query = self.norm1(query)
        src2 = self.self_attn(self.with_pos_embed(query, pos_query),
                              self.with_pos_embed(value, pos_key),
                              value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        query = query + self.dropout1(src2)
        src2 = self.norm2(query)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        query = query + self.dropout2(src2)
        return query

    def forward(self, value=None, query=None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_key: Optional[Tensor] = None,
                pos_query: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(value, query, src_mask, src_key_padding_mask, pos_key, pos_query)
        return self.forward_post(value, query, src_mask, src_key_padding_mask, pos_key, pos_query)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_multiscaleformer(args):
    return MultiScaleFormer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=[2, 2, 2, 2],
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':
    multiscaleformer = MultiScaleFormer()
    #
    # x1 = torch.rand(1, 256, 64, 64)
    # query = x1.flatten(2).permute(2, 0, 1)
    # x2 = torch.rand(1, 256, 64, 64)
    # value = x2.flatten(2).permute(2, 0, 1)
    # query = multiscaleformer.encoder_block1(value=value,
    #                                         query=query,
    #                                         src_key_padding_mask=None,
    #                                         pos_key=value,
    #                                         pos_query=query)
    # print(query.shape)
    # x2 = torch.rand(1, 256, 32, 32)
    # value = x2.flatten(2).permute(2, 0, 1)
    # query = multiscaleformer.encoder_block1(value=value,
    #                                         query=query,
    #                                         src_key_padding_mask=None,
    #                                         pos_key=value,
    #                                         pos_query=query)
    # print(query.shape)
    # x2 = torch.rand(1, 256, 16, 16)
    # value = x2.flatten(2).permute(2, 0, 1)
    # query = multiscaleformer.encoder_block1(value=value,
    #                                         query=query,
    #                                         src_key_padding_mask=None,
    #                                         pos_key=value,
    #                                         pos_query=query)
    # print(query.shape)
    # x2 = torch.rand(1, 256, 8, 8)
    # value = x2.flatten(2).permute(2, 0, 1)
    # query = multiscaleformer.encoder_block1(value=value,
    #                                         query=query,
    #                                         src_key_padding_mask=None,
    #                                         pos_key=value,
    #                                         pos_query=query)
    # print(query.shape)

    x1 = torch.rand(1, 256, 64, 64)
    m1 = torch.rand(1, 64, 64)
    x2 = torch.rand(1, 256, 32, 32)
    m2 = torch.rand(1, 32, 32)
    x3 = torch.rand(1, 256, 16, 16)
    m3 = torch.rand(1, 16, 16)
    x4 = torch.rand(1, 256, 8, 8)
    m4 = torch.rand(1, 8, 8)
    out = multiscaleformer([x1, x2, x3, x4], [m1, m2, m3, m4], nn.Embedding(10, 256).weight, [x1, x2, x3, x4])[0]
    print(out.shape)
