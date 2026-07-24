import math
from collections.abc import Callable

import torch
import torch.nn as nn

class InputEmbeddingBlock(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncodingBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2,)

        # position: (seq_len, 1)
        # div_term: (d_model/2,)
        # position*div_term brroadcasts to (seq_len, d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNormalizationBlock(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)

        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        return (self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        d_k = q.shape[-1]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min
            )

        attn = torch.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        return attn @ v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        batch_size = query.shape[0]

        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, self.d_model)


        return self.w_o(x)


class ResidualConnectionBlock(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizationBlock()

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attn: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.residuals = nn.ModuleList([
            ResidualConnectionBlock(dropout),
            ResidualConnectionBlock(dropout)
        ])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residuals[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attn: MultiHeadAttentionBlock, cross_attn: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()

        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward

        self.residuals = nn.ModuleList([
            ResidualConnectionBlock(dropout),
            ResidualConnectionBlock(dropout),
            ResidualConnectionBlock(dropout)
        ])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock()

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(
                self, 
                encoder: Encoder,
                decoder: Decoder,
                src_embed: InputEmbeddingBlock,
                tgt_embed: InputEmbeddingBlock,
                src_pos: PositionalEncodingBlock,
                tgt_pos: PositionalEncodingBlock,
                proj: ProjectionLayer):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(
                self,
                src: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:

        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
                self,
                tgt: torch.Tensor,
                enc_out: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 256,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 1024
) -> Transformer:

    src_embed = InputEmbeddingBlock(d_model, src_vocab_size)
    tgt_embed = InputEmbeddingBlock(d_model, tgt_vocab_size)

    src_pos = PositionalEncodingBlock(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodingBlock(d_model, tgt_seq_len, dropout)

    enc_layers: list[EncoderBlock] = []
    dec_layers: list[DecoderBlock] = []

    for _ in range(N):
        enc_layers.append(
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            )
        )

        dec_layers.append(
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            )
        )

    encoder = Encoder(nn.ModuleList(enc_layers))
    decoder = Decoder(nn.ModuleList(dec_layers))

    proj = ProjectionLayer(d_model, tgt_vocab_size)

    model = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        proj
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model