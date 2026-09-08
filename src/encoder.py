import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .layers import ResidualConnection, FeedForward, InputEmbedding, PositionalEncoding


class EncoderBlock(nn.Module):

    def __init__(self,  dff, d_model, heads, device,dropout):

        super(EncoderBlock, self).__init__()
        self.multi_attention = MultiHeadAttention(d_model, heads, device,dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

        self.feed_forward = FeedForward(d_model, dff)

    def forward(self, x, mask):

        x1 = lambda x: self.multi_attention(x, x, x, mask)
        x2 = self.residual_connections[0](x, x1)
        x4 = self.residual_connections[1](x2, self.feed_forward)
        return x4


class Encoder(nn.Module):

    def __init__(self, vocab_size, dff, seq_length, d_model, heads,device, dropout, n = 6):

        super(Encoder, self).__init__()

        self.embedding_layer = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_length, device,dropout)
        self.ln = nn.LayerNorm(d_model)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(dff, d_model, heads,device,dropout) for _ in range(n)])

    def forward(self,x_e):

        x = self.positional_encoding(self.embedding_layer(x_e))
        mask = (x_e != 0).long().unsqueeze(1).unsqueeze(2)
        for block in self.encoder_blocks:
            x = block(x, mask)
        return self.ln(x), mask