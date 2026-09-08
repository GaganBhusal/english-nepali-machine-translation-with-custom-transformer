import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .layers import ResidualConnection, FeedForward, InputEmbedding, PositionalEncoding



class DecoderBlock(nn.Module):

    def __init__(self, dff, d_model, heads, device,dropout):

        super(DecoderBlock, self).__init__()
        
        self.masked_attention = MultiHeadAttention(d_model, heads, device, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        self.cross_attention = MultiHeadAttention(d_model, heads, device, dropout)
        self.feed_forward = FeedForward(d_model, dff)


    def forward(self, x, x_enc, tgt_mask, src_mask):
        
        x1= lambda x: self.masked_attention(x, x, x, tgt_mask)
        x2 = self.residual_connections[0](x, x1)

        x3= lambda x: self.cross_attention(x, x_enc, x_enc, src_mask)
        x4 = self.residual_connections[1](x2, x3)

        x5 = self.feed_forward
        x6 = self.residual_connections[2](x4, x5)

        return x6


class Decoder(nn.Module):

    def __init__(self, vocab_size, dff, seq_length, d_model, heads,device ,dropout, n = 6):

        super(Decoder, self).__init__()
        
        self.embedding_layer = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_length, device,dropout)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(dff, d_model, heads,device, dropout) for _ in range(n)])
        self.ln = nn.LayerNorm(d_model)

        self.register_buffer("casual_mask", torch.tril(torch.ones(seq_length, seq_length)))
        self.linear = nn.Linear(d_model, vocab_size)


    def forward(self, x_d, x_enc, src_mask = None, tgt_mask = None):
        
        batch_size, seq_len = x_d.size()
        
        # Padding mask for target (ignore 0s)
        padding_mask = (x_d != 0).unsqueeze(1).unsqueeze(2) 
        
        # Look-ahead mask (triangle)
        look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len)).to(x_d.device)
        
        # Combine them
        tgt_mask = padding_mask & look_ahead_mask.bool()

        # 2. Embeddings
        x = self.positional_encoding(self.embedding_layer(x_d))

        # 3. Pass masks to blocks
        for block in self.decoder_blocks:
            # src_mask comes from the function argument (from Encoder)
            x = block(x, x_enc, tgt_mask, src_mask)
            
        return self.linear(self.ln(x))