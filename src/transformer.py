import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class MyTransformer(nn.Module):

    def __init__(self, vocab_size_source, vocab_size_target, seq_length_source, seq_length_target, device,d_model = 512, heads = 8, dropout = 0.1, dff = 2048):

        super(MyTransformer, self).__init__()

        self.encoder = Encoder(vocab_size_source, dff, seq_length_source, d_model, heads, device, dropout)
        # print(1)
        self.decoder = Decoder(vocab_size_target, dff, seq_length_target, d_model, heads, device, dropout)

        # for parameter in self.parameters():
        #     if isinstance(parameter, nn.Linear):
        #         nn.init.xavier_uniform_(parameter)

    def forward(self, x_in, x_op=None):
        x_enc, src_mask = self.encoder(x_in)
        if x_op is None:
            return x_enc, src_mask
        output = self.decoder(x_op, x_enc, src_mask)
        return output