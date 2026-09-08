import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_length, device, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(self.seq_length, self.d_model)
        positions = torch.arange(0, self.seq_length, dtype = torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe) # By adding this in register buffer this stores pe too while saving the model without considering it as a learning parameter
                

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

    
class ResidualConnection(nn.Module):

    def __init__(self, d_model ,dropout):

        super(ResidualConnection, self).__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, sublayer):

        return x1 + self.dropout(sublayer(self.ln(x1)))


class FeedForward(nn.Module):

    def __init__(self, d_model, dff, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.forward1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.forward2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.forward2(self.dropout(torch.relu(self.forward1(x))))