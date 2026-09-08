import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, heads, device,dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_heads = d_model//heads
        self.device = device

        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask = None):
        
        batch_size, seq_len, d_model = x_q.shape

        query = self.w_q(x_q).view(batch_size, x_q.shape[1], self.heads, self.d_heads).permute(0, 2, 1, 3)
        key = self.w_k(x_k).view(batch_size, x_k.shape[1], self.heads, self.d_heads).permute(0, 2, 1, 3)
        value = self.w_v(x_v).view(batch_size, x_v.shape[1], self.heads, self.d_heads).permute(0, 2, 1, 3)

        similarity = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_heads)
        if mask is not None:
            # print(mask.shape, similarity.shape)
            # mask = mask.to(self.device)
            # print(mask.shape)
            similarity = similarity.masked_fill(~mask.bool() , float('-inf'))
        sim = F.softmax(similarity, dim = -1)
        final = torch.matmul(sim, value)
        final = self.dropout(final)

        final = final.permute(0, 2, 1, 3).contiguous()
        final = final.view(batch_size, seq_len, self.d_model)
        

        return self.w_o(final)