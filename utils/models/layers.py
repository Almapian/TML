"""Encoder and attention blocks shared by stages 2 and 3.

Both classes were previously copy-pasted verbatim between
`stage2_utide_attention.ipynb` and `stage3_meteo_fusion.ipynb`; stage 3 instantiates
them three times (tide / meteo / river branches).
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """LSTM encoder returning the full hidden-state sequence, so a decoder's attention
    can look back over the whole history rather than only the final state."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):  # x: (B, T_hist, input_size)
        enc_outputs, (h_n, c_n) = self.lstm(x)
        return enc_outputs, (h_n, c_n)


class BatchedAttention(nn.Module):
    """Additive dot-product attention: project decoder output and encoder states into
    `attn_dim`, score, softmax, then take the weighted sum of the raw encoder states."""

    def __init__(self, dec_dim, enc_dim, attn_dim):
        super().__init__()
        self.dec_proj = nn.Linear(dec_dim, attn_dim)
        self.enc_proj = nn.Linear(enc_dim, attn_dim)

    def forward(self, dec_out, enc_outputs):
        q = self.dec_proj(dec_out)                 # (B, H, attn_dim)
        k = self.enc_proj(enc_outputs)             # (B, T_hist, attn_dim)
        scores = torch.bmm(q, k.transpose(1, 2))   # (B, H, T_hist)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, enc_outputs)  # (B, H, enc_dim) -- raw enc_outputs, not k
        return context, weights
