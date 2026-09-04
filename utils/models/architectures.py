"""Every model architecture used across the three modelling stages.

Stage 1  MLP / RNNModel / LSTMModel                    -- single-step, lookback only
Stage 2  TidalSeq2Seq (+ flat multi-horizon baselines) -- UTide-informed BiLSTM decoder
Stage 3  TidalSeq2SeqMulti / TidalSeq2SeqTideOnly      -- multi-branch meteo fusion

Layer names and shapes are unchanged from the notebooks these were extracted from, so
checkpoints written by the notebooks load into these classes without conversion. The
stage-3 models took no constructor arguments in the notebook (they closed over module
globals); here the same values are explicit keyword defaults.
"""
import torch
import torch.nn as nn

from .layers import BatchedAttention, Encoder

# --- stage 2/3 sizes, previously globals in the stage notebooks' setup cells ---
ENC_HIDDEN_SIZE = 64
ENC_NUM_LAYERS = 1
DEC_HIDDEN_SIZE = 64
DEC_NUM_LAYERS = 1
ATTN_DIM_TIDE = 64
METEO_HIDDEN_SIZE = 64
ATTN_DIM_METEO = 64
RIVER_HIDDEN_SIZE = 32
ATTN_DIM_RIVER = 32


# ======================================================================================
# Stage 1: single-feature, (batch, lookback) in -> (batch,) out
# ======================================================================================
class MLP(nn.Module):
    def __init__(self, lookback, hidden=(128, 64)):
        super().__init__()
        dims = [lookback, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RNNModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


# ======================================================================================
# Stage 2: UTide-informed BiLSTM decoder with attention, plus flat multi-horizon baselines
# ======================================================================================
class TidalDecoder(nn.Module):
    """Bidirectional LSTM stepping once per horizon, fed UTide(t+k) at step k. The initial
    hidden/cell state is projected from the encoder's final layer, attention pulls context
    from the encoder at each step, and the prediction is a linear readout of
    [decoder_output; attention_context].
    """

    def __init__(self, feat_size, hidden_size, enc_hidden_size, num_layers=1, attn_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        attn_dim = attn_dim or hidden_size

        self.lstm = nn.LSTM(feat_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(enc_hidden_size, num_layers * 2 * hidden_size)
        self.init_c = nn.Linear(enc_hidden_size, num_layers * 2 * hidden_size)
        self.attn = BatchedAttention(2 * hidden_size, enc_hidden_size, attn_dim)
        self.out = nn.Linear(2 * hidden_size + enc_hidden_size, 1)

    def forward(self, utide_future, enc_outputs, enc_final):
        h_n, c_n = enc_final                        # each (enc_num_layers, B, enc_hidden_size)
        h_last, c_last = h_n[-1], c_n[-1]           # final encoder layer only: (B, enc_hidden_size)
        B = h_last.size(0)

        h0 = self.init_h(h_last).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = self.init_c(c_last).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()

        dec_out, _ = self.lstm(utide_future, (h0, c0))       # (B, H, 2*hidden_size)
        context, weights = self.attn(dec_out, enc_outputs)   # (B, H, enc_hidden_size)
        combined = torch.cat([dec_out, context], dim=-1)
        return self.out(combined).squeeze(-1), weights        # (B, H)


class TidalSeq2Seq(nn.Module):
    """Stage 2: LSTM encoder + attention + BiLSTM decoder fed known-future UTide values."""

    def __init__(self, enc_hidden_size=ENC_HIDDEN_SIZE, enc_num_layers=ENC_NUM_LAYERS,
                 dec_hidden_size=DEC_HIDDEN_SIZE, dec_num_layers=DEC_NUM_LAYERS,
                 attn_dim=ATTN_DIM_TIDE):
        super().__init__()
        self.encoder = Encoder(1, enc_hidden_size, enc_num_layers)
        self.decoder = TidalDecoder(1, dec_hidden_size, enc_hidden_size, dec_num_layers, attn_dim)

    def forward(self, x_hist, utide_future):
        enc_outputs, enc_final = self.encoder(x_hist.unsqueeze(-1))
        preds, weights = self.decoder(utide_future.unsqueeze(-1), enc_outputs, enc_final)
        return preds, weights


class FlatMultiHorizonLSTM(nn.Module):
    """Stage-1 reproduction: same LSTM encoder, but a flat linear head predicting all
    horizons at once from the final hidden state -- no decoder, no UTide input, no
    attention. Stage 1 only ever trained single-step models (evaluated by recursive
    rollout), so this is the fair direct-multi-horizon baseline stage 2 has to beat.
    """

    def __init__(self, hidden_size, num_layers, n_horizons):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x_hist):
        out, _ = self.lstm(x_hist.unsqueeze(-1))
        return self.head(out[:, -1, :])


class FlatMultiHorizonMLP(nn.Module):
    """Stage-1 reproduction of MLP(lookback, hidden=(128, 64)): same body, but the final
    layer predicts all horizons at once instead of a single next-step value.
    """

    def __init__(self, lookback, n_horizons, hidden=(128, 64)):
        super().__init__()
        dims = [lookback, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_horizons))
        self.net = nn.Sequential(*layers)

    def forward(self, x_hist):
        return self.net(x_hist)


# ======================================================================================
# Stage 3: multi-resolution meteorological fusion
# ======================================================================================
class MultiBranchTidalDecoder(nn.Module):
    """1. Bridge: init_h/init_c project from all three branches' final hidden states.
    2. Decoder sequence: the BiLSTM runs over the full padded UTide sequence, which is
       then sliced to the trained horizon positions.
    3. Attention: one BatchedAttention per branch, concatenated with dec_out before the
       output head.
    """

    def __init__(self, dec_hidden_size, dec_num_layers, branch_hidden_sizes, attn_dims,
                 decoder_seq_len, target_slice):
        super().__init__()
        self.hidden_size = dec_hidden_size
        self.num_layers = dec_num_layers
        self.decoder_seq_len = decoder_seq_len
        self.target_slice = target_slice
        bridge_dim = sum(branch_hidden_sizes)  # tide + meteo + river final-hidden concat

        self.lstm = nn.LSTM(1, dec_hidden_size, dec_num_layers, batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(bridge_dim, dec_num_layers * 2 * dec_hidden_size)
        self.init_c = nn.Linear(bridge_dim, dec_num_layers * 2 * dec_hidden_size)

        self.attn_tide = BatchedAttention(2 * dec_hidden_size, branch_hidden_sizes[0], attn_dims[0])
        self.attn_meteo = BatchedAttention(2 * dec_hidden_size, branch_hidden_sizes[1], attn_dims[1])
        self.attn_river = BatchedAttention(2 * dec_hidden_size, branch_hidden_sizes[2], attn_dims[2])

        out_dim = 2 * dec_hidden_size + sum(branch_hidden_sizes)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, utide_dense, enc_tide, final_tide, enc_meteo, final_meteo, enc_river, final_river):
        # final encoder layer only from each branch, matching stage 2's convention
        h_tide, c_tide = final_tide[0][-1], final_tide[1][-1]
        h_meteo, c_meteo = final_meteo[0][-1], final_meteo[1][-1]
        h_river, c_river = final_river[0][-1], final_river[1][-1]

        h_bridge = torch.cat([h_tide, h_meteo, h_river], dim=-1)
        c_bridge = torch.cat([c_tide, c_meteo, c_river], dim=-1)
        B = h_bridge.size(0)

        h0 = self.init_h(h_bridge).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = self.init_c(c_bridge).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()

        dec_out_dense, _ = self.lstm(utide_dense.unsqueeze(-1), (h0, c0))   # (B, 34, 2*hidden)
        dec_out = dec_out_dense[:, self.target_slice, :]                    # (B, 30, 2*hidden)

        context_tide, w_tide = self.attn_tide(dec_out, enc_tide)
        context_meteo, w_meteo = self.attn_meteo(dec_out, enc_meteo)
        context_river, w_river = self.attn_river(dec_out, enc_river)

        combined = torch.cat([dec_out, context_tide, context_meteo, context_river], dim=-1)
        preds = self.out(combined).squeeze(-1)   # (B, 30)
        return preds, (w_tide, w_meteo, w_river)


class TidalSeq2SeqMulti(nn.Module):
    """Stage 3: three encoders (tide/meteo/river) + multi-branch attention + UTide-informed
    BiLSTM decoder over a densified, padded horizon sequence.
    """

    def __init__(self, decoder_seq_len, target_slice,
                 enc_hidden_size=ENC_HIDDEN_SIZE, enc_num_layers=ENC_NUM_LAYERS,
                 meteo_hidden_size=METEO_HIDDEN_SIZE, river_hidden_size=RIVER_HIDDEN_SIZE,
                 dec_hidden_size=DEC_HIDDEN_SIZE, dec_num_layers=DEC_NUM_LAYERS,
                 attn_dims=(ATTN_DIM_TIDE, ATTN_DIM_METEO, ATTN_DIM_RIVER)):
        super().__init__()
        self.encoder_tide = Encoder(1, enc_hidden_size, enc_num_layers)
        self.encoder_meteo = Encoder(3, meteo_hidden_size, enc_num_layers)
        self.encoder_river = Encoder(4, river_hidden_size, enc_num_layers)
        self.decoder = MultiBranchTidalDecoder(
            dec_hidden_size, dec_num_layers,
            branch_hidden_sizes=(enc_hidden_size, meteo_hidden_size, river_hidden_size),
            attn_dims=attn_dims,
            decoder_seq_len=decoder_seq_len, target_slice=target_slice,
        )

    def forward(self, x_tide, x_meteo, x_river, utide_dense):
        enc_tide, final_tide = self.encoder_tide(x_tide.unsqueeze(-1))
        enc_meteo, final_meteo = self.encoder_meteo(x_meteo)
        enc_river, final_river = self.encoder_river(x_river)
        preds, weights = self.decoder(utide_dense, enc_tide, final_tide,
                                      enc_meteo, final_meteo, enc_river, final_river)
        return preds, weights


class TideOnlyDecoder(nn.Module):
    """Single-branch counterpart to MultiBranchTidalDecoder: same decoder sequence and
    horizons, fed only the tide branch.
    """

    def __init__(self, dec_hidden_size, dec_num_layers, tide_hidden_size, attn_dim,
                 decoder_seq_len, target_slice):
        super().__init__()
        self.hidden_size = dec_hidden_size
        self.num_layers = dec_num_layers
        self.decoder_seq_len = decoder_seq_len
        self.target_slice = target_slice

        self.lstm = nn.LSTM(1, dec_hidden_size, dec_num_layers, batch_first=True, bidirectional=True)
        self.init_h = nn.Linear(tide_hidden_size, dec_num_layers * 2 * dec_hidden_size)
        self.init_c = nn.Linear(tide_hidden_size, dec_num_layers * 2 * dec_hidden_size)
        self.attn_tide = BatchedAttention(2 * dec_hidden_size, tide_hidden_size, attn_dim)
        self.out = nn.Linear(2 * dec_hidden_size + tide_hidden_size, 1)

    def forward(self, utide_dense, enc_tide, final_tide):
        h_tide, c_tide = final_tide[0][-1], final_tide[1][-1]
        B = h_tide.size(0)

        h0 = self.init_h(h_tide).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = self.init_c(c_tide).view(B, self.num_layers * 2, self.hidden_size).permute(1, 0, 2).contiguous()

        dec_out_dense, _ = self.lstm(utide_dense.unsqueeze(-1), (h0, c0))
        dec_out = dec_out_dense[:, self.target_slice, :]

        context_tide, w_tide = self.attn_tide(dec_out, enc_tide)
        combined = torch.cat([dec_out, context_tide], dim=-1)
        preds = self.out(combined).squeeze(-1)
        return preds, (w_tide,)


class TidalSeq2SeqTideOnly(nn.Module):
    """Same redesigned decoder (densified sequence, 30 horizons) as TidalSeq2SeqMulti, but
    without the meteo/river branches -- the meteo-contribution ablation.
    """

    def __init__(self, decoder_seq_len, target_slice,
                 enc_hidden_size=ENC_HIDDEN_SIZE, enc_num_layers=ENC_NUM_LAYERS,
                 dec_hidden_size=DEC_HIDDEN_SIZE, dec_num_layers=DEC_NUM_LAYERS,
                 attn_dim=ATTN_DIM_TIDE):
        super().__init__()
        self.encoder_tide = Encoder(1, enc_hidden_size, enc_num_layers)
        self.decoder = TideOnlyDecoder(
            dec_hidden_size, dec_num_layers, enc_hidden_size, attn_dim,
            decoder_seq_len=decoder_seq_len, target_slice=target_slice,
        )

    def forward(self, x_tide, utide_dense):
        enc_tide, final_tide = self.encoder_tide(x_tide.unsqueeze(-1))
        preds, weights = self.decoder(utide_dense, enc_tide, final_tide)
        return preds, weights
