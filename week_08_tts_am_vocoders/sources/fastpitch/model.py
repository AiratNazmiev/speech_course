
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import typing as tp

from sources.fastpitch.common.layers import TemporalPredictor
from sources.fastpitch.common.utils import DeviceGetterMixin
from sources.fastpitch.common.utils import regulate_len
from sources.fastpitch.data import FastPitchBatch, SymbolsSet
from sources.fastpitch.hparams import HParamsFastpitch
from sources.fastpitch.common.transformer import FFTransformer


class FastPitch(nn.Module, DeviceGetterMixin):
    def __init__(self, hparams: HParamsFastpitch):
        super().__init__()
        self.hparams = hparams
        n_symbols = len(SymbolsSet().symbols_to_id)

        self.symbol_emb = nn.Embedding(n_symbols, hparams.symbols_embedding_dim)

        self.encoder = FFTransformer(
            n_layer=hparams.in_fft_n_layers,
            n_head=hparams.in_fft_n_heads,
            d_model=hparams.symbols_embedding_dim,
            d_head=hparams.in_fft_d_head,
            d_inner=4 * hparams.symbols_embedding_dim,
            kernel_size=hparams.in_fft_conv1d_kernel_size,
            dropout=hparams.p_in_fft_dropout,
            dropatt=hparams.p_in_fft_dropatt,
            dropemb=hparams.p_in_fft_dropemb
        )

        self.duration_predictor = TemporalPredictor(
            input_size=hparams.symbols_embedding_dim,
            filter_size=hparams.dur_predictor_filter_size,
            kernel_size=hparams.dur_predictor_kernel_size,
            dropout=hparams.p_dur_predictor_dropout,
            n_layers=hparams.dur_predictor_n_layers
        )

        self.pitch_predictor = TemporalPredictor(
            input_size=hparams.symbols_embedding_dim,
            filter_size=hparams.pitch_predictor_filter_size,
            kernel_size=hparams.pitch_predictor_kernel_size,
            dropout=hparams.p_pitch_predictor_dropout,
            n_layers=hparams.pitch_predictor_n_layers
        )

        self.pitch_emb = nn.Conv1d(1, hparams.symbols_embedding_dim, kernel_size=3, padding=1)

        self.decoder = FFTransformer(
            n_layer=hparams.out_fft_n_layers,
            n_head=hparams.out_fft_n_heads,
            d_model=hparams.symbols_embedding_dim,
            d_head=hparams.out_fft_d_head,
            d_inner=4 * hparams.symbols_embedding_dim,
            kernel_size=hparams.out_fft_conv1d_kernel_size,
            dropout=hparams.p_out_fft_dropout,
            dropatt=hparams.p_out_fft_dropatt,
            dropemb=hparams.p_out_fft_dropemb
        )

        self.proj = nn.Linear(hparams.symbols_embedding_dim, hparams.n_mel_channels, bias=True)
        
    # def repeat_enc_out(self, durations, enc_out):
    #     repeats = torch.round(durations).long()
    #     enc_out_rep = pad_sequence(
    #         [torch.repeat_interleave(outp, reps, dim=0) for outp, reps in zip(enc_out, repeats)],
    #         batch_first=True
    #     )
    #     dec_lens = repeats.sum(dim=1)
        
    #     return enc_out_rep, dec_lens

    def get_encoder_out(self, batch: FastPitchBatch):
        '''
        Return: 
        enc_out: 
            Output of the first series of FFT blocks (before adding pitch embedding)
            shape: (batch, len(text), symbols_embedding_dim)
        enc_mask:
            Boolean padding mask for the input text sequences
            shape: (batch, len(text), 1)
        '''
        text_emb = self.symbol_emb(batch.texts)
        enc_out, enc_mask = self.encoder(text_emb, batch.text_lengths)
        
        return enc_out, enc_mask

    def forward(
        self, 
        batch: FastPitchBatch, 
        use_gt_durations: bool = True, 
        use_gt_pitch: bool = True, 
        max_duration: int = 75
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Flags `use_gt_durations` and `use_gt_pitch` should be both True during training and either True or False during inference.

        Use the function `regulate_len` to duplicate phonemes according to durations before passing them to the decoder.
        
        Return:
        mel_out:
            Predicted mel-spectrograms
            shape: (batch, time, mel_bins)
        mel_lens:
            Number of time frames in each of the predicted spectrograms
            shape: (batch,)
        log_dur_pred:
            The predicted log-durations for each phoneme (the output of the duration predictor).
            shape: (batch, len(text))
        dur_pred:
            The exponent of the predicted log-durations for each phoneme. Clamped to the range (0, max_duration) for numeric stability
            shape: (batch, len(text))
        pitch_pred:
            The predicted pitch for each phoneme
            shape: (batch, len(text))
        '''
        
        enc_out, enc_mask = self.get_encoder_out(batch)
        
        log_dur_pred = self.duration_predictor(enc_out, enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, min=0, max=max_duration)
        
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)
        
        if use_gt_pitch:
            pitches = batch.pitches
        else:
            pitches = pitch_pred
            
        pitch_emb = self.pitch_emb(pitches.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)
        
        if use_gt_durations:
            durations = batch.durations
        else:
            durations = dur_pred
            
        enc_out_rep, mel_lens, reps = regulate_len(durations, enc_out, batch.paces)
            
        dec_out, dec_mask = self.decoder(enc_out_rep, mel_lens)
        mel_out = self.proj(dec_out)
        
        return mel_out, mel_lens, dur_pred, log_dur_pred, pitch_pred

    @torch.no_grad()
    def infer(
        self, 
        batch: FastPitchBatch, 
        max_duration: int = 75
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out, dur_pred, pitch_pred = self.infer_encoder(batch, max_duration=max_duration)
        mel_out, mel_lens = self.infer_decoder(enc_out, dur_pred)
        return mel_out, mel_lens, dur_pred, pitch_pred

    def infer_encoder(
        self, 
        batch: FastPitchBatch, 
        max_duration: int = 75
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out, enc_mask = self.get_encoder_out(batch)
        
        log_dur_pred = self.duration_predictor(enc_out, enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, min=0, max=max_duration)
        
        pitch_pred = self.pitch_predictor(enc_out, enc_mask)
        pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))
        
        enc_out = enc_out + pitch_emb.transpose(1, 2)
        
        return enc_out, dur_pred, pitch_pred

    def infer_decoder(
        self, 
        enc_out: torch.Tensor, 
        dur_pred: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        enc_out_rep, mel_lens, reps = regulate_len(dur_pred, enc_out) #self.repeat_enc_out(dur_pred, enc_out)
        dec_out, dec_mask = self.decoder(enc_out_rep, mel_lens)
        mel_out = self.proj(dec_out)
        
        return mel_out, mel_lens
