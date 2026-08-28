"""Building blocks: DownsamplingLayer, UpsamplingLayer, GatedSkipConnection."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FILTER_SIZE


class DownsamplingLayer(nn.Module):
    """Conv1D + LayerNorm + GELU. Conv handles channel mismatch directly."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, FILTER_SIZE, padding="same")
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = F.gelu(x, approximate='tanh')
        x = x.permute(0, 2, 1)
        return x


class UpsamplingLayer(nn.Module):
    """Gated wavelet-aware processing. Does NOT change time resolution.

    half = in_channels // 2.  output_conv projects from in_channels to out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half = in_channels // 2
        self.approx_conv = nn.Conv1d(half, half, FILTER_SIZE, padding="same")
        self.approx_norm = nn.LayerNorm(half)
        self.approx_gate = nn.Conv1d(half, half, kernel_size=1)
        self.detail_conv = nn.Conv1d(half, half, FILTER_SIZE, padding="same")
        self.detail_norm = nn.LayerNorm(half)
        self.detail_gate = nn.Conv1d(half, half, kernel_size=1)
        self.output_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.output_norm = nn.LayerNorm(out_channels)

    def _norm_gelu(self, x, norm):
        x = x.permute(0, 2, 1)
        x = norm(x)
        x = F.gelu(x, approximate='tanh')
        return x.permute(0, 2, 1)

    def forward(self, x):
        C = x.shape[1]
        half = C // 2
        a, d = x[:, :half, :], x[:, half:, :]

        a = self.approx_conv(a)
        a = self._norm_gelu(a, self.approx_norm)
        a = a * torch.sigmoid(self.approx_gate(a))

        d = self.detail_conv(d)
        d = self._norm_gelu(d, self.detail_norm)
        d = d * torch.sigmoid(self.detail_gate(d))

        merged = torch.cat([a, d], dim=1)
        out = self.output_conv(merged)
        out = self._norm_gelu(out, self.output_norm)
        return out


class GatedSkipConnection(nn.Module):
    """Learned fusion of decoder features with encoder skip connection."""

    def __init__(self, decoder_ch: int, encoder_ch: int):
        super().__init__()
        self.decoder_gate = nn.Conv1d(decoder_ch, decoder_ch, kernel_size=1)
        self.encoder_gate = nn.Conv1d(encoder_ch, encoder_ch, kernel_size=1)
        self.norm = nn.LayerNorm(decoder_ch + encoder_ch)

    def forward(self, x):
        dec, enc = x if isinstance(x, (list, tuple)) else (x, None)
        dec = dec * torch.sigmoid(self.decoder_gate(dec))
        enc = enc * torch.sigmoid(self.encoder_gate(enc))
        out = torch.cat([dec, enc], dim=1)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        return out.permute(0, 2, 1)
