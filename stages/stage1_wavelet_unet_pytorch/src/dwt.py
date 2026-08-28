"""DWT and IDWT layers using frozen Daubechies-4 filters."""
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WAVELET_FAMILY


def _get_wavelet_filters():
    """Return dec_lo, dec_hi, rec_lo, rec_hi as torch tensors [filter_len]."""
    w = pywt.Wavelet(WAVELET_FAMILY)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
    return dec_lo, dec_hi, rec_lo, rec_hi


class DWTLayer(nn.Module):
    """Single-level DWT decomposition. Halves time, doubles channels."""

    def __init__(self):
        super().__init__()
        dec_lo, dec_hi, _, _ = _get_wavelet_filters()
        # Register as buffers (not trainable)
        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

    def _pad(self, x):
        """REFLECT-pad by filter_len-1 on each side."""
        pad = len(self.dec_lo) - 1
        return F.pad(x, (pad, pad), mode="reflect")

    @staticmethod
    def _trim(x, trim=3):
        """Trim boundary artifacts from both ends."""
        return x[..., trim:-trim] if x.shape[-1] > 2 * trim else x

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        padded = self._pad(x)       # (B, C, T + 2*filter_len - 2)

        # Per-channel grouped convolution with stride 2
        w_lo = self.dec_lo.view(1, 1, -1).repeat(C, 1, 1)  # (C, 1, filter_len)
        w_hi = self.dec_hi.view(1, 1, -1).repeat(C, 1, 1)  # (C, 1, filter_len)

        lo = F.conv1d(padded, w_lo, stride=2, groups=C, padding=0)
        hi = F.conv1d(padded, w_hi, stride=2, groups=C, padding=0)

        lo = self._trim(lo)
        hi = self._trim(hi)

        return torch.cat([lo, hi], dim=1)  # (B, 2*C, T_dwt)


class IDWTLayer(nn.Module):
    """Single-level IDWT reconstruction. Doubles time, halves channels."""

    def __init__(self):
        super().__init__()
        _, _, rec_lo, rec_hi = _get_wavelet_filters()
        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)

    def _upsample(self, x):
        """Zero-insertion upsampling by factor 2."""
        B, C, T = x.shape
        # Insert zeros between samples via reshape + pad trick
        out = torch.zeros(B, C, T * 2, device=x.device, dtype=x.dtype)
        out[..., ::2] = x
        return out

    def forward(self, x):
        # x: (B, 2*C, T)
        B, total_C, T = x.shape
        C = total_C // 2
        approx, detail = x[:, :C, :], x[:, C:, :]  # (B, C, T)

        # Upsample
        approx_up = self._upsample(approx)  # (B, C, 2*T)
        detail_up = self._upsample(detail)  # (B, C, 2*T)

        # Reconstruct with TF-style asymmetric SAME padding (left=3, right=4 for k=8)
        fl = len(self.rec_lo)
        pad_l = (fl - 1) // 2  # 3
        pad_r = fl // 2         # 4

        w_lo = self.rec_lo.view(1, 1, -1).repeat(C, 1, 1)
        w_hi = self.rec_hi.view(1, 1, -1).repeat(C, 1, 1)

        approx_padded = F.pad(approx_up, (pad_l, pad_r), mode="constant")
        detail_padded = F.pad(detail_up, (pad_l, pad_r), mode="constant")

        lo_out = F.conv1d(approx_padded, w_lo, stride=1, groups=C)
        hi_out = F.conv1d(detail_padded, w_hi, stride=1, groups=C)

        target_T = T * 2
        return (lo_out + hi_out)[..., :target_T]
