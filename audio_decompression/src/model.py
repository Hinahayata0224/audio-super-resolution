"""WaveletUNet: PyTorch port of upscalemp3_v2."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_LAYERS, NUM_INIT_FILTERS, FILTER_SIZE, SEGMENT_LENGTH, num_filters_for_level
from .dwt import DWTLayer, IDWTLayer
from .blocks import DownsamplingLayer, UpsamplingLayer, GatedSkipConnection


class WaveletUNet(nn.Module):
    """U-Net with wavelet-based down/upsampling and gated skip connections."""

    def __init__(self):
        super().__init__()

        # Initial block
        self.initial_conv = nn.Conv1d(1, NUM_INIT_FILTERS, FILTER_SIZE, padding="same")
        self.initial_norm = nn.LayerNorm(NUM_INIT_FILTERS)

        # Encoder: 10 levels of Downsampling + DWT + down_process
        self.ds_blocks = nn.ModuleList()
        self.dwt_layers = nn.ModuleList()
        self.down_process = nn.ModuleList()
        for i in range(NUM_LAYERS):
            nf = num_filters_for_level(i)
            in_ch = NUM_INIT_FILTERS if i == 0 else num_filters_for_level(i - 1) * 2
            self.ds_blocks.append(DownsamplingLayer(in_ch, nf))
            self.dwt_layers.append(DWTLayer())
            self.down_process.append(UpsamplingLayer(nf * 2, nf * 2))

        # Bottleneck
        bn_out = NUM_INIT_FILTERS * (NUM_LAYERS + 1)  # 176
        self.bottleneck_conv = nn.Conv1d(num_filters_for_level(9) * 2, bn_out, FILTER_SIZE, padding="same")
        self.bottleneck_norm = nn.LayerNorm(bn_out)

        # Decoder: 10 levels of IDWT + up_process + Skip + UpsamplingBlock
        self.idwt_layers = nn.ModuleList()
        self.up_process = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(NUM_LAYERS):
            level = NUM_LAYERS - 1 - i  # 9, 8, ..., 0
            nf = num_filters_for_level(level)
            prev_nf = bn_out if i == 0 else num_filters_for_level(level + 1)

            self.idwt_layers.append(IDWTLayer())             # prev_nf -> prev_nf//2
            self.up_process.append(UpsamplingLayer(prev_nf // 2, nf))
            self.skip_connections.append(GatedSkipConnection(nf, nf))
            self.up_blocks.append(nn.Sequential(
                nn.Conv1d(nf * 2, nf, FILTER_SIZE, padding="same"),
                PermuteNormGELU(nf),
                nn.Conv1d(nf, nf, FILTER_SIZE, padding="same"),
                PermuteNormGELU(nf),
            ))

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv1d(NUM_INIT_FILTERS, NUM_INIT_FILTERS, FILTER_SIZE, padding="same"),
            PermuteNormGELU(NUM_INIT_FILTERS),
            nn.Conv1d(NUM_INIT_FILTERS, NUM_INIT_FILTERS, FILTER_SIZE, padding="same"),
            PermuteNormGELU(NUM_INIT_FILTERS),
        )

        # Output: concat with input, then 1x1 conv + tanh
        self.output_conv = nn.Conv1d(NUM_INIT_FILTERS + 1, 1, kernel_size=1)

    def _norm_gelu(self, x, norm):
        x = x.permute(0, 2, 1)
        x = norm(x)
        x = F.gelu(x, approximate='tanh')
        return x.permute(0, 2, 1)

    def forward(self, x):
        # x: (B, 1, T) where T ≈ 44100
        B, _, T_in = x.shape

        # Store original input for residual
        orig_mix = x.mean(dim=1, keepdim=True)  # (B, 1, T)

        # Initial
        h = self.initial_conv(x)
        h = self._norm_gelu(h, self.initial_norm)

        # Encoder
        skips = {}
        for i in range(NUM_LAYERS):
            h = self.ds_blocks[i](h)
            skips[i] = h  # store AFTER downsampling, BEFORE DWT
            h = self.dwt_layers[i](h)
            h = self.down_process[i](h)

        # Bottleneck
        h = self.bottleneck_conv(h)
        h = self._norm_gelu(h, self.bottleneck_norm)

        # Decoder
        for i in range(NUM_LAYERS):
            level = NUM_LAYERS - 1 - i
            h = self.idwt_layers[i](h)
            h = self.up_process[i](h)

            # Time alignment with skip (centered, matching TF SYMMETRIC)
            enc = skips[level]
            if h.shape[-1] != enc.shape[-1]:
                diff = enc.shape[-1] - h.shape[-1]
                if diff > 0:
                    pad_start = diff // 2
                    pad_end = diff - pad_start
                    h = F.pad(h, (pad_start, pad_end), mode='reflect')
                else:
                    diff = -diff
                    crop_start = diff // 2
                    h = h[..., crop_start:crop_start + enc.shape[-1]]

            h = self.skip_connections[i]([h, enc])
            h = self.up_blocks[i](h)

        # Final
        h = self.final_conv(h)

        # Time align to input length (centered, matching TF SYMMETRIC)
        if h.shape[-1] != x.shape[-1]:
            diff = x.shape[-1] - h.shape[-1]
            if diff > 0:
                pad_start = diff // 2
                pad_end = diff - pad_start
                h = F.pad(h, (pad_start, pad_end), mode='reflect')
            else:
                diff = -diff
                crop_start = diff // 2
                h = h[..., crop_start:crop_start + x.shape[-1]]

        # Concat with original mix and output
        combined = torch.cat([orig_mix, h], dim=1)  # (B, 17, T)
        out = self.output_conv(combined)
        return torch.tanh(out)


class PermuteNormGELU(nn.Module):
    """Helper: LayerNorm + GELU with (B,C,T) <-> (B,T,C) permutation."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = F.gelu(x, approximate='tanh')
        return x.permute(0, 2, 1)
