"""Overlap-add inference for processing long audio."""
import numpy as np
import torch

from .config import SAMPLERATE, SEGMENT_LENGTH

# 每批送入模型的段数。模型为纯 Conv1D，batch 推理可显著提升吞吐，
# 且各段独立、不改变 overlap-add 的数学结果。
DEFAULT_BATCH_SIZE = 8


def process_audio(model, audio_np, sr, device="cpu", batch_size=DEFAULT_BATCH_SIZE):
    """Process numpy audio array through the WaveletUNet model.

    Args:
        model: WaveletUNet in eval mode
        audio_np: numpy array (samples,) for mono, (samples, channels) for multi-channel
        sr: sample rate of input audio
        device: torch device
        batch_size: number of 1s segments fed to the model per forward pass

    Returns:
        numpy array (samples, channels) at 44.1kHz
    """
    model.eval()
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    n_channels = audio_np.shape[1]

    # Resample to 44.1kHz if needed
    if sr != SAMPLERATE:
        import torchaudio
        audio_t = torch.from_numpy(audio_np.T.copy()).float()
        audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLERATE)
        audio_np = audio_t.T.numpy()
        sr = SAMPLERATE

    chunk_samples = SEGMENT_LENGTH  # 44100
    overlap_ratio = 0.5
    step_samples = int(chunk_samples * (1 - overlap_ratio))
    window = np.hanning(chunk_samples).astype(np.float32)
    window = overlap_ratio * window + (1 - overlap_ratio)

    ch_outputs = []
    for ch in range(n_channels):
        ch_data = audio_np[:, ch].astype(np.float32)
        total = len(ch_data)

        # Pad end to complete final chunk
        n_chunks = max(1, (total - chunk_samples + step_samples - 1) // step_samples + 1)
        padded_len = (n_chunks - 1) * step_samples + chunk_samples
        if padded_len > total:
            ch_data = np.pad(ch_data, (0, padded_len - total))

        out_accum = np.zeros(padded_len, dtype=np.float64)
        win_accum = np.zeros(padded_len, dtype=np.float64)

        # Pre-window every chunk once, then run batched forward passes.
        clips_w = np.stack([
            ch_data[k * step_samples:k * step_samples + chunk_samples] * window
            for k in range(n_chunks)
        ]).astype(np.float32)  # (n_chunks, chunk_samples)

        with torch.no_grad():
            for b0 in range(0, n_chunks, batch_size):
                batch = clips_w[b0:b0 + batch_size]  # (B, chunk_samples)
                inp = torch.from_numpy(batch).unsqueeze(1).to(device)  # (B, 1, chunk_samples)
                preds = model(inp).squeeze(1).cpu().numpy()  # (B, chunk_samples)
                for j, pred in enumerate(preds):
                    k = b0 + j
                    start = k * step_samples
                    out_accum[start:start + chunk_samples] += pred * window
                    win_accum[start:start + chunk_samples] += window * window

        out_accum = out_accum / np.maximum(win_accum, 1e-10)
        ch_outputs.append(out_accum[:total])

    return np.column_stack(ch_outputs) if n_channels > 1 else ch_outputs[0]

