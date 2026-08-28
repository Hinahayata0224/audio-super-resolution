"""Overlap-add inference for processing long audio."""
import numpy as np
import torch

from .config import SAMPLERATE, SEGMENT_LENGTH


def process_audio(model, audio_np, sr, device="cpu"):
    """Process numpy audio array through the WaveletUNet model.

    Args:
        model: WaveletUNet in eval mode
        audio_np: numpy array (samples,) for mono, (samples, channels) for multi-channel
        sr: sample rate of input audio
        device: torch device

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

        with torch.no_grad():
            for k in range(n_chunks):
                start = k * step_samples
                clip = ch_data[start:start + chunk_samples]
                clip_w = clip * window

                inp = torch.from_numpy(clip_w).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(inp).squeeze().cpu().numpy()

                out_accum[start:start + chunk_samples] += pred * window
                win_accum[start:start + chunk_samples] += window * window

        out_accum = out_accum / np.maximum(win_accum, 1e-10)
        ch_outputs.append(out_accum[:total])

    return np.column_stack(ch_outputs) if n_channels > 1 else ch_outputs[0]
