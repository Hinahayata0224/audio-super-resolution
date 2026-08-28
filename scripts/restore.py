#!/usr/bin/env python
"""
MP3/AAC Music Restoration Pipeline
  Stage 1: upscalemp3_v2 wavelet-based MP3 artifact removal @ 44.1kHz
  Stage 2: AudioSR super-resolution @ 48kHz

Usage:
  set HF_HOME=G:\huggingface_cache
  set HF_ENDPOINT=https://hf-mirror.com

  python restore.py -i "music.mp3"                       # full pipeline
  python restore.py -i "music.mp3" --stage1 skip          # Stage 2 only
  python restore.py -i "music.mp3" -o ./out --ddim_steps 100
"""

import os
import sys
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("FORCE_COLOR", "0")

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Use local model cache (models/huggingface/)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_CACHE = os.path.join(PROJECT, "models", "huggingface")
os.environ["HF_HUB_CACHE"] = _LOCAL_CACHE
os.environ["TRANSFORMERS_CACHE"] = _LOCAL_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Monkey-patch roberta-base lookups before audiosr imports them at module level
_ROBERTA_LOCAL = os.path.join(PROJECT, "models", "huggingface", "roberta-base")
import transformers
from transformers import RobertaTokenizer, RobertaConfig
_rt_orig = RobertaTokenizer.from_pretrained.__func__
_rc_orig = RobertaConfig.from_pretrained.__func__
def _patch_fp(cls, name, *args, **kwargs):
    if name == "roberta-base":
        name = _ROBERTA_LOCAL
    return _rt_orig(cls, name, *args, **kwargs)
def _patch_fp_cfg(cls, name, *args, **kwargs):
    if name == "roberta-base":
        name = _ROBERTA_LOCAL
    return _rc_orig(cls, name, *args, **kwargs)
RobertaTokenizer.from_pretrained = classmethod(_patch_fp)
RobertaConfig.from_pretrained = classmethod(_patch_fp_cfg)

import argparse
import numpy as np
import pyloudnorm as pyln
import shutil
import soundfile as sf
import torch
import torchaudio
from audiosr import build_model, super_resolution

# Ensure project root is on path
sys.path.insert(0, PROJECT)


def main():
    parser = argparse.ArgumentParser(
        description="Restore compressed music (MP3/AAC) to near-lossless quality"
    )
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("--stage1", default="upscalemp3", choices=["upscalemp3", "skip"],
                        help="Stage 1 de-artifacting (skip for Stage 2 only)")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--model", default="basic", choices=["basic", "speech"],
                        help="AudioSR model: basic (music) / speech")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="AudioSR sampling steps (auto if not set)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="AudioSR guidance scale (auto if not set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep Stage 1 intermediate file")
    parser.add_argument("--skip-loudness-match", action="store_true",
                        help="Skip loudness matching after Stage 2")
    parser.add_argument("--no-lowpass", action="store_true",
                        help="Bypass lowpass filter, use full-band mel conditioning")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be very slow!)")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device}, TF32: ON")

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    print(f"Loading: {args.input}")
    wav_np, sr = sf.read(args.input)
    wav = torch.from_numpy(wav_np.T).float()
    duration = wav.shape[1] / sr
    print(f"  Original: {sr}Hz, {wav.shape[0]}ch, {duration:.1f}s")

    # ── Pre-analysis ──
    import pyloudnorm as pyln

    file_size = os.path.getsize(args.input)
    bitrate = (file_size * 8) / (duration * 1000)
    peak = wav.abs().max().item()
    peak_db = 20 * np.log10(max(peak, 1e-10))
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(wav_np)

    print(f"  Input: {file_size/1024/1024:.1f} MB, ~{bitrate:.0f} kbps")
    if peak > 0.99:
        print(f"  WARNING: True peak = {peak_db:.1f} dBFS (clipping!)")
    elif peak > 0.5:
        print(f"  True peak: {peak_db:.1f} dBFS, headroom: {-peak_db:.1f} dB")
    else:
        print(f"  True peak: {peak_db:.1f} dBFS (quiet)")
    print(f"  Integrated loudness: {lufs:.1f} LUFS")

    # Spectral roll-off for auto-tuning Stage 2 params
    segment = wav[:, :min(wav.shape[1], int(sr * 5))]
    n_fft = 2048
    win = torch.hann_window(n_fft)
    spec = torch.stft(segment.float().mean(dim=0), n_fft=n_fft,
                      hop_length=n_fft // 4, window=win, return_complex=True)
    spec_mag = spec.abs().mean(dim=-1)
    energy = torch.cumsum(spec_mag, dim=0)
    rolloff_hz = torch.searchsorted(energy, energy[-1] * 0.995).item() * sr / n_fft

    if rolloff_hz < 17000:
        comp = "heavy"
        print(f"  Spectral roll-off @ 99.5%: {rolloff_hz:.0f} Hz -> may be lossy-compressed source")
    elif rolloff_hz < 21000:
        comp = "moderate"
        print(f"  Spectral roll-off @ 99.5%: {rolloff_hz:.0f} Hz (typical AAC 192-256k)")
    else:
        comp = "wideband"
        print(f"  Spectral roll-off @ 99.5%: {rolloff_hz:.0f} Hz (wideband)")

    auto_s2 = {"heavy": (75, 4.0), "moderate": (50, 3.5), "wideband": (30, 3.0)}
    auto_ddim, auto_g = auto_s2[comp]
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else auto_ddim
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else auto_g
    print(f"  Auto params [{comp}]: ddim={ddim_steps}, guidance={guidance_scale:.1f}")

    temp_dir = os.path.join(args.output, ".temp")
    os.makedirs(temp_dir, exist_ok=True)

    # ================================================================
    # Stage 1: upscalemp3_v2 MP3 Artifact Removal
    # ================================================================

    if args.stage1 == "skip":
        print("\n" + "=" * 60)
        print("Stage 1: SKIPPED (Stage 2 only)")
        print("=" * 60)
        temp_path = args.input
        stage1_sr = sr

    else:  # upscalemp3
        from audio_decompression import load_model, process_audio

        print("\n" + "=" * 60)
        print("Stage 1: audio_decompression (Wavelet U-Net) @ 44.1kHz")
        print("=" * 60)

        model = load_model(device=device)

        print("  Processing...")
        stage1_output = process_audio(model, wav_np, sr, device=device)
        if stage1_output.ndim == 1:
            stage1_output = stage1_output[:, None]
        stage1_sr = 44100
        print(f"  Stage 1 done, shape: {stage1_output.shape} @ {stage1_sr}Hz")

    if args.stage1 != "skip":
        stage1_name = f"{base_name}_stage1_upscalemp3.wav"
        if args.keep_temp:
            sf.write(os.path.join(temp_dir, stage1_name), stage1_output, stage1_sr)
            print(f"  Stage 1 output saved: {temp_dir}/{stage1_name}")

    if device == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # Stage 2: AudioSR Super-Resolution to 48kHz (per-channel)
    # ================================================================
    print("\n" + "=" * 60)
    print("Stage 2: AudioSR Super-Resolution @ 48kHz")
    print("=" * 60)

    # Use local checkpoint to avoid network
    import audiosr.pipeline as audiosr_pipeline
    _original_download = audiosr_pipeline.download_checkpoint
    audiosr_pipeline.download_checkpoint = lambda name: os.path.join(
        PROJECT, "models", "audiosr", "pytorch_model.bin")

    audiosr_model = build_model(model_name=args.model, device=device)

    audiosr_pipeline.download_checkpoint = _original_download

    if args.no_lowpass:
        _orig_make = audiosr_pipeline.make_batch_for_super_resolution
        def _bypass_lowpass(input_file, waveform=None, fbank=None):
            batch, duration = _orig_make(input_file, waveform, fbank)
            batch["lowpass_mel"] = batch["log_mel_spec"]
            return batch, duration
        audiosr_pipeline.make_batch_for_super_resolution = _bypass_lowpass
        print("  Lowpass: OFF (using full-band mel)")

    if args.stage1 == "skip":
        st1_np, st1_sr = sf.read(temp_path)
        st1_audio = torch.from_numpy(st1_np.T).float()
    else:
        st1_audio = torch.from_numpy(stage1_output.T.copy()).float()
        st1_sr = stage1_sr

    if st1_sr != 48000:
        st1_audio = torchaudio.functional.resample(st1_audio, st1_sr, 48000, lowpass_filter_width=64, rolloff=0.995)

    total_s = st1_audio.shape[1] / 48000
    chunk_s = 10.24
    overlap_s = 2.56
    max_chunk_samples = int(chunk_s * 48000)
    overlap_samples = int(overlap_s * 48000)
    step_samples = max_chunk_samples - overlap_samples
    n_channels = st1_audio.shape[0]

    def process_one_channel(ch_mono, label):
        import time
        ch_path = os.path.join(temp_dir, f"{base_name}_{label}.wav")
        t0 = time.time()
        sf.write(ch_path, ch_mono.numpy(), 48000, subtype='FLOAT')
        t1 = time.time()
        out = super_resolution(audiosr_model, ch_path, seed=args.seed,
                                guidance_scale=guidance_scale, ddim_steps=ddim_steps)
        t2 = time.time()
        os.remove(ch_path)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out.squeeze())
        else:
            out = out.squeeze().cpu()
        print(f"  [{label}] write={t1-t0:.1f}s  sr={t2-t1:.1f}s  total={t2-t0:.1f}s")
        return out

    if total_s <= chunk_s + 0.01:
        print(f"  Single-pass ({total_s:.1f}s, {n_channels}ch)")
        ch_outputs = []
        for ch_idx in range(n_channels):
            ch_mono = st1_audio[ch_idx]
            ch_out = process_one_channel(ch_mono, f"ch{ch_idx}")
            if ch_out.shape[0] > st1_audio.shape[1]:
                ch_out = ch_out[:st1_audio.shape[1]]
            elif ch_out.shape[0] < st1_audio.shape[1]:
                ch_out = torch.nn.functional.pad(ch_out, (0, st1_audio.shape[1] - ch_out.shape[0]))
            ch_outputs.append(ch_out.unsqueeze(0))
            print(f"  Channel {ch_idx+1}/{n_channels} done")
        output = torch.cat(ch_outputs, dim=0)

    else:
        total_samples = st1_audio.shape[1]
        n_chunks = max(1, (total_samples - overlap_samples + step_samples - 1) // step_samples)
        print(f"  Chunked: {total_s:.1f}s -> {n_chunks} chunks x {n_channels}ch (overlap {overlap_s}s)")

        hann_full = torch.hann_window(max_chunk_samples)
        ch_outputs = []

        for ch_idx in range(n_channels):
            ch_audio = st1_audio[ch_idx:ch_idx+1]
            out_accum = torch.zeros(1, total_samples + max_chunk_samples, device='cpu')
            window_accum = torch.zeros(1, total_samples + max_chunk_samples)

            for i in range(n_chunks):
                start = i * step_samples
                end = min(start + max_chunk_samples, total_samples)
                actual_len = end - start

                chunk = ch_audio[:, start:end]
                if actual_len < max_chunk_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, max_chunk_samples - actual_len))

                chunk_mono = chunk.squeeze(0)
                ch_label = f"ch{ch_idx}_blk{i}"
                chunk_out = process_one_channel(chunk_mono, ch_label)

                if chunk_out.shape[0] > max_chunk_samples:
                    chunk_out = chunk_out[:max_chunk_samples]
                elif chunk_out.shape[0] < max_chunk_samples:
                    chunk_out = torch.nn.functional.pad(chunk_out, (0, max_chunk_samples - chunk_out.shape[0]))

                window = hann_full[:max_chunk_samples]
                out_accum[0, start:start + max_chunk_samples] += chunk_out * window
                window_accum[0, start:start + max_chunk_samples] += window
                print(f"  Ch {ch_idx+1}/{n_channels} #{i+1}/{n_chunks}: {start/48000:.1f}s - {end/48000:.1f}s")

            ch_out = out_accum / window_accum.clamp(min=1e-8)
            ch_out = ch_out[:, :total_samples]
            ch_outputs.append(ch_out)

        output = torch.cat(ch_outputs, dim=0)

    out_path = os.path.join(args.output, f"{base_name}_restored.wav")
    out_data = output.cpu().numpy()
    if out_data.ndim >= 2:
        out_data = out_data.T

    # Loudness matching: match Stage 2 output level to pre-Stage2 input
    if not args.skip_loudness_match:
        ref_np = st1_audio.cpu().numpy().T  # (samples, channels)
        meter = pyln.Meter(48000)
        lufs_in = meter.integrated_loudness(ref_np)
        lufs_out = meter.integrated_loudness(out_data)
        gain = 10 ** ((lufs_in - lufs_out) / 20)
        out_data = out_data * gain
        print(f"\n  Loudness: {lufs_in:.1f} LUFS → {lufs_out:.1f} LUFS, gain={gain:.3f} ({20*np.log10(gain):+.1f} dB)")

    sf.write(out_path, out_data.astype(np.float32), 48000, subtype='FLOAT')

    del audiosr_model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not args.keep_temp:
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"Restoration complete!")
    print(f"  Output: {out_path}")
    print(f"  Format: 48kHz 32-bit float WAV")
    print("=" * 60)


if __name__ == "__main__":
    main()
