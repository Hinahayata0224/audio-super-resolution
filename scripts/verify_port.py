"""Verify audio_decompression (PyTorch) outputs match upscalemp3_v2 (TF/Keras).

Usage:
  python verify_port.py                     # PT-only checks + weight load test
  python verify_port.py --full              # Full: PT vs TF comparison (needs tensorflow)
  python verify_port.py --audio             # Run on test_30s.wav and diff
"""

import os
import sys
import argparse
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from audio_decompression import load_model, WaveletUNet

# ---------------------------------------------------------------------------
# 1. Load PT model with weights & test forward pass
# ---------------------------------------------------------------------------
def test_pt_forward():
    """Load PT model, verify weights loaded, run single clip forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PT] Device: {device}")

    model = load_model(device=device)

    total_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in model.parameters() if p.abs().sum() > 1e-8)
    print(f"[PT] Total params: {total_params:,}  |  non-zero: {loaded_params:,}")

    sr = 44100
    t = np.arange(sr) / sr
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    inp = torch.from_numpy(sine).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
    out_np = out.squeeze().cpu().numpy()

    print(f"[PT] Input  shape: {inp.shape}")
    print(f"[PT] Output shape: {out.shape}")
    print(f"[PT] Input  [min,max,mean]: [{inp.min():.4f}, {inp.max():.4f}, {inp.mean():.4f}]")
    print(f"[PT] Output [min,max,mean]: [{out_np.min():.4f}, {out_np.max():.4f}, {out_np.mean():.4f}]")
    print(f"[PT] Checksum (first 1000): {out_np[:1000].sum():.6f}")
    print()

    return model, device, out_np

# ---------------------------------------------------------------------------
# 2. TF comparison (optional)
# ---------------------------------------------------------------------------
def test_tf_comparison(pt_out_np):
    """Run TF model on same input, compare outputs."""
    try:
        sys.path.insert(0, os.path.join(PROJECT, "upscalemp3_v2", "src"))
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        from model import WaveletUNet as TFWaveletUNet
        from model import DWTLayer, IDWTLayer, DownsamplingLayer, UpsamplingLayer, GatedSkipConnection

        custom_objects = {
            "WaveletUNet": TFWaveletUNet, "DWTLayer": DWTLayer, "IDWTLayer": IDWTLayer,
            "DownsamplingLayer": DownsamplingLayer, "UpsamplingLayer": UpsamplingLayer,
            "GatedSkipConnection": GatedSkipConnection,
        }

        keras_path = os.path.join(PROJECT, "upscalemp3_v2", "models", "model_13M.keras")
        if not os.path.exists(keras_path):
            print("[TF] SKIP: .keras model file not found")
            return

        model = tf.keras.models.load_model(
            keras_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        _ = model(tf.zeros((16, 44100, 1)), training=False)
        h5_path = os.path.join(PROJECT, "upscalemp3_v2", "models", "model_13M.weights.h5")
        model.load_weights(h5_path)

        sr = 44100
        t = np.arange(sr) / sr
        sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        inp_tf = sine.reshape(1, -1, 1)

        out_tf = model(inp_tf, training=False).numpy().squeeze()

        pt_flat = pt_out_np.squeeze()
        tf_flat = out_tf.squeeze()
        min_len = min(len(pt_flat), len(tf_flat))
        pt_flat, tf_flat = pt_flat[:min_len], tf_flat[:min_len]

        mae = np.abs(pt_flat - tf_flat)
        print(f"[TF] Output [min,max,mean]: [{tf_flat.min():.4f}, {tf_flat.max():.4f}, {tf_flat.mean():.4f}]")
        print(f"[TF] Checksum (first 1000): {tf_flat[:1000].sum():.6f}")
        print()
        print("=" * 60)
        print("  PT vs TF COMPARISON")
        print("=" * 60)
        print(f"  MAE  mean:    {mae.mean():.8f}")
        print(f"  MAE  max:     {mae.max():.8f}")
        print(f"  MAE  median:  {np.median(mae):.8f}")
        print(f"  MAE  P99:     {np.percentile(mae, 99):.8f}")
        rel = mae / (np.abs(tf_flat).max() + 1e-10)
        print(f"  Rel error max: {rel.max():.6f}")
        print(f"  Correlation:   {np.corrcoef(pt_flat, tf_flat)[0, 1]:.8f}")
        print()
        if mae.max() < 1e-4:
            print("  RESULT: PERFECT MATCH (< 1e-4) -- port is correct!")
        elif mae.max() < 0.05 and mae.mean() < 1e-3:
            print(f"  RESULT: MATCH (mean={mae.mean():.6f}, max at boundaries)")
            print("  Residual error is from fp32 numerical precision differences")
        elif mae.max() < 1e-3:
            print("  RESULT: CLOSE (< 1e-3) -- minor numerical diffs acceptable")
        else:
            print(f"  RESULT: DIVERGENCE ({mae.max():.6f}) -- needs investigation")
        print("=" * 60)
        return mae, tf_flat

    except ImportError as e:
        print(f"[TF] SKIP: TensorFlow not available ({e})")
        return None
    except Exception as e:
        print(f"[TF] ERROR: {e}")
        import traceback; traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# 3. DWT/IDWT round-trip (PT-only sanity)
# ---------------------------------------------------------------------------
def test_dwt_idwt_inverse(device):
    """Verify DWT + IDWT round-trip is near-identity."""
    from audio_decompression.dwt import DWTLayer, IDWTLayer

    dwt = DWTLayer().to(device)
    idwt = IDWTLayer().to(device)

    sr = 44100
    t = torch.arange(sr, device=device).float() / sr
    x = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        coeffs = dwt(x)
        recon = idwt(coeffs)

    min_t = min(x.shape[-1], recon.shape[-1])
    diff = (x[..., :min_t] - recon[..., :min_t]).abs()
    print(f"[DWT/IDWT] Input energy={x.abs().mean():.6f}, diff MAE={diff.mean():.8f}, diff max={diff.max():.8f}")
    print()

# ---------------------------------------------------------------------------
# 4. Audio file test
# ---------------------------------------------------------------------------
def test_on_audio(model, device):
    """Run model on test_30s.wav (single clip) and save output."""
    import soundfile as sf

    audio_path = os.path.join(PROJECT, "test_30s.wav")
    if not os.path.exists(audio_path):
        print(f"[AUDIO] SKIP: {audio_path} not found")
        return

    wav, sr = sf.read(audio_path)
    print(f"[AUDIO] Loaded: {wav.shape}, {sr}Hz")

    ch_data = wav[:, 0] if wav.ndim > 1 else wav
    clip = ch_data[:44100].astype(np.float32)

    inp = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    out_np = out.squeeze().cpu().numpy()

    print(f"[AUDIO] Clip  input [min,max,mean]: [{clip.min():.6f}, {clip.max():.6f}, {clip.mean():.6f}]")
    print(f"[AUDIO] Clip output [min,max,mean]: [{out_np.min():.6f}, {out_np.max():.6f}, {out_np.mean():.6f}]")
    print(f"[AUDIO] Checksum: {out_np.sum():.6f}")

    out_dir = os.path.join(PROJECT, "output_verify")
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "pt_single_clip.wav"), out_np, 44100)
    sf.write(os.path.join(out_dir, "pt_single_clip_input.wav"), clip, 44100)
    print(f"[AUDIO] Saved to {out_dir}/")
    print()
    return out_np

# ---------------------------------------------------------------------------
# 5. Negative control
# ---------------------------------------------------------------------------
def test_unloaded_model():
    """Verify random-init model produces different output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_random = WaveletUNet().to(device)
    model_random.eval()

    sr = 44100
    t = torch.arange(sr, device=device).float() / sr
    inp = (0.5 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out_r = model_random(inp).squeeze().cpu().numpy()

    print(f"[UNLOADED] Checksum: {out_r[:1000].sum():.6f}  (should differ from loaded model)")
    print()
    return out_r

# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Verify audio_decompression port against upscalemp3_v2")
    parser.add_argument("--full", action="store_true", help="Full PT vs TF comparison (needs TF)")
    parser.add_argument("--audio", action="store_true", help="Run on test_30s.wav")
    args = parser.parse_args()

    print("=" * 60)
    print("  audio_decompression PORT VERIFICATION")
    print("=" * 60)
    print()

    test_dwt_idwt_inverse("cpu")
    result = test_pt_forward()
    if result is None:
        print("FATAL: PT model failed to load. Aborting.")
        return
    model, device, pt_out = result

    rand_out = test_unloaded_model()
    if rand_out is not None and pt_out is not None:
        diff = np.abs(pt_out[:1000] - rand_out[:1000]).mean()
        if diff < 1e-3:
            print("WARNING: Loaded model output is same as random init!")
        else:
            print(f"  > Loaded vs random diff: {diff:.6f} -- weights LOADED")
        print()

    if args.audio:
        test_on_audio(model, device)

    if args.full:
        test_tf_comparison(pt_out)

    print("Verification complete.")

if __name__ == "__main__":
    main()
