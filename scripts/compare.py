"""Quantitative comparison between original and restored audio.
Matches loudness before computing LSD (level-independent but gain offsets inflate per-band values)."""
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

def mel_spectrogram(y, sr, n_fft=2048, hop=480, n_mels=256, fmax=24000):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    return np.dot(mel, S)

def log_spec(mel_spec):
    return np.log(np.clip(mel_spec, 1e-5, None))

def lsd(ref_mel, deg_mel):
    return np.sqrt(np.mean((log_spec(ref_mel) - log_spec(deg_mel)) ** 2))

def lsd_per_band(ref_mel, deg_mel):
    return np.sqrt(np.mean((log_spec(ref_mel) - log_spec(deg_mel)) ** 2, axis=1))

def loudness_match(ref, deg, sr):
    """Attenuate / amplify ref to match deg's integrated LUFS."""
    meter_ref = pyln.Meter(sr)
    meter_deg = pyln.Meter(sr)
    l_ref = meter_ref.integrated_loudness(ref)
    l_deg = meter_deg.integrated_loudness(deg)
    gain = 10 ** ((l_deg - l_ref) / 20)
    return ref * gain, l_ref, l_deg

def main():
    PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) >= 3:
        ref_path, deg_path = sys.argv[1], sys.argv[2]
    else:
        ref_path = os.path.join(PROJECT, "audio", "test_30s.wav")
        deg_path = os.path.join(PROJECT, "output", "test_30s_restored.wav")

    ref, sr_ref = sf.read(ref_path)
    deg, sr_daq = sf.read(deg_path)

    # Convert to mono
    if ref.ndim > 1:
        ref = np.mean(ref, axis=1)
    if deg.ndim > 1:
        deg = np.mean(deg, axis=1)

    # Match sample rates
    target_sr = max(sr_ref, sr_daq)
    if sr_ref != target_sr:
        ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=target_sr)
    if sr_daq != target_sr:
        deg = librosa.resample(deg, orig_sr=sr_daq, target_sr=target_sr)

    # Loudness-match reference to degraded
    ref_matched, lufs_ref, lufs_deg = loudness_match(ref, deg, target_sr)

    # Match lengths
    min_len = min(len(ref_matched), len(deg))
    ref_matched, deg = ref_matched[:min_len], deg[:min_len]

    # Mel specs
    mel_ref = mel_spectrogram(ref_matched, target_sr)
    mel_deg = mel_spectrogram(deg, target_sr)
    min_frames = min(mel_ref.shape[1], mel_deg.shape[1])
    mel_ref, mel_deg = mel_ref[:, :min_frames], mel_deg[:, :min_frames]

    # Metrics
    overall_lsd = lsd(mel_ref, mel_deg)
    band_lsd = lsd_per_band(mel_ref, mel_deg)

    def hz_to_bin(hz, n_mels=256, fmax=24000):
        mel_freq = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax)
        return np.searchsorted(mel_freq, hz)

    lo, mi, hi = hz_to_bin(4000), hz_to_bin(8000), hz_to_bin(16000)
    lsd_low  = np.sqrt(np.mean(band_lsd[:lo] ** 2))
    lsd_mid  = np.sqrt(np.mean(band_lsd[lo:mi] ** 2))
    lsd_high = np.sqrt(np.mean(band_lsd[mi:]  ** 2))

    e_ref = np.sum(mel_ref ** 2, axis=1)
    e_deg = np.sum(mel_deg ** 2, axis=1)
    e_ratio = e_deg / (e_ref + 1e-10)
    e_low  = np.mean(e_ratio[:lo])
    e_mid  = np.mean(e_ratio[lo:mi])
    e_high = np.mean(e_ratio[mi:])

    peak_ref = 20 * np.log10(np.max(np.abs(ref_matched)) + 1e-10)
    peak_deg = 20 * np.log10(np.max(np.abs(deg)) + 1e-10)
    cent_ref = librosa.feature.spectral_centroid(y=ref_matched, sr=target_sr).mean()
    cent_deg = librosa.feature.spectral_centroid(y=deg, sr=target_sr).mean()
    roll_ref = librosa.feature.spectral_rolloff(y=ref_matched, sr=target_sr, roll_percent=0.90).mean()
    roll_deg = librosa.feature.spectral_rolloff(y=deg, sr=target_sr, roll_percent=0.90).mean()

    print("=" * 55)
    print(f"  Reference:  {os.path.basename(ref_path)}")
    print(f"  Degraded:   {os.path.basename(deg_path)}")
    print("-" * 55)
    print(f"  Orig LUFS:  {lufs_ref:.1f} → {lufs_deg:.1f}  (Δ = {lufs_deg-lufs_ref:+.1f}; ref matched)")
    print("-" * 55)
    print(f"  Duration:   {min_len/target_sr:.1f}s @ {target_sr}Hz")
    print(f"  Peak:        {peak_ref:.1f} dBFS  →  {peak_deg:.1f} dBFS")
    print(f"  Spec.Centroid: {cent_ref:.0f} Hz  →  {cent_deg:.0f} Hz  ({cent_deg-cent_ref:+.0f})")
    print(f"  Roll-off(90%): {roll_ref:.0f} Hz  →  {roll_deg:.0f} Hz  ({roll_deg-roll_ref:+.0f})")
    print("-" * 55)
    print(f"  LSD overall: {overall_lsd:.2f} dB  (lower = closer)")
    print(f"  LSD  <4kHz:  {lsd_low:.2f} dB")
    print(f"  LSD 4-8kHz: {lsd_mid:.2f} dB")
    print(f"  LSD >8kHz:  {lsd_high:.2f} dB")
    print("-" * 55)
    print(f"  Energy ratio (<4k / 4-8k / >8k):")
    print(f"    {e_low:.2f}x  /  {e_mid:.2f}x  /  {e_high:.1f}x")
    print("-" * 55)

    # ── High-frequency quality (no reference needed) ──
    # Use librosa's STFT for full-band spectral analysis
    S = np.abs(librosa.stft(deg, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=4096)
    hf_mask = freqs >= 8000

    # 1. Spectral Crest Factor per band (tonal vs noise-like)
    fft_lo = np.searchsorted(freqs, 4000)
    fft_hi = np.searchsorted(freqs, 8000)
    scf_all = 20 * np.log10(np.max(S, axis=1) / (np.mean(S, axis=1) + 1e-10))
    scf_low  = np.mean(scf_all[:fft_lo])
    scf_high = np.mean(scf_all[fft_hi:])  # >8kHz
    print(f"  HF tonality: {scf_high:.1f} dB crest  (ref low-band: {scf_low:.1f} dB)")
    print(f"    >15dB = clearly harmonic, 10-15 = mixed, <10 = noise-like")

    # 2. Spectral flatness (0 = pure tone, 1 = white noise)
    gmean = np.exp(np.mean(np.log(S + 1e-10), axis=1))
    amean = np.mean(S, axis=1)
    sfm = gmean / (amean + 1e-10)
    sfm_high = np.mean(sfm[hf_mask])
    print(f"  HF flatness: {sfm_high:.3f}  (0=tonal, 1=white noise)")

    # 3. High-frequency harmonicity via comb filtering
    #    Build a comb of fundamental harmonics, correlate with actual spectrum
    spec_proc = np.mean(S, axis=1)  # average spectrum over time
    hf_spec = spec_proc[hf_mask]
    # Autocorrelation of HF spectrum → peak spacing reveals harmonicity
    hf_ac = np.correlate(hf_spec - np.mean(hf_spec),
                         hf_spec - np.mean(hf_spec), mode='full')
    hf_ac = hf_ac[len(hf_ac)//2:]  # keep positive lags
    hf_ac_norm = hf_ac / (hf_ac[0] + 1e-10)
    # Find strongest secondary peak (first lag > 0)
    peaks = []
    for k in range(3, min(len(hf_ac_norm) - 1, 60)):
        if hf_ac_norm[k] > hf_ac_norm[k-1] and hf_ac_norm[k] > hf_ac_norm[k+1]:
            peaks.append((k, hf_ac_norm[k]))
    if peaks:
        best_peak = max(peaks, key=lambda x: x[1])
        harm_spacing_hz = best_peak[0] * target_sr / 4096
        harm_strength = best_peak[1]
        print(f"  HF harmonicity: {harm_strength:.2f} (peak @ {harm_spacing_hz:.0f} Hz spacing)")
        print(f"    >0.3 = clear harmonics, <0.1 = unstructured HF")
    else:
        print(f"  HF harmonicity: no harmonic peaks detected")

    # 4. HF band SNR estimate: 90th percentile / 10th percentile energy
    band_energy = np.sum(S[hf_mask, :] ** 2, axis=0)
    p10, p50, p90 = np.percentile(band_energy, [10, 50, 90])
    hf_snr = 10 * np.log10(p90 / (p10 + 1e-10))
    print(f"  HF SNR estimate: {hf_snr:.1f} dB (P90/P10 dynamic range)")

    print("-" * 55)
    print(f"  Paper ref: LSD 0.61-0.84 (GT-Mel), 0.74 (AudioSR music 8k)")
    print("=" * 55)

if __name__ == "__main__":
    main()
