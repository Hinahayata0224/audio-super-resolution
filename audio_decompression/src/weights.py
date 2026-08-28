"""Load Keras model_13M.weights.h5 into PyTorch WaveletUNet — zero TF dependency."""
import os
import h5py
import torch
import numpy as np
from .model import WaveletUNet
from .config import NUM_LAYERS, num_filters_for_level

# Default weights path relative to this file's directory
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "model_13M.weights.h5")


def load_model(weights_path=None, device="cpu"):
    """Load WaveletUNet with pretrained weights.

    Args:
        weights_path: Path to .weights.h5 file. Uses built-in weights if None.
        device: torch device string.
    Returns:
        WaveletUNet in eval mode with loaded weights.
    """
    model = WaveletUNet().to(device)
    model.eval()
    path = weights_path or _DEFAULT_WEIGHTS
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    load_keras_weights(model, path)
    return model


def _suffix(i: int) -> str:
    """Keras layer naming: '' for first, '_1' for second, '_N' for N+1-th."""
    return "" if i == 0 else f"_{i}"


def load_keras_weights(model, h5_path):
    """Load weights from HDF5 file directly — by-name mapping, no TensorFlow."""
    with h5py.File(h5_path, "r") as f:
        raw = {}
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                raw[name] = obj[()]
        f.visititems(collect)

    # Helper: get array from HDF5
    def _arr(path):
        return torch.from_numpy(raw.get(path, raw.get(path.replace("/vars/0", "/vars/0"), np.zeros(1))))

    def _t(arr):
        """TF conv [k, in, out] -> PT [out, in, k]."""
        if arr.ndim == 3:
            arr = arr.transpose(2, 1, 0)
        return torch.from_numpy(arr.copy())

    def _set(param, key0, key1=None):
        """Set param from raw[key0] (conv) or raw[key0] and raw[key1] (LayerNorm)."""
        if param is None:
            return
        if key1 is None:
            # Single tensor (bias or gate bias)
            val = raw.get(key0)
            if val is not None:
                param.data.copy_(torch.from_numpy(val))
        else:
            # Conv kernel [k,in,out] + bias
            k = raw.get(key0)
            b = raw.get(key1)
            if k is not None:
                param.data.copy_(_t(k))
            if b is not None and hasattr(param, 'bias') and param.bias is not None:
                # Actually, weight and bias are separate params
                pass

    state = model.state_dict()

    def _w(pt_key, h5_base, sub, conv=True):
        """Load weight (and optionally bias) from HDF5."""
        w = raw.get(f"{h5_base}/{sub}/vars/0")
        if w is None:
            return
        if conv and w.ndim == 3:
            state[pt_key].data.copy_(_t(w))
        elif conv:
            state[pt_key].data.copy_(torch.from_numpy(w))
        else:
            state[pt_key].data.copy_(torch.from_numpy(w))

    def _b(pt_key, h5_base, sub):
        b = raw.get(f"{h5_base}/{sub}/vars/1")
        if b is not None:
            state[pt_key].data.copy_(torch.from_numpy(b))

    def _conv(pt_prefix, h5_base, sub):
        _w(f"{pt_prefix}.weight", h5_base, sub)
        _b(f"{pt_prefix}.bias", h5_base, sub)

    def _ln(pt_prefix, h5_base, sub):
        g = raw.get(f"{h5_base}/{sub}/vars/0")
        b = raw.get(f"{h5_base}/{sub}/vars/1")
        if g is not None:
            state[f"{pt_prefix}.weight"].data.copy_(torch.from_numpy(g))
        if b is not None:
            state[f"{pt_prefix}.bias"].data.copy_(torch.from_numpy(b))

    # ── Initial (flat structure: initial_conv/vars/0, no sub-layer) ──
    # Direct load because these layers have no sub-path in HDF5
    def _flat_w(pt_key, h5_base):
        w = raw.get(f"{h5_base}/vars/0")
        b = raw.get(f"{h5_base}/vars/1")
        if w is not None and w.ndim == 3:
            state[f"{pt_key}.weight"].data.copy_(_t(w))
        if b is not None:
            state[f"{pt_key}.bias"].data.copy_(torch.from_numpy(b))

    def _flat_ln(pt_key, h5_base):
        g = raw.get(f"{h5_base}/vars/0")
        b = raw.get(f"{h5_base}/vars/1")
        if g is not None:
            state[f"{pt_key}.weight"].data.copy_(torch.from_numpy(g))
        if b is not None:
            state[f"{pt_key}.bias"].data.copy_(torch.from_numpy(b))

    _flat_w("initial_conv", "initial_conv")
    _flat_ln("initial_norm", "initial_norm")

    # ── Encoder ds_blocks ──
    for i in range(NUM_LAYERS):
        s = _suffix(i)
        base = f"downsampling_blocks/downsampling_layer{s}"
        _conv(f"ds_blocks.{i}.conv", base, "conv")
        _ln(f"ds_blocks.{i}.norm", base, "layer_norm")

    # ── Encoder down_process ──
    for i in range(NUM_LAYERS):
        s = _suffix(i)
        base = f"down_process_blocks/upsampling_layer{s}"
        p = f"down_process.{i}"
        for sub in ["approx_conv", "detail_conv", "approx_gate", "detail_gate", "output_conv"]:
            _conv(f"{p}.{sub}", base, sub)
        for norm in ["approx_norm", "detail_norm", "output_norm"]:
            _ln(f"{p}.{norm}", base, norm)

    # ── Bottleneck ──
    _conv("bottleneck_conv", "bottle_neck/layers", "conv1d")
    _ln("bottleneck_norm", "bottle_neck/layers", "layer_normalization")

    # ── Decoder up_process (indices 10..19) ──
    for i in range(NUM_LAYERS):
        idx = 10 + i
        base = f"layers/upsampling_layer_{idx}"
        p = f"up_process.{i}"
        for sub in ["approx_conv", "detail_conv", "approx_gate", "detail_gate", "output_conv"]:
            _conv(f"{p}.{sub}", base, sub)
        for norm in ["approx_norm", "detail_norm", "output_norm"]:
            _ln(f"{p}.{norm}", base, norm)

    # ── Decoder skip_connections ──
    for i in range(NUM_LAYERS):
        s = _suffix(i)
        base = f"layers/gated_skip_connection{s}"
        p = f"skip_connections.{i}"
        _conv(f"{p}.decoder_gate", base, "decoder_gate")
        _conv(f"{p}.encoder_gate", base, "encoder_gate")
        _ln(f"{p}.norm", base, "norm")

    # ── Decoder up_blocks (sequential_1..sequential_10) ──
    for i in range(NUM_LAYERS):
        seq = i + 1
        base = f"layers/sequential_{seq}/layers"
        p = f"up_blocks.{i}"
        _conv(f"{p}.0", base, "conv1d")
        _ln(f"{p}.1.norm", base, "layer_normalization")
        _conv(f"{p}.2", base, "conv1d_1")
        _ln(f"{p}.3.norm", base, "layer_normalization_1")

    # ── Final conv ──
    base = "final_conv/layers"
    _conv("final_conv.0", base, "conv1d")
    _ln("final_conv.1.norm", base, "layer_normalization")
    _conv("final_conv.2", base, "conv1d_1")
    _ln("final_conv.3.norm", base, "layer_normalization_1")

    # ── Output conv ──
    _conv("output_conv", "layers", "conv1d_1")

    # Count loaded
    loaded = 0
    for k, v in state.items():
        before = v.clone().sum().item()
    # Actually check non-default
    param_count = sum(1 for _ in state.values())
    missing = 0
    for k, v in state.items():
        # Check if sum is non-zero (or close to default init)
        s = v.abs().sum().item()
        if s < 1e-6 and "norm" not in k and "bias" not in k:
            missing += 1
    if missing > 0:
        print(f"  Warning: {missing}/{param_count} params appear uninitialized")

    return model
