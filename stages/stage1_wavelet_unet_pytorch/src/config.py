"""Model configuration matching the original upscalemp3_v2."""

SAMPLERATE = 44100
SEGMENT_LENGTH = 44100       # 1 second of 44.1kHz audio
NUM_LAYERS = 10              # wavelet decomposition levels
NUM_INIT_FILTERS = 16        # base channel count
FILTER_SIZE = 16             # Conv1D kernel size
WAVELET_FAMILY = "db4"       # Daubechies-4
MAX_SOURCES = 1              # mono restoration

# Derived constants
def num_filters_for_level(level: int) -> int:
    """Channel count for encoder/decoder level i (0-based)."""
    return NUM_INIT_FILTERS + NUM_INIT_FILTERS * level
