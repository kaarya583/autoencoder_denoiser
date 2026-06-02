"""Shared constants (aligned with Adaptive_Autoencoders_Project cell 12)."""

from pathlib import Path

SEED = 1
TARGET_SR = 16000
FRAME_SIZE = 512
HOP_SIZE = 512
LATENT_DIM_FAIR = 64

CHANNEL_NAMES = [
    "hiss",
    "pulse",
    "stepped",
    "warbler",
    "recorded",
    "spark",
]
NUM_FAMILIES = len(CHANNEL_NAMES)

MAX_TRAIN_FILES = 1200
MAX_TEST_FILES = 300
NOISE_CLF_N_PER_FAMILY = 3000

LIBRISPEECH_ARCHIVES = {
    "dev-clean.tar.gz": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean.tar.gz": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}


def data_root() -> Path:
    try:
        import google.colab  # noqa: F401

        return Path("/content")
    except ImportError:
        return Path.cwd() / "data" / "librispeech"
