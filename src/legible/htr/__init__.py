"""Phase 1: handwriting text recognition (CNN-ViT hybrid) for hard-to-read handwriting."""

from .config import Config, ModelConfig, load_config
from .vocab import Vocab, normalize_text

__all__ = ["Config", "ModelConfig", "Vocab", "load_config", "normalize_text"]
