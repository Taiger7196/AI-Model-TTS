"""CHIMERA-TTS: from-scratch voice-cloning TTS. Just cook."""
from .config import ChimeraConfig, tiny_preset, chimera_3b_preset
from .acoustic import ChimeraTTS

__all__ = ["ChimeraConfig", "tiny_preset", "chimera_3b_preset", "ChimeraTTS"]
