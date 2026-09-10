"""Config + presets for CHIMERA-TTS."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields


@dataclass
class ChimeraConfig:
    name: str = "chimera-tiny"
    target: str = "mel"  # "mel" (phase-0 prototype) | "codec" (phase-2 scale-up)

    # text
    vocab_size: int = 256
    max_text_len: int = 256

    # audio framing
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mel: int = 80
    max_mel_len: int = 768
    max_codec_len: int = 1024

    # backbone (decoder-only transformer, GQA + RoPE + SwiGLU)
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    grad_checkpoint: bool = True

    # speaker encoder
    spk_d_model: int = 256
    spk_layers: int = 4
    spk_heads: int = 4

    # codec head (phase-2)
    n_codebooks: int = 4
    codebook_size: int = 1024

    # performance (turbo)
    compile: bool = False
    cache_data: bool = True
    pin_memory: bool = True
    fused_opt: bool = True
    max_sec: float = 12.0
    sample_len: int = 400
    max_items: int | None = None

    # training
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20000
    batch_size: int = 8
    grad_accum: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 2
    log_every: int = 25
    ckpt_every: int = 1000
    sample_every: int = 1000
    out_dir: str = "checkpoints/chimera-tiny"
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str) -> "ChimeraConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_yaml(self, path: str):
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def tiny_preset() -> ChimeraConfig:
    """~85M params. Fits Colab T4 free. Prove the architecture here first."""
    return ChimeraConfig(name="chimera-tiny")


def chimera_3b_preset() -> ChimeraConfig:
    """~3B params. Same code, scaled config. Needs FSDP / multi-GPU — NOT single T4."""
    return ChimeraConfig(
        name="chimera-3b",
        target="codec",
        vocab_size=8192,
        max_text_len=512,
        max_mel_len=1024,
        max_codec_len=1536,
        d_model=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        d_ff=8192,
        spk_d_model=512,
        spk_layers=6,
        spk_heads=8,
        n_codebooks=8,
        codebook_size=2048,
        lr=1e-4,
        warmup_steps=2000,
        max_steps=500000,
        batch_size=4,
        grad_accum=16,
        num_workers=4,
        ckpt_every=2000,
        sample_every=2000,
        out_dir="checkpoints/chimera-3b",
    )
