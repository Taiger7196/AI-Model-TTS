#!/usr/bin/env python3
"""Parameter count for tiny + 3B configs (3B counted on meta device, no RAM needed)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chimera.config import tiny_preset, chimera_3b_preset
from chimera.acoustic import ChimeraTTS


def fmt(n: int) -> str:
    return f"{n/1e9:.3f}B" if n >= 1e9 else f"{n/1e6:.1f}M"


def report(cfg, meta: bool = False):
    ctx = torch.device("meta") if meta else torch.device("cpu")
    with ctx:
        model = ChimeraTTS(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n=== {cfg.name} (target={cfg.target}) ===")
    print(f"  backbone : {fmt(sum(p.numel() for p in model.backbone.parameters()))}")
    print(f"  speaker  : {fmt(sum(p.numel() for p in model.speaker.parameters()))}")
    head = model.mel_head if cfg.target == "mel" else model.codec_head
    print(f"  head     : {fmt(sum(p.numel() for p in head.parameters()))}")
    print(f"  go_mel   : {fmt(model.go_mel.numel())}")
    print(f"  TOTAL    : {fmt(total)}")


if __name__ == "__main__":
    report(tiny_preset())
    report(chimera_3b_preset(), meta=True)
