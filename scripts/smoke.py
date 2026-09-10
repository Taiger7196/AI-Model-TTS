#!/usr/bin/env python3
"""Smoke test: 4 training steps on synthetic data. No downloads, ~1 min.
Run: python scripts/smoke.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chimera import ChimeraConfig, ChimeraTTS
from chimera.data import TinyDemoDataset, collate
from chimera.trainer import AcousticTrainer, seed_all


def main():
    seed_all(0)
    cfg = ChimeraConfig(max_steps=4, batch_size=2, grad_accum=2, log_every=1,
                        ckpt_every=100, sample_every=100, out_dir="/tmp/chim_smoke")
    loader = DataLoader(TinyDemoDataset(cfg, n=16), batch_size=2, collate_fn=collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    AcousticTrainer(cfg, ChimeraTTS(cfg), loader, loader, dev).train()
    print("SMOKE OK 🔥")


if __name__ == "__main__":
    main()
