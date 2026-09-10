#!/usr/bin/env python3
"""Train CHIMERA acoustic model. Example:
    python scripts/train.py --config configs/tiny_100m.yaml --max-items 2000 --max-steps 2000
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chimera.config import ChimeraConfig
from chimera.acoustic import ChimeraTTS
from chimera.data import get_loaders
from chimera.trainer import AcousticTrainer, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_100m.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--url", default="dev-clean")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = ChimeraConfig.from_yaml(args.config)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    seed_all(cfg.seed)

    device = torch.device(args.device)
    print(f"[train] {cfg.name} | target={cfg.target} | device={device}")
    print(f"[train] steps={cfg.max_steps} batch={cfg.batch_size} accum={cfg.grad_accum}")

    train_loader, val_loader = get_loaders(cfg, root=args.data_root, url=args.url,
                                           max_items=args.max_items)
    model = ChimeraTTS(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n/1e6:.1f}M")

    trainer = AcousticTrainer(cfg, model, train_loader, val_loader, device)
    if args.resume:
        trainer.load(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
