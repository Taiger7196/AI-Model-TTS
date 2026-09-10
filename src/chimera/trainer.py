"""Training loop: AMP + grad accumulation + warmup/cosine + checkpoint resume + audio samples."""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import GriffinLimVocoder


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(opt, cfg):
    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        prog = min(max(prog, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class AcousticTrainer:
    def __init__(self, cfg, model: nn.Module, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        self.sched = build_scheduler(self.opt, cfg)
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.vocoder = GriffinLimVocoder(cfg)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "samples").mkdir(exist_ok=True)
        self.step = 0

    def save(self, name: str | None = None):
        path = self.out / (name or f"step_{self.step:06d}.pt")
        torch.save({"step": self.step, "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                    "scaler": self.scaler.state_dict()}, path)
        print(f"[ckpt] saved {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step = ck["step"]
        print(f"[ckpt] resumed from {path} @ step {self.step}")

    @torch.no_grad()
    def sample(self):
        """Generate one fixed validation sample so you can HEAR overfitting happen."""
        import soundfile as sf

        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            if batch is None:
                return
            text_ids = batch["text_ids"][:1].to(self.device)
            ref = batch["ref_mel"][:1].to(self.device)
            tgt = batch["mel"][:1]
            gen = self.model.generate_mel(text_ids, ref, max_len=400)
            wav_g = self.vocoder(gen.cpu()).squeeze(0).numpy()
            wav_t = self.vocoder(tgt).squeeze(0).numpy()
            sr = self.cfg.sample_rate
            sf.write(self.out / "samples" / f"step_{self.step:06d}_gen.wav", wav_g, sr)
            sf.write(self.out / "samples" / f"step_{self.step:06d}_tgt.wav", wav_t, sr)
            print(f"[sample] wrote step_{self.step:06d}_*.wav "
                  f"(gen {wav_g.shape[0]/sr:.1f}s)")
        except Exception as e:
            print(f"[sample] skipped: {e}")
        finally:
            self.model.train()

    def train(self):
        cfg = self.cfg
        self.model.train()
        it = iter(self.train_loader)
        self.opt.zero_grad(set_to_none=True)
        while self.step < cfg.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            if batch is None:
                continue
            text_ids = batch["text_ids"].to(self.device)
            mel = batch["mel"].to(self.device)
            mel_len = batch["mel_len"].to(self.device)
            ref = batch["ref_mel"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(text_ids, ref, mel=mel, mel_len=mel_len)
                loss = out["loss"] / cfg.grad_accum
            self.scaler.scale(loss).backward()

            if (self.step + 1) % cfg.grad_accum == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)

            if self.step % cfg.log_every == 0:
                l1 = float(out.get("l1", out["loss"]))
                st = float(out.get("stop", 0.0))
                lr = self.sched.get_last_lr()[0]
                print(f"[step {self.step:06d}] loss={float(out['loss'].detach()):.4f} "
                      f"l1={l1:.4f} stop={st:.4f} lr={lr:.2e}", flush=True)

            self.step += 1
            if self.step % cfg.ckpt_every == 0:
                self.save()
            if self.step % cfg.sample_every == 0:
                self.sample()
        self.save("final.pt")
        print("[done] training complete. Just cooked. 🔥")
