"""Compact HiFi-GAN vocoder (from scratch): mel -> waveform.

Phase-0 uses Griffin-Lim for instant pipeline validation (no vocoder training needed).
Train this GAN in phase-1 for real quality. Discriminators: MPD + MSD.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _pad(k: int, d: int = 1) -> int:
    return (k - 1) * d // 2


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [weight_norm(nn.Conv1d(ch, ch, k, 1, _pad(k, d), dilation=d)) for d in dilations])
        self.convs2 = nn.ModuleList(
            [weight_norm(nn.Conv1d(ch, ch, k, 1, _pad(k, 1))) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
        return x


class MRF(nn.Module):
    def __init__(self, ch: int, kernels=(3, 7, 11)):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(ch, k) for k in kernels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(b(x) for b in self.blocks) / len(self.blocks)


class HiFiGANGenerator(nn.Module):
    def __init__(self, n_mel: int = 80, base: int = 512,
                 ups=(8, 8, 2, 2), up_kernels=(16, 16, 4, 4)):
        super().__init__()
        assert __import__("math").prod(ups) == 256, "upsampling must equal hop_length 256"
        self.pre = weight_norm(nn.Conv1d(n_mel, base, 7, 1, 3))
        ch = base
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        for u, k in zip(ups, up_kernels):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, u, (k - u) // 2)))
            self.mrfs.append(MRF(ch // 2))
            ch //= 2
        self.post = weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = mrf(F.leaky_relu(up(F.leaky_relu(x, 0.1)), 0.1))
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chs = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(1 if i == 0 else chs[i - 1], c, (5, 1), (3, 1),
                                      padding=(2, 0))) for i, c in enumerate(chs)])
        self.out = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, wav: torch.Tensor):
        B, _, T = wav.shape
        if T % self.period:
            wav = F.pad(wav, (0, self.period - T % self.period))
        x = wav.view(B, 1, -1, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return self.out(x).flatten(1), feats


class MPD(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, wav: torch.Tensor):
        outs, feats = [], []
        for d in self.discs:
            o, f = d(wav)
            outs.append(o)
            feats.append(f)
        return outs, feats


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv1d(1, 128, 15, 1, 7))]
            + [weight_norm(nn.Conv1d(128 * 2**i, 128 * 2 ** (i + 1), 41, 4,
                                     padding=20, groups=4 if i else 1))
               for i in range(3)]
            + [weight_norm(nn.Conv1d(1024, 1024, 41, 4, padding=20, groups=16)),
               weight_norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
               weight_norm(nn.Conv1d(1024, 1, 3, 1, 1))])

    def forward(self, wav: torch.Tensor):
        feats = []
        x = wav
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return x.flatten(1), feats


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])

    def forward(self, wav: torch.Tensor):
        outs, feats = [], []
        x = wav
        for i, d in enumerate(self.discs):
            if i:
                x = self.pools[i - 1](x)
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        return outs, feats


def disc_hinge_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss = loss + F.relu(1 - r).mean() + F.relu(1 + f).mean()
    return loss


def gen_hinge_loss(fake_outs) -> torch.Tensor:
    return sum(-f.mean() for f in fake_outs)


def feature_matching_loss(real_feats, fake_feats) -> torch.Tensor:
    loss = 0.0
    n = 0
    for rf, ff in zip(real_feats, fake_feats):
        for r, f in zip(rf, ff):
            loss = loss + F.l1_loss(f, r.detach())
            n += 1
    return loss / max(n, 1)
