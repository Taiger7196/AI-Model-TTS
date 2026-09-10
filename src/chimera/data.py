"""Data: char tokenizer, mel frontend, LibriTTS loader, Griffin-Lim resynth.

Phase-0 keeps it simple on purpose:
- char-level tokenizer (upgrade to phoneme/BPE in phase-1)
- self-reference crops (upgrade to cross-utterance same-speaker refs in phase-1)
- Griffin-Lim waveform (upgrade to trained HiFi-GAN in phase-1)

Turbo: preprocessed items are cached to disk (.pt) — first epoch pays,
every later run/epoch loads instantly.
"""
from __future__ import annotations

import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    CHARSET = " abcdefghijklmnopqrstuvwxyz0123456789'-,.?!;:"

    def __init__(self):
        self.stoi = {c: i + 4 for i, c in enumerate(self.CHARSET)}
        self.itos = {i + 4: c for i, c in enumerate(self.CHARSET)}

    @property
    def vocab_size(self) -> int:
        return len(self.CHARSET) + 4

    def encode(self, text: str, max_len: int = 256) -> list[int]:
        ids = [self.BOS]
        for ch in text.lower():
            ids.append(self.stoi.get(ch, self.UNK))
        ids.append(self.EOS)
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            out.append(self.itos.get(i, "?"))
        return "".join(out)


class MelFrontend(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mel, power=2.0,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [T] or [B,T] at cfg.sample_rate -> log-power mel [n_mel, T'] / [B,n_mel,T']."""
        m = self.mel(wav)
        return torch.log(torch.clamp(m, min=1e-5))


class GriffinLimVocoder:
    """Zero-training waveform resynth. Ugly but proves the pipeline end-to-end."""

    def __init__(self, cfg):
        self.inv = torchaudio.transforms.InverseMelScale(
            n_mels=cfg.n_mel, sample_rate=cfg.sample_rate, n_stft=cfg.n_fft // 2 + 1)
        self.gl = torchaudio.transforms.GriffinLim(
            n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_iter=32)

    @torch.no_grad()
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel [B,n_mel,T] -> wav [B,Twav]."""
        linear = self.inv(torch.exp(mel))
        return self.gl(linear)


class LibriTTSDataset(Dataset):
    def __init__(self, cfg, root: str = "data", url: str = "dev-clean",
                 download: bool = True, max_items: int | None = None,
                 ref_sec: float = 3.0):
        super().__init__()
        self.cfg = cfg
        self.tok = CharTokenizer()
        self.front = MelFrontend(cfg)
        os.makedirs(root, exist_ok=True)  # torchaudio won't create root for download
        self.ds = torchaudio.datasets.LIBRITTS(root=root, url=url, download=download)
        mi = cfg.max_items if max_items is None else max_items
        self.n = len(self.ds) if mi is None else min(len(self.ds), mi)
        self.max_sec = cfg.max_sec
        self.ref_frames = int(ref_sec * cfg.sample_rate / cfg.hop_length)
        self.use_cache = cfg.cache_data
        self.cache_dir = os.path.join(
            root, "cache", f"{url}_sr{cfg.sample_rate}_mel{cfg.n_mel}_hop{cfg.hop_length}")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[data] LIBRITTS {url}: {len(self.ds)} utterances, using {self.n} "
              f"(max {self.max_sec}s, cache={'on' if self.use_cache else 'off'})")

    def __len__(self):
        return self.n

    def _process(self, i):
        wav, sr, text, *_ = self.ds[i]
        if not text or not text.strip():
            return None
        wav = wav.mean(dim=0)  # mono
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        if wav.numel() / self.cfg.sample_rate > self.max_sec:
            return None
        mel = self.front(wav)  # [n_mel, T]
        if mel.shape[1] < 32:
            return None
        # self-reference crop (phase-0) — forces the model to extract voice, not copy frames
        T = mel.shape[1]
        L = min(T, self.ref_frames)
        s = random.randint(0, T - L)
        ref = mel[:, s:s + L]
        ids = self.tok.encode(text, self.cfg.max_text_len)
        return {"text": torch.tensor(ids, dtype=torch.long),
                "mel": mel, "ref": ref}

    def __getitem__(self, i):
        if not self.use_cache:
            return self._process(i)
        p = os.path.join(self.cache_dir, f"{i}.pt")
        if os.path.exists(p):
            try:
                return torch.load(p, weights_only=True)
            except Exception:
                pass
        d = self._process(i)
        try:
            torch.save(d, p)
        except Exception:
            pass
        return d


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    texts = torch.nn.utils.rnn.pad_sequence(
        [b["text"] for b in batch], batch_first=True, padding_value=0)
    mels = torch.nn.utils.rnn.pad_sequence(
        [b["mel"].transpose(0, 1) for b in batch], batch_first=True,
        padding_value=-11.5).transpose(1, 2)  # log-floor padding
    refs = torch.nn.utils.rnn.pad_sequence(
        [b["ref"].transpose(0, 1) for b in batch], batch_first=True,
        padding_value=-11.5).transpose(1, 2)
    mel_len = torch.tensor([b["mel"].shape[1] for b in batch], dtype=torch.long)
    return {"text_ids": texts, "mel": mels, "mel_len": mel_len, "ref_mel": refs}


def get_loaders(cfg, root: str = "data", url: str = "dev-clean",
                max_items: int | None = None):
    mi = cfg.max_items if max_items is None else max_items
    train_ds = LibriTTSDataset(cfg, root=root, url=url, max_items=mi)
    val_ds = LibriTTSDataset(cfg, root=root, url=url, download=False, max_items=8)
    use_pin = cfg.pin_memory and torch.cuda.is_available()
    nw = cfg.num_workers
    loader_kw: dict = dict(batch_size=cfg.batch_size, shuffle=True, num_workers=nw,
                           collate_fn=collate, drop_last=True, pin_memory=use_pin)
    if nw > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=0, collate_fn=collate)
    return train_loader, val_loader


class TinyDemoDataset(Dataset):
    """Torch-only synthetic data for smoke tests (no downloads)."""

    def __init__(self, cfg, n: int = 32):
        self.cfg = cfg
        self.tok = CharTokenizer()
        self.n = n
        self.sentences = ["hello world this is a test",
                          "voice cloning from scratch",
                          "just cook and lock in",
                          "chimera speaks tonight"]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        T = random.randint(64, 160)
        mel = torch.randn(self.cfg.n_mel, T) * 2 - 4
        L = min(T, 48)
        s = random.randint(0, T - L)
        ids = self.tok.encode(random.choice(self.sentences), self.cfg.max_text_len)
        return {"text": torch.tensor(ids, dtype=torch.long), "mel": mel,
                "ref": mel[:, s:s + L]}
