"""CHIMERA acoustic model: text + reference voice -> mel (phase-0) or codec tokens (phase-2).

Training: teacher-forced next-frame prediction over the backbone's audio stream.
Inference: autoregressive generation with KV-cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ChimeraBackbone
from .speaker import SpeakerEncoder


class PostNet(nn.Module):
    """5-layer conv residual refiner (Tacotron-style). Cleans predicted mels."""

    def __init__(self, n_mel: int, d: int = 512, k: int = 5):
        super().__init__()
        layers = []
        for i in range(5):
            c_in = n_mel if i == 0 else d
            c_out = n_mel if i == 4 else d
            layers += [nn.Conv1d(c_in, c_out, k, padding=k // 2),
                       nn.BatchNorm1d(c_out),
                       nn.Tanh() if i < 4 else nn.Identity(),
                       nn.Dropout(0.1) if i < 4 else nn.Identity()]
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel + self.net(mel)  # [B, n_mel, T]


class MelHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, 256), nn.ReLU(),
            nn.Linear(256, cfg.n_mel),
        )
        self.stop = nn.Linear(cfg.d_model, 1)
        self.postnet = PostNet(cfg.n_mel)

    def forward(self, h: torch.Tensor):
        """h: [B,T,d] -> mel [B,T,n_mel], stop logits [B,T]."""
        return self.proj(h), self.stop(h).squeeze(-1)


class CodecHead(nn.Module):
    """One LM head per RVQ codebook level (phase-2)."""

    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.codebook_size) for _ in range(cfg.n_codebooks)]
        )

    def forward(self, h: torch.Tensor):
        return [head(h) for head in self.heads]  # list of [B,T,V]


def _length_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class ChimeraTTS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = ChimeraBackbone(cfg)
        self.speaker = SpeakerEncoder(cfg)
        self.go_mel = nn.Parameter(torch.zeros(cfg.n_mel))
        if cfg.target == "mel":
            self.mel_head = MelHead(cfg)
        elif cfg.target == "codec":
            self.codec_head = CodecHead(cfg)
        else:
            raise ValueError(f"unknown target {cfg.target}")

    @property
    def go_id(self) -> int:
        return self.cfg.codebook_size  # extra GO id (embedding has +1 row)

    # ---------------- training ----------------
    def forward(self, text_ids: torch.Tensor, ref_mel: torch.Tensor,
                mel: torch.Tensor | None = None, mel_len: torch.Tensor | None = None,
                codec: torch.Tensor | None = None):
        """Returns dict(loss, ...). mel: [B,n_mel,T], codec: [B,T,K]."""
        spk = self.speaker(ref_mel)
        use_ckpt = self.cfg.grad_checkpoint
        if self.cfg.target == "mel":
            assert mel is not None and mel_len is not None
            B, _, T = mel.shape
            go = self.go_mel.view(1, 1, -1).expand(B, 1, -1).to(mel.dtype)
            mel_in = torch.cat([go, mel.transpose(1, 2)[:, :-1, :]], dim=1)  # shift right
            h, _ = self.backbone(text_ids, mel_in, spk, use_ckpt)
            ha = h[:, -T:, :]
            mel_pred, stop = self.mel_head(ha)
            mel_t = mel.transpose(1, 2)
            mask = _length_mask(mel_len, T).to(mel.dtype)
            l1 = (F.l1_loss(mel_pred, mel_t, reduction="none").mean(-1) * mask).sum() / mask.sum()
            stop_tgt = torch.zeros_like(stop)
            stop_tgt[torch.arange(B, device=mel.device), (mel_len - 1).clamp(min=0)] = 1.0
            bce = F.binary_cross_entropy_with_logits(stop, stop_tgt, reduction="none")
            bce = (bce * mask).sum() / mask.sum()
            return {"loss": l1 + bce, "l1": l1.detach(), "stop": bce.detach(),
                    "mel_pred": mel_pred.detach()}
        else:
            assert codec is not None
            B, T, K = codec.shape
            go = torch.full((B, 1, K), self.go_id, device=codec.device, dtype=torch.long)
            codec_in = torch.cat([go, codec[:, :-1, :]], dim=1)
            h, _ = self.backbone(text_ids, codec_in, spk, use_ckpt)
            logits = self.codec_head(h[:, -T:, :])
            loss = 0.0
            for lvl, lg in enumerate(logits):
                loss = loss + F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                              codec[:, :, lvl].reshape(-1))
            loss = loss / len(logits)
            return {"loss": loss}

    # ---------------- inference ----------------
    @torch.no_grad()
    def generate_mel(self, text_ids: torch.Tensor, ref_mel: torch.Tensor,
                     max_len: int | None = None, stop_thresh: float = 0.5,
                     temp: float = 1.0) -> torch.Tensor:
        """AR mel generation. Returns [B, n_mel, T]."""
        self.eval()
        max_len = max_len or self.cfg.max_mel_len
        spk = self.speaker(ref_mel)
        B = text_ids.shape[0]
        go = self.go_mel.view(1, 1, -1).expand(B, 1, -1)
        h, caches = self.backbone.prefill(text_ids, go, spk)
        frames: list[torch.Tensor] = []
        for t in range(max_len):
            ha = h[:, -1:, :]
            m, s = self.mel_head(ha)
            if temp != 1.0:
                m = m * temp
            frames.append(m)
            if t > 10 and bool((s.sigmoid() > stop_thresh).all()):
                break
            h, caches = self.backbone.forward_step(m, caches)
        mel = torch.cat(frames, dim=1).transpose(1, 2)  # [B,n_mel,T]
        return self.mel_head.postnet(mel)

    @torch.no_grad()
    def generate_codec(self, text_ids: torch.Tensor, ref_mel: torch.Tensor,
                       max_len: int | None = None, temp: float = 0.9,
                       top_k: int = 50) -> torch.Tensor:
        """AR codec generation. Level-0 sampled, levels 1+ greedy. Returns [B,T,K]."""
        self.eval()
        max_len = max_len or self.cfg.max_codec_len
        spk = self.speaker(ref_mel)
        B = text_ids.shape[0]
        K = self.cfg.n_codebooks
        go = torch.full((B, 1, K), self.go_id, device=text_ids.device, dtype=torch.long)
        h, caches = self.backbone.prefill(text_ids, go, spk)
        out: list[torch.Tensor] = []
        for _ in range(max_len):
            logits = self.codec_head(h[:, -1:, :])  # list of [B,1,V]
            toks = []
            for lvl, lg in enumerate(logits):
                if lvl == 0 and temp > 0:
                    lg = lg / temp
                    if top_k:
                        v, _ = lg.topk(top_k, dim=-1)
                        lg = torch.where(lg < v[..., -1:], torch.full_like(lg, -1e9), lg)
                    tok = torch.multinomial(lg.softmax(-1).squeeze(1), 1).unsqueeze(1)
                else:
                    tok = lg.argmax(-1, keepdim=True)
                toks.append(tok)
            step = torch.cat(toks, dim=-1)  # [B,1,K]
            out.append(step)
            h, caches = self.backbone.forward_step(step, caches)
        return torch.cat(out, dim=1)
