"""CHIMERA backbone: decoder-only transformer with GQA + RoPE + SwiGLU + RMSNorm.

Sequence layout (single stream, like a codec-LM):
    [ SPK ] [ TEXT ... ] [ AUDIO ... ]
      1        Tt            Ta

AUDIO frames are either continuous mel frames (phase-0) or summed RVQ
codebook embeddings (phase-2). Modality bias vectors tell the model which is which.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps).type_as(x)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """NeoX-style RoPE. x: [..., L, D], freqs: [L, D/2]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    shape = [1] * (x.dim() - 2) + [freqs.shape[0], freqs.shape[1]]
    cos = cos.view(*shape)
    sin = sin.view(*shape)
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class GQACausalAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv_heads: int,
                 theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0 and n_heads % n_kv_heads == 0
        self.d = d
        self.h = n_heads
        self.kv = n_kv_heads
        self.hd = d // n_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.drop = dropout
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer("rope_inv", inv, persistent=False)

    def _freqs(self, offset: int, L: int, device, dtype) -> torch.Tensor:
        t = torch.arange(offset, offset + L, device=device).float()
        return torch.outer(t, self.rope_inv).to(dtype)

    def forward(self, x: torch.Tensor, cache=None):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)   # [B,H,L,D]
        k = self.k(x).view(B, L, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.kv, self.hd).transpose(1, 2)
        off = cache[0].shape[2] if cache is not None else 0
        freqs = self._freqs(off, L, x.device, x.dtype)
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        if self.kv != self.h:
            rep = self.h // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(cache is None),
            dropout_p=self.drop if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, L, self.d)
        return self.o(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GQACausalAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                                       cfg.rope_theta, cfg.dropout)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))[0]
        x = x + self.mlp(self.n2(x))
        return x

    def prefill(self, x: torch.Tensor):
        """Run over a full prefix once, returning output + KV-cache for stepwise decode."""
        a, cache = self.attn(self.n1(x), None)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, cache

    def forward(self, x: torch.Tensor, cache=None, use_ckpt: bool = False):
        if cache is not None:  # inference path, KV-cache
            a, new_cache = self.attn(self.n1(x), cache)
            x = x + a
            x = x + self.mlp(self.n2(x))
            return x, new_cache
        if use_ckpt and self.training:
            return checkpoint(self._core, x, use_reentrant=False), None
        return self._core(x), None


class ChimeraBackbone(nn.Module):
    """Embeds [spk, text, audio] and runs the decoder stack."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.frame_in = nn.Linear(cfg.n_mel, cfg.d_model)          # mel path
        self.code_emb = nn.Embedding(cfg.codebook_size + 1, cfg.d_model)  # codec path (+1 GO id)
        self.mod_text = nn.Parameter(torch.zeros(cfg.d_model))
        self.mod_audio = nn.Parameter(torch.zeros(cfg.d_model))
        self.mod_spk = nn.Parameter(torch.zeros(cfg.d_model))
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        import math

        res_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "weight" in name:
                if any(k in name for k in ("attn.o", "mlp.down", "tok_emb", "code_emb")):
                    nn.init.normal_(p, std=res_std if "emb" not in name else 0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.code_emb.weight, std=0.02)

    def _embed_audio(self, audio) -> torch.Tensor:
        if audio.dtype == torch.long:  # codec ids [B,Ta,K] -> sum of codebook embs
            return self.code_emb(audio).sum(dim=2) + self.mod_audio
        return self.frame_in(audio) + self.mod_audio  # mel [B,Ta,n_mel]

    def forward(self, text_ids: torch.Tensor, audio_in: torch.Tensor,
                spk: torch.Tensor, use_ckpt: bool = False):
        h_t = self.tok_emb(text_ids) + self.mod_text
        h_a = self._embed_audio(audio_in)
        h_s = (spk + self.mod_spk).unsqueeze(1)
        h = torch.cat([h_s, h_t, h_a], dim=1)
        for blk in self.layers:
            h, _ = blk(h, None, use_ckpt)
        return self.nf(h), None

    @torch.no_grad()
    def prefill(self, text_ids: torch.Tensor, audio_go: torch.Tensor,
                spk: torch.Tensor):
        """Inference step 0: consumes [spk, text, go-frame], returns (h, caches)."""
        h_t = self.tok_emb(text_ids) + self.mod_text
        h_a = self._embed_audio(audio_go)
        h_s = (spk + self.mod_spk).unsqueeze(1)
        h = torch.cat([h_s, h_t, h_a], dim=1)
        caches = []
        for blk in self.layers:
            h, c = blk.prefill(h)
            caches.append(c)
        return self.nf(h), caches

    def forward_step(self, frame: torch.Tensor, cache: list | None):
        """Inference step t>0: embed ONE frame, run blocks with KV-cache."""
        h = self._embed_audio(frame)  # [B,1,d]
        new_caches = []
        for i, blk in enumerate(self.layers):
            c = cache[i] if cache else None
            h, c2 = blk(h, c, False)
            new_caches.append(c2)
        return self.nf(h), new_caches
