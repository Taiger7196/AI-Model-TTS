#!/usr/bin/env python3
"""Synthesize: text + reference voice -> wav.
    python scripts/synth.py --ckpt checkpoints/chimera-tiny/step_010000.pt \\
        --ref samples/voice.wav --text "hello world" --out out.wav
"""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chimera.config import ChimeraConfig
from chimera.acoustic import ChimeraTTS
from chimera.data import CharTokenizer, MelFrontend, GriffinLimVocoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/tiny_100m.yaml")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = ChimeraConfig.from_yaml(args.config)
    device = torch.device(args.device)
    tok = CharTokenizer()

    model = ChimeraTTS(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"[synth] loaded {args.ckpt} @ step {ck.get('step', '?')}")

    wav, sr = torchaudio.load(args.ref)
    wav = wav.mean(dim=0)
    if sr != cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
    front = MelFrontend(cfg)
    ref_mel = front(wav).unsqueeze(0).to(device)
    text_ids = torch.tensor([tok.encode(args.text, cfg.max_text_len)],
                            dtype=torch.long, device=device)

    with torch.no_grad():
        gen_mel = model.generate_mel(text_ids, ref_mel, max_len=args.max_len)
    voc = GriffinLimVocoder(cfg)
    wav_out = voc(gen_mel.cpu()).squeeze(0).numpy()
    sf.write(args.out, wav_out, cfg.sample_rate)
    print(f"[synth] wrote {args.out} ({wav_out.shape[0]/cfg.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
