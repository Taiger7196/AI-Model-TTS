# CHIMERA-TTS 🔥

**Voice-cloning TTS, rewritten from scratch. Target: 3B params. Rule: Just Cook.**

Zero-shot voice cloning: feed any 3–10s reference clip + text → speech in that voice.
One codebase, two scales: an **85M prototype that trains on Colab T4 free**, and the
**2.92B CHIMERA-3B config** for the scale-up phase.

## T4 or CPU? — verdict: T4, always.

| Runtime | Verdict |
|---|---|
| **Colab T4 free (16GB VRAM)** | ✅ Train the tiny prototype here. Mixed precision + grad checkpointing + accumulation fit fine. |
| **12.7GB CPU, no GPU** | ❌ Never for training (~100x slower, a single epoch = days). ✅ Only for dataset prep / tokenization to save GPU hours. |

Honest math on "the impossible": a 3B model needs ~12GB just for fp32 weights, plus
gradients + optimizer states + activations. Full 3B pretraining on a single free T4 is not
realistic — so we do it the smart way: **prove the architecture at 85M on the T4, then scale
the exact same code to 3B with FSDP/multi-GPU** (Phase 2). That's how the impossible gets cooked.

## Architecture

Decoder-only transformer over a single stream (codec-LM style):

```
  text ──▶ [BOS] t1 t2 ... tT                               (char/BPE tokens)
  ref  ──▶ SpeakerEncoder(ref_mel) ──▶ [SPK]                (L2-norm voice vector)
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    │  CHIMERA backbone (GQA + RoPE + SwiGLU + RMSNorm)  │
                    │  stream: [ SPK ] [ TEXT ... ] [ AUDIO ... ]        │
                    └──────────────┬──────────────────┬──────────────────┘
                       phase-0     │                  │     phase-2
                       MelHead     │                  │     CodecHead (8×RVQ levels)
                          │        │                  │        │
                     mel ─┴─▶ HiFi-GAN (phase-1)  DAC-tokens ─┴─▶ DAC decoder
                     (Griffin-Lim for instant pipeline validation)
```

| Component | File | Tiny (T4) | CHIMERA-3B |
|---|---|---|---|
| Backbone (L×d, GQA, RoPE, SwiGLU) | `src/chimera/backbone.py` | 12×768, 76.6M | 28×3072, 2.850B |
| Speaker encoder (conv + transformer + attentive pool) | `src/chimera/speaker.py` | 4.0M | 22.8M |
| Head (mel + stop + PostNet / 8× codec) | `src/chimera/acoustic.py` | 4.6M | 50.3M |
| Vocoder (HiFi-GAN G + MPD + MSD) | `src/chimera/vocoder.py` | separate stage | separate stage |
| **Total acoustic** | | **85.1M** ✅ verified | **2.924B** ✅ verified |

Verify yourself: `python scripts/count_params.py`

## Layout

```
configs/          tiny_100m.yaml (T4) · chimera_3b.yaml (scale-up target)
src/chimera/      config · backbone · speaker · acoustic · vocoder · data · trainer
scripts/          train.py · count_params.py · synth.py
notebooks/        00_T4_prototype.ipynb (Colab entry point)
```

## Quickstart (Colab T4)

Open `notebooks/00_T4_prototype.ipynb`, or manually:

```bash
git clone https://github.com/Taiger7196/AI-Model-TTS.git
cd AI-Model-TTS && git checkout arena/01a08c88-ai-model-tts
pip install -r requirements.txt

python scripts/count_params.py
# FLASH demo (~10 min on T4): hear cloning fast
python scripts/train.py --config configs/flash.yaml
# TURBO-FLASH overfit demo: hear cloning in ~20 min on T4
python scripts/train.py --config configs/turbo.yaml
# Full prototype run
python scripts/train.py --config configs/tiny_100m.yaml --max-items 2000 --max-steps 2000
python scripts/synth.py --ckpt checkpoints/chimera-tiny/step_000XXX.pt \
    --ref myvoice.wav --text "chimera speaks" --out out.wav
```

T4 survival tips: checkpoint often (`ckpt_every`), training auto-resumes with `--resume`,
generated samples land in `checkpoints/chimera-tiny/samples/` so you can **hear** overfitting happen.

## Roadmap

- **Phase 0 — Prototype (now, T4):** 85M mel-based, LibriTTS, Griffin-Lim. Prove text→speech + voice conditioning converge.
- **Phase 1 — Quality:** train HiFi-GAN vocoder, phoneme/BPE frontend, cross-utterance refs, multilingual (EN+IT).
- **Phase 2 — Scale-up:** flip `target: codec`, train RVQ/DAC tokenizer, launch 3B with FSDP + DeepSpeed on multi-GPU.
- **Phase 3 — Clone:** long-form stability, prosody control, eval harness (SIM-O, WER, MOS proxy).

## Status

- [x] Architecture rewritten from scratch (backbone, speaker encoder, heads, vocoder)
- [x] Smoke-tested: train step, AR generate (mel + codec), KV-cache equivalence (err 1.7e-6), vocoder, trainer loop, wav output
- [ ] Phase-0 convergence run on T4
- [ ] HiFi-GAN training script
- [ ] 3B launch

Built different. Just cook. 🔥
