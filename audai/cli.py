"""Endless continuation: context = [anchor from the original sample] + [sep] + [most recent audio].

The anchor keeps the output tied to the source's voice/character; the sliding recent window
keeps it contiguous while letting it drift.
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .backends import BACKENDS, FRAME_RATE, SAMPLE_RATE, SAMPLES_PER_FRAME, MimiCodec, Sampling


def load_audio(path: str, trim_db: float | None) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if trim_db:
        _, (a, b) = librosa.effects.trim(wav, top_db=trim_db)
        wav = wav[max(0, a - 2400) : b + 2400]
    peak = np.abs(wav).max()
    return wav / peak * 0.9 if peak > 0 else wav


def frames(seconds: float) -> int:
    return int(round(seconds * FRAME_RATE))


def silent_mask(be, codes: torch.Tensor, thresh_db: float = -45) -> torch.Tensor:
    """Per-frame bool: is this frame's decoded audio below thresh_db?"""
    audio = be.decode(codes)
    n = len(audio) // SAMPLES_PER_FRAME
    rms = np.sqrt((audio[: n * SAMPLES_PER_FRAME].reshape(n, -1) ** 2).mean(1))
    return torch.from_numpy(20 * np.log10(rms + 1e-9) < thresh_db)


def cut_long_pause(mask: torch.Tensor, tail: int, max_pause: int) -> tuple[int, int]:
    """Return (frames to keep, trailing silent run) so no silent run exceeds max_pause,
    counting `tail` silent frames already at the end of the history."""
    run = tail
    for i, s in enumerate(mask.tolist()):
        run = run + 1 if s else 0
        if run > max_pause:
            return i, run - 1
    return len(mask), run


def build_context(prompt, generated, anchor, sep, window_frames, max_frames):
    history = torch.cat([prompt, generated], dim=1)
    if history.shape[1] <= max_frames:
        return history
    parts = [anchor] + ([sep] if sep is not None else [])
    budget = max_frames - sum(p.shape[1] for p in parts)
    recent = history[:, -min(window_frames, budget) :]
    return torch.cat(parts + [recent], dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="prompt audio (any format ffmpeg/librosa can read)")
    p.add_argument("-o", "--output", help="output wav (default: outputs/<input>_<model>.wav)")
    p.add_argument("-m", "--model", choices=BACKENDS, default="llama-mimi")
    p.add_argument("--model-id", help="override HF model id (e.g. llm-jp/Llama-Mimi-8B, soda-research/soda-600m-base)")
    p.add_argument("-d", "--duration", type=float, default=30, help="seconds of new audio to generate")
    p.add_argument("--prompt-seconds", type=float, default=None, help="use only the first N s of the input")
    p.add_argument("--anchor", type=float, default=None, help="seconds of the original kept at the start of every context")
    p.add_argument("--window", type=float, default=None, help="seconds of most-recent audio in the context")
    p.add_argument("--chunk", type=float, default=None, help="seconds generated per step")
    p.add_argument("--sep", choices=["none", "silence"], default="silence", help="what goes between anchor and recent audio")
    p.add_argument("-t", "--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--rep-penalty", type=float, default=1.0)
    p.add_argument("--max-pause", type=float, default=1.5, help="longest silence allowed before resampling (0 disables)")
    p.add_argument("--trim-db", type=float, default=40, help="trim leading/trailing silence (0 disables)")
    p.add_argument("--include-prompt", action="store_true", help="prepend the (re-encoded) prompt to the output")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live when piped

    if a.seed is not None:
        torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = MimiCodec(device)
    kw = {"model_id": a.model_id} if a.model_id else {}
    be = BACKENDS[a.model](codec, **kw)

    # Per-backend defaults sized to each model's trained context.
    if a.model == "llama-mimi":
        dflt = dict(anchor=5, window=9, chunk=5, temperature=0.8, top_k=30, top_p=None)
    else:
        dflt = dict(anchor=8, window=16, chunk=6, temperature=1.0, top_k=None, top_p=0.9)
    for k, v in dflt.items():
        attr = k.replace("-", "_")
        if getattr(a, attr) is None:
            setattr(a, attr, v)

    wav = load_audio(a.input, a.trim_db)
    if a.prompt_seconds:
        wav = wav[: int(a.prompt_seconds * SAMPLE_RATE)]
    prompt = be.encode(wav)
    anchor = prompt[:, : frames(a.anchor)]
    sep = be.encode(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)) if a.sep == "silence" else None
    chunk = frames(a.chunk)
    max_ctx = be.max_context_frames - chunk
    s = Sampling(a.temperature, a.top_k, a.top_p, a.rep_penalty)  # temperature is bumped on retries

    print(
        f"{be.name}: prompt {prompt.shape[1] / FRAME_RATE:.1f}s, anchor {anchor.shape[1] / FRAME_RATE:.1f}s, "
        f"window {a.window}s, chunk {a.chunk}s, max ctx {max_ctx / FRAME_RATE:.1f}s, {s}"
    )
    generated = torch.zeros(be.K, 0, dtype=torch.long)
    target = frames(a.duration)
    max_pause = frames(a.max_pause)
    tail_silence, retries = 0, 0
    t0 = time.time()
    while generated.shape[1] < target:
        ctx = build_context(prompt, generated, anchor, sep, frames(a.window), max_ctx)
        # Retries use shorter chunks: they usually fail early, so don't pay for a full chunk.
        n = min(max(frames(1), chunk >> retries), target - generated.shape[1])
        # Each failed attempt (the model trailing off into silence) samples a bit hotter.
        s.temperature = a.temperature + 0.1 * min(retries, 4)
        new = be.continue_codes(ctx, n, s)
        note = ""
        if max_pause and new.shape[1]:
            keep, tail_silence = cut_long_pause(silent_mask(be, new), tail_silence, max_pause)
            if keep < new.shape[1]:
                new, retries = new[:, :keep], retries + 1
                note = f"  [cut {n - keep} frames of silence, retry {retries}]"
            else:
                retries = 0
        if retries > 6:
            # Stuck in silence: back up a second and try from there.
            generated = generated[:, : max(0, generated.shape[1] - frames(1))]
            tail_silence, retries = 0, 0
            note += "  [backtracked 1s]"
        elif new.shape[1] == 0 and not max_pause:
            print("model produced no audio; stopping")
            break
        generated = torch.cat([generated, new], dim=1)
        el = time.time() - t0
        done = generated.shape[1] / FRAME_RATE
        print(f"  {done:6.1f}s / {a.duration:.0f}s  (ctx {ctx.shape[1] / FRAME_RATE:.1f}s, {done / el:.2f}x realtime){note}")

    codes = torch.cat([prompt, generated], dim=1) if a.include_prompt else generated
    audio = be.decode(codes)
    out = Path(a.output or f"outputs/{Path(a.input).stem}_{be.name}.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, SAMPLE_RATE)
    print(f"wrote {out} ({len(audio) / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
