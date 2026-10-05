"""Rough objective checks on generated audio (the last --tail seconds of each file):

- lang / p:   Whisper's detected language and its confidence
- logprob:    mean per-token log-prob of Whisper's transcript; lower = less intelligible
- words:      transcript word count per 10 s
- silent:     fraction of 100 ms frames below -45 dBFS (catches collapse to silence)
"""

import argparse
import glob

import librosa
import numpy as np
import whisper

SR = 16000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="wav files or globs")
    p.add_argument("--tail", type=float, default=30, help="analyse only the last N seconds")
    p.add_argument("--model", default="turbo")
    p.add_argument("--show-text", action="store_true")
    a = p.parse_args()

    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    model = whisper.load_model(a.model)
    print(f"{'file':55} {'lang':>4} {'p':>5} {'logprob':>8} {'words/10s':>9} {'silent':>6}")
    for f in files:
        wav, _ = librosa.load(f, sr=SR, mono=True)
        wav = wav[-int(a.tail * SR) :].astype(np.float32)
        dur = len(wav) / SR

        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(wav), n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        r = model.transcribe(wav, temperature=0.0, condition_on_previous_text=False)
        segs = r["segments"]
        lp = np.mean([s["avg_logprob"] for s in segs]) if segs else float("nan")
        words = len(r["text"].split()) / dur * 10

        rms = librosa.feature.rms(y=wav, frame_length=1600, hop_length=1600)[0]
        silent = float(np.mean(20 * np.log10(rms + 1e-9) < -45))

        print(f"{f[-55:]:55} {lang:>4} {probs[lang]:5.2f} {lp:8.2f} {words:9.1f} {silent:6.2f}")
        if a.show_text:
            print(f"    {r['text'].strip()[:300]}")


if __name__ == "__main__":
    main()
