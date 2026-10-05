"""Make distorted/obscured prompt variants with ffmpeg, to see how the models continue them.

    audai-fx in.wav radio reverse pitch-down    -> outputs/fx/in_radio.wav, ...
"""

import argparse
import subprocess
from pathlib import Path

EFFECTS = {
    # band-limited, crushed, compressed: walkie-talkie / police radio
    "radio": "highpass=f=400,lowpass=f=2800,acrusher=bits=6:mode=log:aa=1:mix=0.35,acompressor=threshold=0.1:ratio=6,volume=2",
    "telephone": "highpass=f=300,lowpass=f=3400,acompressor=threshold=0.1:ratio=4",
    # speech played backwards: human voice, no words
    "reverse": "areverse",
    "pitch-up": "asetrate=24000*1.25,aresample=24000,atempo=0.8",
    "pitch-down": "asetrate=24000*0.8,aresample=24000,atempo=1.25",
    "muffled": "lowpass=f=700,volume=1.5",  # through a wall
    "fast": "atempo=1.4",
    "slow": "atempo=0.75",
    "underwater": "lowpass=f=900,aphaser=type=t:speed=0.6,chorus=0.6:0.9:50:0.4:0.25:2",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("effects", nargs="+", help=f"any of: {', '.join(EFFECTS)}; chain with '+', e.g. reverse+radio")
    p.add_argument("--outdir", default="outputs/fx")
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fx in a.effects:
        chain = ",".join(EFFECTS[e] for e in fx.split("+"))
        out = outdir / f"{Path(a.input).stem}_{fx.replace('+', '-')}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", a.input, "-ac", "1", "-ar", "24000", "-af", chain, str(out)],
            check=True,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
