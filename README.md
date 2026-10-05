# audai

Endless, textless audio continuation from a sample (radio chatter, ambience, ASMR).

Each step generates a chunk from the context `[anchor: start of the original sample] + [0.5 s silence] + [most recent audio]`,
so the output stays tied to the source but keeps moving.

## Backends

| `-m` | model | codebooks | context | speed (RTX 5070 laptop) | license |
|---|---|---|---|---|---|
| `llama-mimi` | llm-jp/Llama-Mimi-1.3B | 4 | ~20 s | ~1x realtime | Llama 3.2 |
| `soda` | soda-research/soda-1.7b-base | 8 (higher fidelity) | ~40 s | ~0.2x realtime | Apache 2.0 |

Both are trained on English speech only.

## Usage

```sh
uv sync
uv run audai-continue samples/great_day_en.wav -d 60                 # llama-mimi, 60 s of new audio
uv run audai-continue in.wav -m soda -d 30 --include-prompt -t 1.1   # soda, prompt + continuation
uv run audai-fx in.wav reverse radio reverse+radio pitch-down        # distorted prompt variants -> outputs/fx/
uv run python -m audai.evaluate "outputs/*.wav" --show-text          # whisper intelligibility check
```

Knobs: `--anchor/--window/--chunk` (seconds), `--sep none|silence`, `-t/--top-k/--top-p/--rep-penalty`, `--seed`.
