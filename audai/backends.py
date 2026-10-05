"""Textless audio-continuation backends.

Every backend works on Mimi codes shaped [K, T] (K codebooks, T frames at 12.5 Hz)
and exposes the same three operations: encode a waveform, continue a code
sequence by N frames, and decode codes back to a waveform.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    MimiModel,
)

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = 1920
CODEBOOK_SIZE = 2048


@dataclass
class Sampling:
    temperature: float = 0.9
    top_k: int | None = None
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0


class MimiCodec:
    def __init__(self, device: str):
        self.device = device
        self.model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    @torch.no_grad()
    def encode(self, wav: np.ndarray, num_quantizers: int) -> torch.Tensor:
        x = torch.tensor(wav, dtype=torch.float32, device=self.device)[None, None]
        codes = self.model.encode(x, num_quantizers=num_quantizers).audio_codes
        return codes[0, :num_quantizers].cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, block: int = 750, overlap: int = 50) -> np.ndarray:
        """Decode [K, T] codes in overlapping blocks so long outputs don't blow up memory."""
        out = []
        T = codes.shape[1]
        for start in range(0, T, block):
            lead = min(overlap, start)
            seg = codes[:, start - lead : start + block].to(self.device)[None]
            audio = self.model.decode(seg).audio_values[0, 0].float().cpu().numpy()
            out.append(audio[lead * SAMPLES_PER_FRAME :])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class _ForceCodebookOrder(LogitsProcessor):
    """Only allow the token ids of the codebook due at each step, so output is always
    well-formed frames and the model can never close the audio segment."""

    def __init__(self, table: torch.Tensor, prompt_len: int):
        self.table = table  # [K, 2048] token ids
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        q = (input_ids.shape[1] - self.prompt_len) % self.table.shape[0]
        allowed = self.table[q].to(scores.device)
        masked = torch.full_like(scores, float("-inf"))
        masked[:, allowed] = scores[:, allowed]
        return masked


class LlamaMimiBackend:
    """llm-jp/Llama-Mimi: Llama 3.2 over flattened Mimi tokens `<value_codebook>`, 4 codebooks."""

    name = "llama-mimi"
    # Trained on 1024-token sequences = 256 frames (~20 s) with 4 codebooks.
    max_context_frames = 256

    def __init__(self, codec: MimiCodec, model_id: str = "llm-jp/Llama-Mimi-1.3B"):
        self.codec = codec
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
            .to(codec.device)
            .eval()
        )
        self.K = self.model.config.num_quantizers
        self.table = torch.tensor(
            [
                self.tok.convert_tokens_to_ids([f"<{v}_{q}>" for v in range(CODEBOOK_SIZE)])
                for q in range(self.K)
            ]
        )
        self.lut = torch.full((len(self.tok),), -1, dtype=torch.long)
        for q in range(self.K):
            self.lut[self.table[q]] = torch.arange(CODEBOOK_SIZE)
        self.prefix = [self.tok.bos_token_id, self.tok.convert_tokens_to_ids("<audio>")]

    def encode(self, wav):
        return self.codec.encode(wav, self.K)

    def decode(self, codes):
        return self.codec.decode(codes)

    @torch.no_grad()
    def continue_codes(self, context: torch.Tensor, n_frames: int, s: Sampling) -> torch.Tensor:
        # [K, T] -> frame-major flat ids
        ids = self.table.gather(1, context.long()).T.reshape(-1)
        input_ids = torch.tensor(self.prefix + ids.tolist(), device=self.codec.device)[None]
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=n_frames * self.K,
            min_new_tokens=n_frames * self.K,
            do_sample=True,
            temperature=s.temperature,
            top_k=s.top_k or 0,
            top_p=s.top_p or 1.0,
            repetition_penalty=s.repetition_penalty,
            logits_processor=LogitsProcessorList([_ForceCodebookOrder(self.table, input_ids.shape[1])]),
            pad_token_id=self.tok.eos_token_id,
        )
        new = out[0, input_ids.shape[1] :].cpu()
        vals = self.lut[new]
        n = (vals.numel() // self.K) * self.K
        return vals[:n].reshape(-1, self.K).T


class SodaBackend:
    """soda-research/SODA: Qwen3-style LM over Mimi codes mapped to private-use unicode
    chars (8 codebooks) with a BPE tokenizer on top, so we go through strings."""

    name = "soda"
    UNICODE_OFFSET = 0xE000
    max_context_frames = 400  # 4096-token positions at ~100 audio tokens/s, minus headroom

    def __init__(self, codec: MimiCodec, model_id: str = "soda-research/soda-1.7b-base"):
        self.codec = codec
        self.K = 8
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
            .to(codec.device)
            .eval()
        )
        # Keep it pure audio: never close the segment, switch to text, or end.
        self.suppress = [
            self.tok.convert_tokens_to_ids(t)
            for t in ("<|audio_end|>", "<|text_start|>", "<|text_end|>", "<|end_of_text|>")
        ]

    def encode(self, wav):
        return self.codec.encode(wav, self.K)

    def decode(self, codes):
        return self.codec.decode(codes)

    def _to_str(self, codes: torch.Tensor) -> str:
        offs = self.UNICODE_OFFSET + torch.arange(self.K)[:, None] * CODEBOOK_SIZE
        return "".join(map(chr, (codes.long() + offs).T.reshape(-1).tolist()))

    def _from_str(self, s: str) -> torch.Tensor:
        """Parse chars into frames by their codebook range, resyncing on any misalignment."""
        frames, cur = [], []
        for ch in s:
            c = ord(ch) - self.UNICODE_OFFSET
            if not 0 <= c < self.K * CODEBOOK_SIZE:
                continue
            q, v = divmod(c, CODEBOOK_SIZE)
            if q != len(cur):
                cur = []
                if q != 0:
                    continue
            cur.append(v)
            if len(cur) == self.K:
                frames.append(cur)
                cur = []
        if not frames:
            return torch.zeros(self.K, 0, dtype=torch.long)
        return torch.tensor(frames, dtype=torch.long).T

    @torch.no_grad()
    def continue_codes(self, context: torch.Tensor, n_frames: int, s: Sampling) -> torch.Tensor:
        prompt = "<|audio_start|>" + self._to_str(context)
        inputs = self.tok(prompt, return_tensors="pt").to(self.codec.device)
        plen = inputs.input_ids.shape[1]
        new = torch.zeros(self.K, 0, dtype=torch.long)
        # BPE packs several codes per token, so the frame count per token varies.
        while new.shape[1] < n_frames:
            out = self.model.generate(
                **inputs,
                max_new_tokens=(n_frames - new.shape[1]) * self.K,
                do_sample=True,
                temperature=s.temperature,
                top_k=s.top_k or 0,
                top_p=s.top_p or 1.0,
                repetition_penalty=s.repetition_penalty,
                suppress_tokens=self.suppress,
                pad_token_id=self.tok.eos_token_id,
            )
            text = self.tok.decode(out[0, plen:], skip_special_tokens=False)
            got = self._from_str(text)
            if got.shape[1] == 0:
                break
            new = torch.cat([new, got], dim=1)
            if new.shape[1] < n_frames:
                inputs = self.tok(prompt + self._to_str(new), return_tensors="pt").to(self.codec.device)
                plen = inputs.input_ids.shape[1]
        return new[:, :n_frames]


BACKENDS = {"llama-mimi": LlamaMimiBackend, "soda": SodaBackend}
