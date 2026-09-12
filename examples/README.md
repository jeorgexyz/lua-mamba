# Example output

Real runs against `state-spaces/mamba-130m-hf`, exported with
`tools/export_mamba.py`. Nothing here is hand-written or edited.

| File | What it is |
|---|---|
| `validation.txt` | per-layer, per-token parity against the Hugging Face implementation |
| `mamba-is.txt` | 30 tokens, greedy |
| `state-space-models.txt` | 30 tokens, greedy |

## The one that matters

`validation.txt` is the repo's claim. Every layer and every logit compared
against PyTorch for the same prompt:

```text
worst max_abs=0.000184514 at token 1 logits
parity passed
```

Per-layer errors run 1e-7 to 8e-5, rising with depth exactly as float32
accumulation does. The worst case sits at the logits after 24 layers. That
is arithmetic noise, not a behavioural difference — a real bug in the
selective scan or the causal convolution would show up orders of magnitude
larger, and at a specific layer rather than smoothly with depth.

Reproduce it:

```bash
python tools/export_mamba.py
python tools/reference.py 'Mamba is' --output reference.ref
lua54 validate.lua mamba-130m.lmb reference.ref
```

## Generation

Both transcripts are temperature 0, so they are deterministic and should
reproduce byte for byte:

```bash
lua54 main.lua mamba-130m.lmb tokenizer.lmt 'Mamba is' 30 0
```

Each prints the line the project is really about:

```text
recurrent cache: 737280 floats (constant with context length)
```

That number does not move as the context grows. A transformer's KV cache
would.

The timings in these files (~75s for 30 tokens, about 0.4 tokens/sec) are
pure Lua on CPU with 130M parameters in ordinary Lua tables. Readability was
the goal; speed was not.
