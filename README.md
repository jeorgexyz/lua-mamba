# Lua Mamba
<p align="center">
  <img width="350px" src="./assets/lua-mamba-art.png" alt="Lua Agent">
</p>


A pure Lua 5.3+ implementation of Mamba-1 inference. CPU only, no Torch,
no C extensions, and no dependencies in the runtime.

This is the third project in the same line as
[lua-llama](https://github.com/jeorgexyz/lua-llama) and
[lua-agent](https://github.com/jeorgexyz/lua-agent): keep the mechanism small
enough to read, then give it a falsifiable correctness test.

> The transformer cache grows with context. The Mamba state does not.

## Scope

- Mamba-1 (S6), inference only
- state-spaces/mamba-130m-hf
- float32 exported weights
- GPT-NeoX/GPT-2 byte-level BPE
- greedy, temperature, and top-p sampling

Training, quantization, batching, Mamba-2, and fused kernels are deliberately
out of scope.

## Run it

The Lua runtime has no packages to install. Python is used once, outside the
runtime, to convert the official Hugging Face checkpoint.

```bash
python -m pip install -r tools/requirements.txt
python tools/export_mamba.py
lua54 main.lua mamba-130m.lmb tokenizer.lmt 'Mamba is' 40 0.8 0.9
```

Arguments after the prompt are max_tokens, temperature, and top_p.
Use temperature 0 for deterministic greedy decoding.

The exported float32 checkpoint is roughly 500 MB. Loading floats into ordinary
Lua tables takes several times that amount of RAM; this is a readable reference
runtime, not a memory-efficient serving engine.

## What is actually new

Each layer keeps two fixed-size pieces of state:

```text
conv state   d_inner x d_conv
SSM state    d_inner x d_state
```

For one new token, ssm.lua computes:

```text
delta = softplus(W_delta dt + bias)
A_bar = exp(delta A), where A = -exp(A_log)
h     = A_bar h + delta B x
y     = (C h + D x) * silu(gate)
```

B, C, and delta depend on the current token. A and D are learned but
input-independent. The official 130M dimensions make the complete recurrent
cache 737,280 floats, about 2.8 MiB, no matter how long the context becomes.

## Correctness before generation

The sampler is not the test. Dump every layer and every logit from the Hugging
Face implementation, then compare the same prompt token by token in Lua:

```bash
python tools/reference.py 'Mamba is' --output reference.ref
lua54 validate.lua mamba-130m.lmb reference.ref
```

validate.lua reports the maximum absolute error at each layer and fails if
the worst error exceeds the tolerance (default 1e-2; pass a third argument
to change it).

Against `state-spaces/mamba-130m-hf`, all 24 layers and all three tokens:

<p align="center">
  <img src="./assets/demo-validate.gif" alt="per-layer parity against PyTorch" width="640">
</p>

```text
worst max_abs=0.000184514 at token 1 logits
parity passed
```

Per-layer errors run 1e-7 to 8e-5 and the worst case is 1.8e-4 at the
logits, which is float32 accumulation noise across 24 layers -- not a
difference in behaviour. The full 77-line transcript is committed at
[examples/validation.txt](examples/validation.txt); this is the claim the
repo exists to make, so the evidence ships with it.

The small offline suite needs no download:

```bash
lua54 test.lua
```

It writes a deterministic toy checkpoint, exercises the complete load and
forward path, proves recurrent state changes the next step, resets the cache,
and checks byte-level BPE merging.

## Generating

<p align="center">
  <img src="./assets/demo-generate.gif" alt="pure Lua Mamba-1 inference" width="720">
</p>

Real runs are in [examples/](examples/): `mamba-is.txt` and
`state-space-models.txt`, both greedy at temperature 0 and therefore
reproducible. Roughly 0.4 tokens/sec for 130M parameters in pure Lua -- this
is a readable reference runtime, not a serving engine.

Regenerate the GIFs with `python tools/make_demos.py` (needs the exported
checkpoint and Pillow).

## Project structure

```text
main.lua              CLI
model.lua             embedding, block stack, final norm, LM head
block.lua             projection, causal conv, SSM, gate, residual
ssm.lua               selective recurrence
rmsnorm.lua           RMS normalization
weights.lua           flat checkpoint loader
tokenizer.lua         byte-level BPE
sample.lua            greedy, temperature, top-p
generate.lua          autoregressive loop
validate.lua          PyTorch/Lua activation parity
tools/export_mamba.py checkpoint and tokenizer exporter
tools/reference.py    reference activation dump
tools/make_gif.py     terminal-session GIF renderer
tools/make_demos.py   regenerates the README GIFs
```

See [docs/architecture.md](docs/architecture.md) for tensor shapes and the
binary format.

## Known boundary

The tokenizer reproduces GPT-NeoX byte-level BPE for ordinary English text.
Its pure-Lua pre-tokenizer uses an ASCII approximation of the GPT-2 Unicode
regex, so parity for non-ASCII scripts and some Unicode punctuation is not yet
claimed. Weight and model parity are independent of that limitation because
the reference file carries token ids directly.

## License

MIT
