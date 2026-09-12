# Architecture

## One token through one block

For residual stream r:

```text
u                 = rmsnorm(r)
[x, gate]         = in_proj(u)
x                 = silu(depthwise_causal_conv1d(x))
[dt, B, C]        = x_proj(x)
delta             = softplus(dt_proj(dt))
y, ssm_state      = selective_scan_step(x, delta, A, B, C, D, ssm_state)
r                 = r + out_proj(y * silu(gate))
```

After the last block, the model applies a final RMSNorm and the tied language
model head.

The causal convolution state is ordered oldest to newest. Each call shifts it
left, writes the current unactivated x in the last slot, then dots it with
the PyTorch kernel in stored order.

## Shapes for 130M

| Symbol | Shape | Value |
|---|---:|---:|
| residual | d_model | 768 |
| mixer channel | d_inner | 1536 |
| SSM state per channel | d_state | 16 |
| convolution history | d_conv | 4 |
| low-rank time step | dt_rank | 48 |
| blocks | n_layers | 24 |

The recurrent cache is:

```text
n_layers * d_inner * (d_conv + d_state)
= 24 * 1536 * (4 + 16)
= 737,280 float32 values
= 2.8125 MiB
```

There is no sequence-length term.

## Checkpoint format

All integers and floats are little-endian. Matrices are float32 row-major.

```text
LMB1
version
d_model, d_inner, d_state, d_conv, dt_rank, n_layers
vocab_size, padded_vocab_size, bos_id, eos_id
shared_classifier, has_conv_bias
norm_epsilon
embedding
for each layer:
  norm
  in_proj
  conv_weight
  conv_bias, when present
  x_proj
  dt_proj.weight
  dt_proj.bias
  A_log
  D
  out_proj
final_norm
classifier, unless tied to embedding
```

The tokenizer file starts with LMT1, then stores the vocabulary by token id
followed by byte-pair merges in rank order.

## Why validation is streaming

Hugging Face evaluates the whole prompt as a sequence. Lua evaluates one token
at a time. The reference dump stores every post-block residual, final normalized
state, and logit vector for every prompt position. Matching those values checks
all three failure-prone boundaries independently:

1. tensor serialization and row-major layout;
2. convolution history order;
3. the selective recurrence and its broadcasts.

Readable text is downstream evidence, not the correctness criterion.
