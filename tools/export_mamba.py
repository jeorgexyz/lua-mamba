#!/usr/bin/env python3
'''Export a Hugging Face Mamba-1 model and byte-level BPE for lua-mamba.'''

import argparse
import struct
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def i32(file, value):
    file.write(struct.pack('<i', int(value)))


def f32(file, value):
    file.write(struct.pack('<f', float(value)))


def tensor(file, value, shape, name):
    value = value.detach().float().cpu().contiguous()
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name}: expected {tuple(shape)}, got {tuple(value.shape)}')
    file.write(value.numpy().tobytes(order='C'))


def export_tokenizer(tokenizer, output):
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    by_id = [None] * size
    for token, token_id in vocab.items():
        by_id[token_id] = token
    if any(token is None for token in by_id):
        raise ValueError('tokenizer ids are not contiguous')

    with tempfile.TemporaryDirectory() as directory:
        saved = tokenizer.backend_tokenizer.model.save(directory)
        merges_path = next((Path(path) for path in saved if path.endswith('merges.txt')), None)
        if merges_path is None:
            raise ValueError('tokenizer is not a byte-level BPE model')
        lines = merges_path.read_text(encoding='utf-8').splitlines()
        merges = [tuple(line.split()) for line in lines if line and not line.startswith('#')]
    if any(len(pair) != 2 for pair in merges):
        raise ValueError('unexpected merge record')

    with open(output, 'wb') as file:
        file.write(b'LMT1')
        i32(file, 1)
        i32(file, size)
        i32(file, tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1)
        i32(file, tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1)
        i32(file, len(merges))
        for token in by_id:
            data = token.encode('utf-8')
            i32(file, len(data))
            file.write(data)
        for left, right in merges:
            for token in (left, right):
                data = token.encode('utf-8')
                i32(file, len(data))
                file.write(data)
    return size


def export_model(model, vocab_size, output):
    c = model.config
    d_model = c.hidden_size
    d_inner = c.intermediate_size
    d_state = c.state_size
    d_conv = c.conv_kernel
    dt_rank = int(c.time_step_rank)
    layers = model.backbone.layers
    embedding = model.backbone.embeddings.weight
    classifier = model.lm_head.weight
    padded_vocab = embedding.shape[0]
    shared = torch.equal(embedding, classifier)
    conv_bias = layers[0].mixer.conv1d.bias is not None

    if any(layer.mixer.in_proj.bias is not None or layer.mixer.out_proj.bias is not None
           for layer in layers):
        raise ValueError('lua-mamba currently supports use_bias=false checkpoints only')

    with open(output, 'wb') as file:
        file.write(b'LMB1')
        i32(file, 1)
        for value in (d_model, d_inner, d_state, d_conv, dt_rank, len(layers),
                      vocab_size, padded_vocab,
                      c.bos_token_id if c.bos_token_id is not None else -1,
                      c.eos_token_id if c.eos_token_id is not None else -1,
                      shared, conv_bias):
            i32(file, value)
        f32(file, c.layer_norm_epsilon)
        tensor(file, embedding, (padded_vocab, d_model), 'embedding')
        for index, layer in enumerate(layers):
            m = layer.mixer
            prefix = f'layer {index}'
            tensor(file, layer.norm.weight, (d_model,), prefix + ' norm')
            tensor(file, m.in_proj.weight, (2 * d_inner, d_model), prefix + ' in_proj')
            tensor(file, m.conv1d.weight, (d_inner, 1, d_conv), prefix + ' conv')
            if conv_bias:
                tensor(file, m.conv1d.bias, (d_inner,), prefix + ' conv_bias')
            tensor(file, m.x_proj.weight,
                   (dt_rank + 2 * d_state, d_inner), prefix + ' x_proj')
            tensor(file, m.dt_proj.weight, (d_inner, dt_rank), prefix + ' dt_weight')
            tensor(file, m.dt_proj.bias, (d_inner,), prefix + ' dt_bias')
            tensor(file, m.A_log, (d_inner, d_state), prefix + ' A_log')
            tensor(file, m.D, (d_inner,), prefix + ' D')
            tensor(file, m.out_proj.weight, (d_model, d_inner), prefix + ' out_proj')
        tensor(file, model.backbone.norm_f.weight, (d_model,), 'final norm')
        if not shared:
            tensor(file, classifier, (padded_vocab, d_model), 'classifier')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='state-spaces/mamba-130m-hf')
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--output', default='mamba-130m.lmb')
    parser.add_argument('--tokenizer-output', default='tokenizer.lmt')
    args = parser.parse_args()

    tokenizer_id = args.tokenizer or args.model
    print(f'loading tokenizer: {tokenizer_id}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    vocab_size = export_tokenizer(tokenizer, args.tokenizer_output)
    print(f'wrote {args.tokenizer_output} ({vocab_size} tokens)')

    print(f'loading model: {args.model}')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    export_model(model, vocab_size, args.output)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
