#!/usr/bin/env python3
'''Dump per-token, per-layer Hugging Face activations for validate.lua.'''

import argparse
import struct

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def i32(file, value):
    file.write(struct.pack('<i', int(value)))


def write_tensor(file, value):
    value = value.detach().float().cpu().contiguous()
    file.write(value.numpy().tobytes(order='C'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--model', default='state-spaces/mamba-130m-hf')
    parser.add_argument('--output', default='reference.ref')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).eval()
    tokens = tokenizer(args.prompt, add_special_tokens=False, return_tensors='pt').input_ids
    if tokens.shape[1] == 0:
        tokens = torch.tensor([[model.config.bos_token_id]])
    with torch.no_grad():
        output = model(
            tokens, output_hidden_states=True, use_cache=False, return_dict=True
        )

    hidden = output.hidden_states
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    vocab_size = len(tokenizer)
    assert len(hidden) == n_layers + 1

    with open(args.output, 'wb') as file:
        file.write(b'LMR1')
        i32(file, 1)
        for value in (tokens.shape[1], n_layers, d_model, vocab_size):
            i32(file, value)
        for position, token in enumerate(tokens[0]):
            i32(file, token)
            for layer in range(n_layers):
                write_tensor(file, hidden[layer][0, position])
            write_tensor(file, hidden[-1][0, position])
            write_tensor(file, output.logits[0, position, :vocab_size])
    print(f'wrote {args.output} ({tokens.shape[1]} tokens)')


if __name__ == '__main__':
    main()
