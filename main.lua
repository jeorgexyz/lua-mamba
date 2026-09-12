-- main.lua - pure Lua Mamba-1 inference

if not string.unpack or not utf8 then
    io.stderr:write('lua-mamba requires Lua 5.3+\n')
    os.exit(1)
end

local checkpoint = arg[1] or 'mamba-130m.lmb'
local tokenizer_path = arg[2] or 'tokenizer.lmt'
local prompt = arg[3] or 'Mamba is'
local max_tokens = tonumber(arg[4]) or 40
local temperature = tonumber(arg[5]) or 0.8
local top_p = tonumber(arg[6]) or 0.9

math.randomseed(os.time())
local Model = require('model')
local Tokenizer = require('tokenizer')
local generate = require('generate')

print('Lua Mamba - pure Lua Mamba-1 inference')
local started = os.clock()
local model = Model.new(checkpoint)
local tokenizer = Tokenizer.new(tokenizer_path)
assert(tokenizer.vocab_size == model.config.vocab_size, 'model/tokenizer vocabulary mismatch')
local c = model.config
print(string.format(
    'd_model=%d d_inner=%d d_state=%d layers=%d vocab=%d',
    c.d_model, c.d_inner, c.d_state, c.n_layers, c.vocab_size
))
print(string.format(
    'recurrent cache: %d floats (constant with context length)',
    c.n_layers * c.d_inner * (c.d_conv + c.d_state)
))
generate.generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
print(string.format('%.2fs', os.clock() - started))
