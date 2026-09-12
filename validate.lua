-- validate.lua - compare streaming Lua activations with a PyTorch reference dump

local utils = require('utils')
local Model = require('model')

local checkpoint = assert(arg[1], 'usage: lua validate.lua model.lmb reference.ref [tolerance]')
local reference = assert(arg[2], 'missing reference file')
local tolerance = tonumber(arg[3]) or 1e-2
local file = assert(io.open(reference, 'rb'))
assert(utils.read_exact(file, 4) == 'LMR1', 'not a lua-mamba reference')
assert(utils.read_int32(file) == 1, 'unsupported reference version')
local n_tokens = utils.read_int32(file)
local n_layers = utils.read_int32(file)
local d_model = utils.read_int32(file)
local vocab_size = utils.read_int32(file)

local model = Model.new(checkpoint)
local c, state = model.config, model:create_state()
assert(n_layers == c.n_layers and d_model == c.d_model, 'reference/model shape mismatch')
assert(vocab_size == c.vocab_size, 'reference/model vocabulary mismatch')
local worst, where = 0.0, ''

local function compare(actual, size, label)
    local expected = utils.read_float32_array(file, size)
    local max = 0.0
    for i = 1, size do max = math.max(max, math.abs(actual[i] - expected[i])) end
    if max > worst then worst, where = max, label end
    return max
end

for position = 1, n_tokens do
    local token = utils.read_int32(file)
    local seen = 0
    local logits = model:forward(token, state, function(layer, hidden)
        seen = seen + 1
        local err = compare(hidden, d_model, string.format('token %d layer %d', position, layer))
        print(string.format('token %d layer %d max_abs=%.6g', position, layer, err))
    end)
    assert(seen == n_layers)
    compare(state.hidden, d_model, string.format('token %d final norm', position))
    compare(logits, vocab_size, string.format('token %d logits', position))
end
file:close()
print(string.format('worst max_abs=%.6g at %s', worst, where))
if worst > tolerance then
    error(string.format('parity failed: %.6g > tolerance %.6g', worst, tolerance))
end
print('parity passed')
