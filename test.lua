-- test.lua - offline tests with a deterministic tiny checkpoint

local utils = require('utils')
local Model = require('model')
local Tokenizer = require('tokenizer')
local sample = require('sample')
local ssm = require('ssm')
local passed = 0

local function near(a, b, eps)
    assert(math.abs(a - b) <= (eps or 1e-9), string.format('%g != %g', a, b))
    passed = passed + 1
end

local function write_i(file, value) file:write(string.pack('<i4', value)) end
local function write_f(file, value) file:write(string.pack('<f', value)) end
local function write_array(file, values)
    for _, value in ipairs(values) do write_f(file, value) end
end
local model_path = 'test-tiny.lmb'
local file = assert(io.open(model_path, 'wb'))
file:write('LMB1')
write_i(file, 1)
for _, value in ipairs({ 2, 2, 2, 2, 1, 1, 3, 3, 0, 0, 1, 1 }) do
    write_i(file, value)
end
write_f(file, 1e-5)
write_array(file, { 1, 0, 0, 1, 0.5, -0.5 })
write_array(file, { 1, 1 })
write_array(file, { 1, 0, 0, 1, 0.5, 0, 0, 0.5 })
write_array(file, { 0.25, 0.75, 0.25, 0.75 })
write_array(file, { 0, 0 })
write_array(file, { 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.3 })
write_array(file, { 1, 1 })
write_array(file, { -2, -2 })
write_array(file, { 0, math.log(2), 0, math.log(2) })
write_array(file, { 1, 1 })
write_array(file, { 0.5, 0, 0, 0.5 })
write_array(file, { 1, 1 })
file:close()

local out = {}
utils.matmul(out, { 2, 3 }, { 1, 2, 3, 4 }, 2, 2)
near(out[1], 8)
near(out[2], 18)
near(utils.softplus(100), 100)
near(utils.softplus(-100), math.exp(-100))
assert(sample.argmax({ -1, 4, 2 }) == 1)
passed = passed + 1

local scan_state, scan_out = { 0.2 }, {}
local scan_config = { d_state = 1, d_inner = 1, dt_rank = 1 }
local scan_weights = {
    x_proj = { 1, 2, 3 }, dt_weight = { 2 }, dt_bias = { 0 },
    A_log = { 0 }, D = { 0.25 },
}
local scan_scratch = { parameters = { 0, 0, 0 }, delta = { 0 } }
ssm.step(scan_out, { 0.5 }, { 0.4 }, scan_state, scan_weights, scan_config, scan_scratch)
local dt = utils.softplus(1)
local expected_state = 0.2 * math.exp(-dt) + dt * 1 * 0.5
local expected_out = (expected_state * 1.5 + 0.25 * 0.5) * utils.silu(0.4)
near(scan_state[1], expected_state)
near(scan_out[1], expected_out)

local model = Model.new(model_path)
local state = model:create_state()
local layers = 0
local first = model:forward(1, state, function() layers = layers + 1 end)
local snapshot = { first[1], first[2], first[3] }
local recurrent_snapshot = {}
for i = 1, #state.layers[1].ssm do recurrent_snapshot[i] = state.layers[1].ssm[i] end
assert(layers == 1)
passed = passed + 1
local second = model:forward(1, state)
local state_changed = false
for i = 1, #state.layers[1].ssm do
    if math.abs(state.layers[1].ssm[i] - recurrent_snapshot[i]) > 1e-8 then state_changed = true end
end
assert(state_changed)
passed = passed + 1
model:reset(state)
local reset = model:forward(1, state)
for i = 1, 3 do near(reset[i], snapshot[i], 1e-12) end

local tokenizer_path = 'test-tiny.lmt'
file = assert(io.open(tokenizer_path, 'wb'))
file:write('LMT1')
write_i(file, 1)
write_i(file, 6)
write_i(file, 0)
write_i(file, 0)
write_i(file, 2)
for _, token in ipairs({ '<|endoftext|>', 'h', 'i', utf8.char(288), 'hi',
                          utf8.char(288) .. 'hi' }) do
    write_i(file, #token)
    file:write(token)
end
for _, pair in ipairs({ { 'h', 'i' }, { utf8.char(288), 'hi' } }) do
    for _, token in ipairs(pair) do write_i(file, #token); file:write(token) end
end
file:close()

local tokenizer = Tokenizer.new(tokenizer_path)
local ids = tokenizer:encode('hi hi')
assert(#ids == 2 and ids[1] == 4 and ids[2] == 5)
assert(tokenizer:decode(5) == ' hi')
passed = passed + 2

os.remove(model_path)
os.remove(tokenizer_path)
print(string.format('%d assertions passed', passed))
