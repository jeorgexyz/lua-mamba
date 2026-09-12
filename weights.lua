-- weights.lua - flat float32 checkpoint loader

local Config = require('config')
local utils = require('utils')
local Weights = {}
local function array(file, n) return utils.read_float32_array(file, n) end

function Weights.load(path)
    local file = assert(io.open(path, 'rb'), 'cannot open checkpoint: ' .. path)
    local c = Config.read(file)
    local w = { layers = {} }

    w.embedding = array(file, c.padded_vocab_size * c.d_model)
    for l = 1, c.n_layers do
        local m = {}
        m.norm = array(file, c.d_model)
        m.in_proj = array(file, 2 * c.d_inner * c.d_model)
        m.conv_weight = array(file, c.d_inner * c.d_conv)
        m.conv_bias = c.conv_bias and array(file, c.d_inner) or nil
        m.x_proj = array(file, (c.dt_rank + 2 * c.d_state) * c.d_inner)
        m.dt_weight = array(file, c.d_inner * c.dt_rank)
        m.dt_bias = array(file, c.d_inner)
        m.A_log = array(file, c.d_inner * c.d_state)
        m.D = array(file, c.d_inner)
        m.out_proj = array(file, c.d_model * c.d_inner)
        w.layers[l] = m
    end
    w.final_norm = array(file, c.d_model)
    w.classifier = c.shared_classifier and w.embedding
        or array(file, c.padded_vocab_size * c.d_model)

    if file:read(1) then error('checkpoint has trailing data (format mismatch)') end
    file:close()
    return c, w
end

return Weights
