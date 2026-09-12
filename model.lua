-- model.lua - embedding, Mamba blocks, final norm, and LM head

local Weights = require('weights')
local Block = require('block')
local rmsnorm = require('rmsnorm')
local utils = require('utils')

local Model = {}
Model.__index = Model

function Model.new(path)
    local c, w = Weights.load(path)
    return setmetatable({ config = c, weights = w }, Model)
end

function Model:create_state()
    local s = { hidden = utils.zeros(self.config.d_model), logits = {}, layers = {} }
    for l = 1, self.config.n_layers do s.layers[l] = Block.new_state(self.config) end
    return s
end

function Model:reset(state)
    for l = 1, self.config.n_layers do
        local layer = state.layers[l]
        for i = 1, #layer.conv do layer.conv[i] = 0.0 end
        for i = 1, #layer.ssm do layer.ssm[i] = 0.0 end
    end
end

function Model:forward(token, state, trace)
    local c, w, x = self.config, self.weights, state.hidden
    assert(token >= 0 and token < c.vocab_size, 'token id out of range: ' .. token)
    local base = token * c.d_model
    for i = 1, c.d_model do x[i] = w.embedding[base + i] end

    for l = 1, c.n_layers do
        Block.forward(x, w.layers[l], c, state.layers[l])
        if trace then trace(l, x) end
    end
    rmsnorm.forward(x, x, w.final_norm, c.d_model, c.norm_eps)

    for id = 0, c.vocab_size - 1 do
        local sum, offset = 0.0, id * c.d_model
        for j = 1, c.d_model do sum = sum + w.classifier[offset + j] * x[j] end
        state.logits[id + 1] = sum
    end
    return state.logits
end

return Model
