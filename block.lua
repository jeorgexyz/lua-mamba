-- block.lua - RMSNorm -> Mamba mixer -> residual

local utils = require('utils')
local rmsnorm = require('rmsnorm')
local ssm = require('ssm')
local Block = {}

function Block.new_state(c)
    return {
        conv = utils.zeros(c.d_inner * c.d_conv),
        ssm = utils.zeros(c.d_inner * c.d_state),
        norm = utils.zeros(c.d_model),
        projected = utils.zeros(2 * c.d_inner),
        input = utils.zeros(c.d_inner),
        gate = utils.zeros(c.d_inner),
        mixed = utils.zeros(c.d_inner),
        output = utils.zeros(c.d_model),
        scan = {
            parameters = utils.zeros(c.dt_rank + 2 * c.d_state),
            delta = utils.zeros(c.d_inner),
        },
    }
end

function Block.forward(hidden, w, c, state)
    rmsnorm.forward(state.norm, hidden, w.norm, c.d_model, c.norm_eps)
    utils.matmul(state.projected, state.norm, w.in_proj, 2 * c.d_inner, c.d_model)

    for d = 1, c.d_inner do
        state.input[d] = state.projected[d]
        state.gate[d] = state.projected[c.d_inner + d]
    end

    for d = 1, c.d_inner do
        local base = (d - 1) * c.d_conv
        for k = 1, c.d_conv - 1 do state.conv[base + k] = state.conv[base + k + 1] end
        state.conv[base + c.d_conv] = state.input[d]
        local sum = w.conv_bias and w.conv_bias[d] or 0.0
        for k = 1, c.d_conv do
            sum = sum + state.conv[base + k] * w.conv_weight[base + k]
        end
        state.input[d] = utils.silu(sum)
    end

    ssm.step(state.mixed, state.input, state.gate, state.ssm, w, c, state.scan)
    utils.matmul(state.output, state.mixed, w.out_proj, c.d_model, c.d_inner)
    for i = 1, c.d_model do hidden[i] = hidden[i] + state.output[i] end
    return hidden
end

return Block
