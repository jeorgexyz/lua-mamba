-- ssm.lua - one selective state-space step

local utils = require('utils')
local M = {}

function M.step(out, x, gate, state, w, c, scratch)
    local n, inner, rank = c.d_state, c.d_inner, c.dt_rank
    local p = scratch.parameters
    utils.matmul(p, x, w.x_proj, rank + 2 * n, inner)

    local delta = scratch.delta
    for d = 1, inner do
        local sum, base = w.dt_bias[d], (d - 1) * rank
        for r = 1, rank do sum = sum + w.dt_weight[base + r] * p[r] end
        delta[d] = utils.softplus(sum)
    end

    for d = 1, inner do
        local y, base = 0.0, (d - 1) * n
        local dt, u = delta[d], x[d]
        for j = 1, n do
            local idx = base + j
            local A = -math.exp(w.A_log[idx])
            local h = state[idx] * math.exp(dt * A) + dt * p[rank + j] * u
            state[idx] = h
            y = y + h * p[rank + n + j]
        end
        out[d] = (y + w.D[d] * u) * utils.silu(gate[d])
    end
    return out
end

return M
