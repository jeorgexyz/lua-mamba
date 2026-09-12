-- rmsnorm.lua - Mamba RMSNorm

local M = {}

function M.forward(out, x, weight, size, eps)
    local sum = 0.0
    for i = 1, size do sum = sum + x[i] * x[i] end
    local scale = 1 / math.sqrt(sum / size + eps)
    for i = 1, size do out[i] = weight[i] * x[i] * scale end
    return out
end

return M
