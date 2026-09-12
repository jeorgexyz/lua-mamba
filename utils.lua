-- utils.lua - scalar math and little-endian binary IO

local M = {}

function M.read_exact(file, n)
    local data = file:read(n)
    if not data or #data ~= n then error('unexpected end of file') end
    return data
end

function M.read_int32(file)
    return string.unpack('<i4', M.read_exact(file, 4))
end

function M.read_float32(file)
    return string.unpack('<f', M.read_exact(file, 4))
end

function M.read_float32_array(file, n)
    local data, out = M.read_exact(file, n * 4), {}
    local pos = 1
    for i = 1, n do out[i], pos = string.unpack('<f', data, pos) end
    return out
end

function M.zeros(n)
    local out = {}
    for i = 1, n do out[i] = 0.0 end
    return out
end

function M.matmul(out, x, weight, rows, cols)
    for i = 1, rows do
        local sum, base = 0.0, (i - 1) * cols
        for j = 1, cols do sum = sum + weight[base + j] * x[j] end
        out[i] = sum
    end
    return out
end

function M.softplus(x)
    if x > 20 then return x end
    if x < -20 then return math.exp(x) end
    return math.log(1 + math.exp(x))
end

function M.silu(x)
    return x / (1 + math.exp(-x))
end

return M
