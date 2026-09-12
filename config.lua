-- config.lua - lua-mamba binary header

local utils = require('utils')
local Config = {}

function Config.read(file)
    if utils.read_exact(file, 4) ~= 'LMB1' then error('not a lua-mamba checkpoint') end
    local version = utils.read_int32(file)
    if version ~= 1 then error('unsupported checkpoint version: ' .. version) end

    local c = {
        version = version,
        d_model = utils.read_int32(file),
        d_inner = utils.read_int32(file),
        d_state = utils.read_int32(file),
        d_conv = utils.read_int32(file),
        dt_rank = utils.read_int32(file),
        n_layers = utils.read_int32(file),
        vocab_size = utils.read_int32(file),
        padded_vocab_size = utils.read_int32(file),
        bos_token_id = utils.read_int32(file),
        eos_token_id = utils.read_int32(file),
        shared_classifier = utils.read_int32(file) == 1,
        conv_bias = utils.read_int32(file) == 1,
        norm_eps = utils.read_float32(file),
    }
    assert(c.d_model > 0 and c.d_inner > 0 and c.n_layers > 0, 'invalid dimensions')
    assert(c.d_state > 0 and c.d_conv > 0 and c.dt_rank > 0, 'invalid mixer dimensions')
    assert(c.vocab_size > 0 and c.vocab_size <= c.padded_vocab_size, 'invalid vocabulary')
    return c
end

return Config
