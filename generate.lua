-- generate.lua - autoregressive generation over constant-size recurrent state

local sample = require('sample')
local M = {}

function M.generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
    local tokens = tokenizer:encode(prompt, false)
    if #tokens == 0 then tokens[1] = model.config.bos_token_id end
    local state, logits = model:create_state(), nil

    io.write(prompt)
    io.flush()
    for _, token in ipairs(tokens) do logits = model:forward(token, state) end

    local generated = {}
    for _ = 1, max_tokens do
        local token = sample.next(logits, temperature, top_p)
        if token == model.config.eos_token_id then break end
        generated[#generated + 1] = token
        io.write(tokenizer:decode(token))
        io.flush()
        logits = model:forward(token, state)
    end
    io.write('\n')
    return generated
end

return M
