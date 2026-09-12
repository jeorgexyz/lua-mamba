-- sample.lua - greedy, temperature, and nucleus sampling

local Sample = {}

function Sample.argmax(logits)
    local best = 1
    for i = 2, #logits do if logits[i] > logits[best] then best = i end end
    return best - 1
end

function Sample.next(logits, temperature, top_p)
    temperature, top_p = temperature or 1.0, top_p or 1.0
    if temperature <= 0 then return Sample.argmax(logits) end

    local max = logits[1]
    for i = 2, #logits do if logits[i] > max then max = logits[i] end end
    local probs, sum = {}, 0.0
    for i = 1, #logits do
        local p = math.exp((logits[i] - max) / temperature)
        probs[i], sum = { id = i - 1, p = p }, sum + p
    end
    for i = 1, #probs do probs[i].p = probs[i].p / sum end

    if top_p < 1.0 then
        table.sort(probs, function(a, b) return a.p > b.p end)
        local kept, cumulative = {}, 0.0
        for _, item in ipairs(probs) do
            kept[#kept + 1], cumulative = item, cumulative + item.p
            if cumulative >= top_p then break end
        end
        probs, sum = kept, cumulative
    else
        sum = 1.0
    end

    local pick, cumulative = math.random() * sum, 0.0
    for _, item in ipairs(probs) do
        cumulative = cumulative + item.p
        if pick <= cumulative then return item.id end
    end
    return probs[#probs].id
end

return Sample
