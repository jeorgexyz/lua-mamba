-- tokenizer.lua - GPT-NeoX/GPT-2 byte-level BPE

local utils = require('utils')
local Tokenizer = {}
Tokenizer.__index = Tokenizer

local function bytes_to_unicode()
    local used, encoder, decoder = {}, {}, {}
    for b = 33, 126 do used[b] = true end
    for b = 161, 172 do used[b] = true end
    for b = 174, 255 do used[b] = true end
    local extra = 0
    for b = 0, 255 do
        local cp
        if used[b] then cp = b else cp, extra = 256 + extra, extra + 1 end
        local char = utf8.char(cp)
        encoder[b], decoder[char] = char, b
    end
    return encoder, decoder
end

local function chars(text)
    local out = {}
    for start in utf8.codes(text) do
        local finish = utf8.offset(text, 2, start)
        out[#out + 1] = text:sub(start, finish and finish - 1 or #text)
    end
    return out
end

function Tokenizer.new(path)
    local file = assert(io.open(path, 'rb'), 'cannot open tokenizer: ' .. path)
    assert(utils.read_exact(file, 4) == 'LMT1', 'not a lua-mamba tokenizer')
    assert(utils.read_int32(file) == 1, 'unsupported tokenizer version')
    local self = setmetatable({}, Tokenizer)
    self.vocab_size = utils.read_int32(file)
    self.bos_token_id = utils.read_int32(file)
    self.eos_token_id = utils.read_int32(file)
    local n_merges = utils.read_int32(file)
    self.vocab, self.ids = {}, {}
    for id = 0, self.vocab_size - 1 do
        local token = utils.read_exact(file, utils.read_int32(file))
        self.vocab[id], self.ids[token] = token, id
    end
    self.merges = {}
    for rank = 0, n_merges - 1 do
        local a = utils.read_exact(file, utils.read_int32(file))
        local b = utils.read_exact(file, utils.read_int32(file))
        self.merges[a .. string.char(0) .. b] = rank
    end
    file:close()
    self.byte_encoder, self.byte_decoder = bytes_to_unicode()
    return self
end

local contractions = { ['s'] = true, ['t'] = true, ['re'] = true,
    ['ve'] = true, ['m'] = true, ['ll'] = true, ['d'] = true }

local function category(byte)
    if byte == 9 or byte == 10 or byte == 13 or byte == 32 then return 'space' end
    if byte >= 48 and byte <= 57 then return 'number' end
    if (byte >= 65 and byte <= 90) or (byte >= 97 and byte <= 122) or byte >= 128 then
        return 'letter'
    end
    return 'other'
end

local function next_piece(text, pos)
    local n, prefix = #text, ''
    if text:byte(pos) == 32 then
        local last = pos
        while last <= n and text:byte(last) == 32 do last = last + 1 end
        if last > n then return text:sub(pos), n + 1 end
        if last - pos > 1 then return text:sub(pos, last - 2), last - 1 end
        prefix, pos = ' ', pos + 1
    elseif category(text:byte(pos)) == 'space' then
        local last = pos + 1
        while last <= n and category(text:byte(last)) == 'space' do last = last + 1 end
        return text:sub(pos, last - 1), last
    end

    if text:byte(pos) == 39 then
        for _, len in ipairs({ 2, 1 }) do
            local suffix = text:sub(pos + 1, pos + len)
            if contractions[suffix] then return prefix .. text:sub(pos, pos + len), pos + len + 1 end
        end
    end
    local kind, last = category(text:byte(pos)), pos + 1
    while last <= n and category(text:byte(last)) == kind do
        if text:byte(last) == 39 and kind ~= 'other' then break end
        last = last + 1
    end
    return prefix .. text:sub(pos, last - 1), last
end

function Tokenizer:bpe(piece)
    local encoded = {}
    for i = 1, #piece do encoded[#encoded + 1] = self.byte_encoder[piece:byte(i)] end
    local word = chars(table.concat(encoded))
    while #word > 1 do
        local best, left, right
        for i = 1, #word - 1 do
            local rank = self.merges[word[i] .. string.char(0) .. word[i + 1]]
            if rank and (not best or rank < best) then best, left, right = rank, word[i], word[i + 1] end
        end
        if not best then break end
        local merged, i = {}, 1
        while i <= #word do
            if i < #word and word[i] == left and word[i + 1] == right then
                merged[#merged + 1], i = left .. right, i + 2
            else
                merged[#merged + 1], i = word[i], i + 1
            end
        end
        word = merged
    end
    return word
end

function Tokenizer:encode(text, add_bos)
    local ids = {}
    if add_bos then ids[#ids + 1] = self.bos_token_id end
    local pos = 1
    while pos <= #text do
        local piece
        piece, pos = next_piece(text, pos)
        for _, token in ipairs(self:bpe(piece)) do
            local id = self.ids[token]
            if id == nil then error('tokenizer vocabulary has no token for: ' .. token) end
            ids[#ids + 1] = id
        end
    end
    return ids
end

function Tokenizer:decode(id)
    if id == self.bos_token_id or id == self.eos_token_id then return '' end
    local token, out = self.vocab[id], {}
    if not token then return '' end
    for _, char in ipairs(chars(token)) do
        local byte = self.byte_decoder[char]
        if byte then out[#out + 1] = string.char(byte) end
    end
    return table.concat(out)
end

return Tokenizer
