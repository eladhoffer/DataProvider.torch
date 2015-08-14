local LMDBProvider = torch.class('DataProvider.LMDBProvider')

function LMDBProvider:__init(...)
    xlua.require('torch',true)
    require 'lmdb'
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider ',
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', req = true},
    {arg='Source', type='userdata', help='LMDB env', req=true},
    {arg='Verbose', type='boolean', help='display messages', default = false}

    )
    for x,val in pairs(args) do
        self[x] = val
    end
    self.Config = ...
end

function LMDBProvider:size()
    self.Source:open()
    local SizeData = self.Source:stat()['entries']
    self.Source:close()
    return SizeData
end

function LMDBProvider:cacheSeq(start_pos, num, data, labels)
    local num = num or 1
    self.Source:open()
    local txn = self.Source:txn(true)
    local cursor = txn:cursor()
    cursor:set(start_pos)

    local Data = data or {}
    local Labels = labels or {}
    for i = 1, num do
        local key, data = cursor:get()
        Data[i], Labels[i] = self.ExtractFunction(data, key)
        if i<num then
            cursor:next()
        end
    end
    cursor:close()
    txn:abort()
    self.Source:close()
    return Data, Labels
end

function LMDBProvider:cacheRand(keys, data, labels)
    local num
    if type(keys) == 'table' then
        num = #keys
    else
        num = keys:size(1)
    end
    self.Source:open()
    local txn = self.Source:txn(true)
    local Data = data or {}
    local Labels = labels or {}

    for i = 1, num do
        local item = txn:get(keys[i])
        Data[i], Labels[i] = self.ExtractFunction(item, keys[i])
    end
    txn:abort()
    self.Source:close()
    return Data, Labels
end

function LMDBProvider:threads(nthread)
    local nthread  = nthread or 1
    local config = self.Config
    local threads = require 'threads'
    threads.serialization('threads.sharedserialize')
    self.threads = threads(nthread,
    function()
        require 'lmdb'
        local DataProvider = require 'DataProvider'
    end,
    function(idx)
        workerProvider = DataProvider.LMDBProvider(config)
        lmdb.verbose = config.Verbose
    end
    )
end

function LMDBProvider:asyncCacheSeq(start, num, dataBuffer, labelsBuffer)
    self.threads:addjob(
    -- the job callback (runs in data-worker thread)
    function()
        local data, labels = workerProvider:cacheSeq(start,num,dataBuffer,labelsBuffer)
        return data, labels
    end,
    -- the endcallback (runs in the main thread)
    function(data,labels)
    end
    )
end

function LMDBProvider:asyncCacheRand(keys, dataBuffer, labelsBuffer)
    self.threads:addjob(
    -- the job callback (runs in data-worker thread)
    function()
        local data, labels = workerProvider:cacheRand(keys,dataBuffer,labelsBuffer)
        return data, labels
    end,
    -- the endcallback (runs in the main thread)
    function(data,labels)
    end
    )
end

function LMDBProvider:synchronize()
    return self.threads:synchronize()
end
