local MultiThreaded, parent = torch.class('DataProvider.MultiThreaded', 'DataProvider.Container')
function MultiThreaded:__init(...)
  xlua.require('torch',true)

  local args = dok.unpack(
  {...},
  'InitializeData',
  'Initializes a MultiThreaded Container ',
  {arg='NumThreads', type='number', help='Number ofThreads',defalut = 1},
  {arg='MaxNumItems', type='number', help='Number of Elements in each Batch',defalut = 1e6},
  {arg='Name', type='string', help='Name of Container',default = ''},
  {arg='TensorType', type='string', help='Type of Tensor', default = 'torch.FloatTensor'},
  {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
  {arg='Source', type='table', help='source of Container', req=true},
  {arg='CachePrefix', type='string', help='path to caches data',default = '.'},
  {arg='CacheFiles', type='boolean', help='cache data into files', default=false},
  {arg='AutoLoad', type='boolean', help='load next data automaticaly', default=false},
  {arg='CopyData', type='boolean', help='Copies data instead of referencing it ', default = true},
  {arg='Verbose', type='boolean', help='display messages', default = false}
  )

  for x,val in pairs(args) do
      self[x] = val
  end

  self.Config = ...

  self.currBuffer = 1
  self.BufferSources = {}
  for i=1,numThreads do
      self.BufferSources[i] = {torch.Tensor():type(self.TensorType),torch.Tensor():type(self.TensorType)}
  end
  local config = self.Config
  local buffers = self.BufferSources
  local threads = require 'threads'
  threads.serialization('threads.sharedserialize')
  self.threads = threads(self.NumThreads,
  function()
      local DataProvider = require 'DataProvider'
  end,
  function(idx)
      workerProvider = DataProvider.LMDBProvider(config)
      lmdb.verbose = config.Verbose
  end
  )

  local currBatch = 1

  local BufferNext = function()
      currBuffer = currBuffer%numBuffers +1
      if currBatch > dataIndices:size(1) then BufferSources[currBuffer] = nil return end
      local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch]+1)
      BufferSources[currBuffer].Data:resize(sizeBuffer ,unpack(config.InputSize))
      BufferSources[currBuffer].Labels:resize(sizeBuffer)
      DB:asyncCacheSeq(config.Key(dataIndices[currBatch]), sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
      currBatch = currBatch + 1
  end
  function MultiThreaded:threads(nthread)
      local nthread  = nthread or 1

  end


  function MultiThreaded:asyncCacheRand(keys, dataBuffer, labelsBuffer)
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

  function MultiThreaded:synchronize()
      return self.threads:synchronize()
  end
