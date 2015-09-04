local ffi = require 'ffi'

local Container = torch.class('DataProvider.Container')
function Container:__init(...)
    xlua.require('torch',true)

    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a Container ',
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

    if self.CacheFiles then
        os.execute('mkdir -p "' .. paths.dirname(self:batchFilename(1)) .. '"')
    end


    self.CurrentBatch = 1
    self.NumBatch = 1
    self.Data = torch.Tensor():type(self.TensorType)
    self.Labels = torch.Tensor():type(self.TensorType)

    self:reset()
    if torch.type(self.Source) =='table' then
        self:loadFrom(self.Source[1], self.Source[2])
    end
end
function Container:loadFrom(data,labels)
    self.Data = data
    self.Labels = labels
end

function Container:size()
    if self.Data:dim() == 0 then
        return 0
    end
    return self.Data:size(1)
end

function Container:reset()
    self.CurrentBatch = 1
    self.NumBatch = 1
    self.CurrentItemSource = 1
end


function Container:__tostring__()
    local str = 'Container:\n'
    if self:size() > 0 then
        str = str .. ' + num samples : '.. self:size()
    else
        str = str .. ' + empty set...'
    end
    return str
end


function Container:batchFilename(num)
    return paths.concat(self.CachePrefix, self.Name .. '_Batch' .. num)
end


function Container:shuffleItems()
    local RandOrder = torch.randperm(self:size()):long()
    self.Data = self.Data:index(1,RandOrder)

    if self.Labels and self.Labels:dim() > 0 then
        self.Labels = self.Labels:index(1,RandOrder)
    end
    if self.Verbose then
        print('(Container)===>Shuffling Items')
    end
end


function Container:getItem(location)
    return self:getItems(location,1)
end

function Container:getItems(location,num)
    --Assumes location and num are valid
    local num = num or 1
    local data = self.Data:narrow(1,location,num)
    if self.Labels then
        return data, self.Labels:narrow(1,location,num)
    else
        return data
    end
end

function Container:currentItemCount()
    return self.CurrentItemSource
end

function Container:loadBatch(batchNumber)
    local batchNumber = batchNumber or self.NumBatch

    local batchFilename = self:batchFilename(batchNumber)
    if paths.filep(batchFilename) then
        if self.Verbose then
            print('(Container)===>Loading Batch N.' .. batchNumber .. ' From ' .. batchFilename)
        end
        local Batch = torch.load(batchFilename)
        self.Data = Batch.Data:type(self.TensorType)
        self.Labels = Batch.Labels:type(self.TensorType)
        self.NumBatch = batchNumber
        self.CurrentItemSource = self.CurrentItemSource + Batch.Data:size(1)
        return true
    else
        return false
    end
end

function Container:saveBatch()
    if self.Verbose then
        print('(Container)===>Saving Batch')
    end
    torch.save(self:batchFilename(self.NumBatch), {Data = self.Data, Labels = self.Labels})
end

function Container:createBatch()
    if not self.Source then
        return nil
    end
    if self.CurrentItemSource > self.Source:size() then
        if not self.AutoLoad then
            return nil
        end

        if not self.Source:getNextBatch() then
            return nil
        else
            self.CurrentItemSource = 1
        end
    end
    if self.Verbose then
        print('(Container)===>Creating Batch')
    end

    local NumInBatch = math.min(self.Source:size() - self.CurrentItemSource + 1, self.MaxNumItems)
    local source_data, source_labels = self.Source:getItems(self.CurrentItemSource, NumInBatch)
    local data, labels = self.ExtractFunction(source_data,source_labels)
    if self.CopyData then
        self.Data:resize(data:size())
        self.Data:copy(data)
        if labels then
          self.Labels:resize(labels:size())
          self.Labels:copy(labels)
        end
    else
        self.Data = data
        self.Labels = labels
    end
    self.CurrentItemSource = self.CurrentItemSource + NumInBatch
    return true

end

function Container:getNextBatch()
    if self:loadBatch() then
        self.NumBatch = self.NumBatch + 1
        return self.Data, self.Labels
    elseif self:createBatch() then
        if self.CacheFiles then
            self:saveBatch()
        end
        self.NumBatch = self.NumBatch + 1
        return self.Data, self.Labels
    else
        return nil
    end
end


function Container:apply(fData,fLabels)
    local function applyData(func, data)
        if func == nil or not data then
            return data
        end
        local newData = func(data[1])
        if (torch.type(newData) == 'number' or torch.type(data) == 'number') or (newData:nElement()==data[1]:nElement()) then --inplace
            newData = data
        else
            if torch.type(newData) == 'number' then
                newData:resize(data:size(1))
            else
                newData:resize(data:size(1),unpack(newData:size():totable()))
            end
        end
        for i=1,data:size(1) do
            newData[i]:copy(func(data[i]))
        end
        return newData
    end

    self.Data = applyData(fData, self.Data)
    self.Labels = applyData(fLabels, self.Labels)

end

function Container:normalize(normType, mean, std)

    --normType can be either
    -- 'simple' - whole sample (mean and std are numbers)
    -- 'channel' - by image channels (mean and std are vectors)
    -- 'image' - mean and std images
    --If mean and std are supplied - normalization is done with them as constants

    local normType = normType or 'simple'

    if normType == 'simple' then
        mean = mean or self.Data:mean()
        std = std or self.Data:std()

        self.Data:add(-mean):div(std)

    else
        local size, channels, y_size, x_size = unpack(self.Data:size():totable())
        if normType == "channel" then
            local function channelMap(x, f, cNum)
                local values = torch.Tensor(cNum)
                for c=1, cNum do
                    values[c] = f(x:select(2,c))
                end
                return values
            end
            mean = mean or channelMap(self.Data, torch.mean, channels):view(1, channels, 1, 1)
            std = std or channelMap(self.Data, torch.std, channels):view(1, channels, 1, 1)
        elseif normType == "image" then
            mean = mean or self.Data:view(size,-1):mean(1):view(1, channels, y_size, x_size)
            std = std or self.Data:view(size,-1):std(1):view(1, channels, y_size, x_size)
        end
        self.Data:add(-1, mean:typeAs(self.Data):expand(size,channels,y_size,x_size))
        self.Data:cdiv(std:typeAs(self.Data):expand(size,channels,y_size,x_size))
    end

    return mean, std
end

return Container
