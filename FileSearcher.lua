local ffi = require 'ffi'
--------------------------File Searcher-------------------------------

local FileSearcher, parent = torch.class('DataProvider.FileSearcher', 'DataProvider.Container')
function FileSearcher:__init(...)
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider ',
    {arg='MaxNumItems', type='number', help='Number of Elements in each Batch', default = 1e8},
    {arg='Name', type='string', help='Name of DataProvider',req = true},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='CachePrefix', type='string', help='path to caches data',default = '.'},
    {arg='CacheFiles', type='boolean', help='cache data into files', default=false},
    {arg='SubFolders', type='boolean', help='Recursive check for folders', default=false},
    {arg='Extensions', type='table', help='File Extensions to search for'},
    {arg='Shuffle', type='boolean', help='Shuffle list before saving', default=false},
    {arg='PathList', type='table', help='Table of paths to search for files in', req = true},
    {arg='MaxFilenameLength', type='number', help='Maximum length of filename', default = 100},
    {arg='Verbose', type='boolean', help='display messages', default = false}

    )

    for x,val in pairs(args) do
        self[x] = val
    end
    self.Source = {}
    self.TensorType = 'torch.CharTensor'

    self.NumBatch = 1
    if self.CacheFiles then
        os.execute('mkdir -p "' .. paths.dirname(self:batchFilename(1)) .. '"')
    end


    self.CurrentBatch = 1
    self.NumBatch = 1

    self.Data = torch.Tensor():type(self.TensorType)
    self:reset()

    if not self:loadBatch() then
        local tmpFile = path.tmpname()
        --local pathString = '"' .. paths.concat(self.PathList[1]) .. '"'
        local pathString = '"' .. table.concat(self.PathList, '" "') .. '" '
        local findCmd = 'find ' .. pathString
        local subfolders = args.SubFolders
        if not args.SubFolders then
            findCmd = findCmd .. '-maxdepth 1 '
        end
        findCmd = findCmd .. '-type f '
        local filePattern = ''
        if self.Extensions then
            filePattern = '\\( -iname "*.'
            filePattern = filePattern .. table.concat(self.Extensions, '" -o -iname "*.') .. '" \\) '
        end
        findCmd = findCmd .. filePattern .. '> ' .. tmpFile
        sys.execute(findCmd)
        local numItems = tonumber(sys.execute('wc -l ' .. tmpFile .. " | cut -f1 -d ' '"))
        local maxStringLength = tonumber(sys.execute('wc -L ' .. tmpFile .. " | cut -f1 -d ' '")) + 1

        if numItems == 0 then
            return
        end
        self.Data:resize(numItems, maxStringLength):zero()
        local cData = self.Data:data()
        for line in io.lines(tmpFile) do
            ffi.copy(cData, line)
            cData = cData + maxStringLength
        end
    end

end

function FileSearcher:getItem(location)
    --Assumes location and num are valid
    return ffi.string(torch.data(self.Data[location]))
end
function FileSearcher:getItems(location,num)
    --Assumes location and num are valid
    local num = num or 1
    local data = {}
    for i=1, num do
        data[i] = ffi.string(torch.data(self.Data[location+i-1])) -- FileSearcher.__index(self, location + i -1)
    end
    return data
end
