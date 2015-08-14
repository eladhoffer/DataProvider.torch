require 'torch'
DataProvider = {}
torch.include('DataProvider', 'Container.lua')
torch.include('DataProvider', 'LMDBProvider.lua')
torch.include('DataProvider', 'FileSearcher.lua')

return DataProvider
