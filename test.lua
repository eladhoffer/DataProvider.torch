require 'FileSearcher'
testFiles = FileSearcher{
    Name = 'Filenames',
    CachePrefix = './cache',
    MaxNumItems = 1e8,
    CacheFiles = true,
    PathList = {'../ImageNet-Training/', '../ConvNet-torch/'},
    SubFolders = false,
    Extensions = {'lua','txt'}
}
