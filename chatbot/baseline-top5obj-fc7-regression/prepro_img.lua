require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson')
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json_train','train_imgs.json','path to the json file containing vocab and answers')
cmd:option('-input_json_test','val_imgs.json','path to the json file containing vocab and answers')
cmd:option('-image_root','/srv/share/data/mscoco/images','path to the image root')
cmd:option('-cnn_proto', 'models/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('-cnn_model', 'models/VGG_ILSVRC_16_layers.caffemodel', 'path to the cnn model')
cmd:option('-batch_size', 20, 'batch_size')

cmd:option('-out_name', 'data_img.h5', 'output name')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

cmd:option('-img_size', 224)
cmd:option('-layer_num', 39)

opt = cmd:parse(arg)
print(opt)

net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:remove()
print(net, #net.modules)
net:evaluate()
net=net:cuda()

function loadim(imname)
    im=image.load(imname)

    if im:size(1) == 1 then
        im = im:repeatTensor(3, 1, 1)
    elseif im:size(1) == 4 then
        im = im[{{1,3}, {}, {}}]
    end

    im = image.scale(im, opt.img_size, opt.img_size)
    local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    im = im:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(im)
    im:add(-1, mean_pixel)
    return im
end

local file = io.open(opt.input_json_train, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file) do
    table.insert(train_list, string.format('%s/train2014/COCO_train2014_%012d.jpg', opt.image_root, imname))
end

local file = io.open(opt.input_json_test, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local test_list={}
for i,imname in pairs(json_file) do
    table.insert(test_list, string.format('%s/val2014/COCO_val2014_%012d.jpg', opt.image_root, imname))
end

local ndims=4096
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.FloatTensor(sz,1000)
local feat_train_fc7=torch.FloatTensor(sz,4096)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,opt.img_size,opt.img_size)
    for j=1,r-i+1 do
        ims[j]=loadim(train_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[opt.layer_num].output:float()
    feat_train_fc7[{{i,r},{}}]=net.modules[37].output:float()
    collectgarbage()
end

print('DataLoader loading h5 file: ', 'data_train')
local sz=#test_list
local feat_test=torch.FloatTensor(sz,1000)
local feat_test_fc7=torch.FloatTensor(sz,4096)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,opt.img_size,opt.img_size)
    for j=1,r-i+1 do
        ims[j]=loadim(test_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[opt.layer_num].output:float()
    feat_test_fc7[{{i,r},{}}]=net.modules[37].output:float()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_train_fc7', feat_train_fc7:float())
train_h5_file:write('/images_val', feat_test:float())
train_h5_file:write('/images_val_fc7', feat_test_fc7:float())
train_h5_file:close()

