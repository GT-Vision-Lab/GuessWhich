-- Code to train svqa mode
-- author: satwik kottur
-------------------------------------------------------------------------------
-- S-VQA training model using VQA
require 'nn'
require 'nngraph'
require 'io'
require 'rnn'
-- require 'maskSoftmax'
require 'MaskTime'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
local opt = require 'opts';
print(opt)

-- seed for reproducibility
torch.manualSeed(1234);

------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
--local dataloader = dofile('dataloader.lua')
local dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, {'train', 'val'});
collectgarbage();

-- set default tensor based on gpu usage
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    --if opt.backend == 'cudnn' then require 'cudnn' end
    torch.setdefaulttensortype('torch.CudaTensor');
else
    torch.setdefaulttensortype('torch.DoubleTensor');
end

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
-- transfer all options to model
local modelParams = opt;

-- transfer parameters from datalaoder to model
paramNames = {'numTrainThreads', 'numTestThreads', 'numValThreads',
                'vocabSize', 'maxQuesCount', 'maxQuesLen', 'maxAnsLen',
                'numOptions'};
for _, value in pairs(paramNames) do
    modelParams[value] = dataloader[value];
end

-- path to save the model
local modelPath = opt.save_path;
-- creating the directory to save the model
paths.mkdir(modelPath);

-- image feature dimension
--modelParams.imgFeatDim = dataset['fv_im']:size(2);
-- Iterations per epoch
modelParams.numIterPerEpoch = math.ceil(modelParams.numTrainThreads /
                                                modelParams.batchSize);
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'modelAnswerer'
local svqaModel = VisDialAModel(modelParams);

if opt.load_path_a ~= '' then
    print('Initializing from ' .. opt.load_path_a)
    answererModel = torch.load(opt.load_path_a)
    svqaModel.wrapperW:copy(answererModel.modelW)
end

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
-- validation accuracy
print('Evaluation before training..')
svqaModel:evaluate(dataloader, 'val');
-- if opt.useMI then
    -- print('\nComputing probability before training..')
    -- svqaModel:computeProbability(dataloader, 'val');
-- end

-- print('\nRetrieval before training..')
-- svqaModel:retrieve(dataloader, 'val');
collectgarbage()

runningLoss = 0;
for iter = 1, modelParams.numEpochs * modelParams.numIterPerEpoch do
    -- forward and backward propagation
    svqaModel:trainIteration(dataloader);
    
    -- evaluate on val and save model every epoch
    if iter % (modelParams.numIterPerEpoch) == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch;
        
        -- save model and optimization parameters
        torch.save(string.format(modelPath .. 'model_epoch_%d.t7', currentEpoch),
                                                    {modelW = svqaModel.wrapperW,
                                                    optims = svqaModel.optims,
                                                    modelParams = modelParams});
        -- validation accuracy
        -- svqaModel:evaluate(dataloader, 'val');
        -- if opt.useMI then svqaModel:computeProbability(dataloader, 'val'); end
        svqaModel:retrieve(dataloader, 'val');
        --svqaModel:retrieve(dataloader, 'val', {0.0, 0.1, 0.5, 1.0});
    end
    
    -- print after every few iterations
    if iter % 100 == 0 then
        local currentEpoch = iter / modelParams.numIterPerEpoch;
        
        -- print current time, running average, learning rate, iteration, epoch
        print(string.format('[%s][Epoch:%.02f][Iter:%d] Loss: %.02f ; lr : %f',
                                os.date(), currentEpoch, iter, runningLoss, 
                                            svqaModel.optims.learningRate))
    end
    if iter % 10 == 0 then collectgarbage(); end
end

-- Saving the final model
torch.save(modelPath .. 'model_final.t7', {modelW = svqaModel.wrapperW,
                                            optims = svqaModel.optims,
                                            modelParams = modelParams});
------------------------------------------------------------------------
