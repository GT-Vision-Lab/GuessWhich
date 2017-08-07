-- Code to make the questioner and answerer models talk. Fun! :D

require 'nn';
require 'nngraph';
require 'rnn';
require 'MaskTime';
utils = dofile('utils.lua')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Make the VisDial models talk to each other')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data/visdial_0.5/data_img.h5','h5file path with image feature')
cmd:option('-input_ques_h5','data/visdial_0.5/chat_processed_data.h5','h5file file with preprocessed questions')
cmd:option('-input_ques_opts_h5','utils/working/qoptions_cap_qatm1mean_sample25k_visdial_0.5_test.h5','h5file file with preprocessed question options')
cmd:option('-input_json','data/visdial_0.5/chat_processed_params.json','json path with info and vocab')
cmd:option('-identifier', '', 'default is blank')

cmd:option('-load_path_q', 'models/model-2-20-2017-11:20:55-im-hist-enc-dec-questioner/model_epoch_15.t7', 'path to saved questioner model')
cmd:option('-load_path_a', 'models/model-2-14-2017-22:43:51-im-hist-enc-dec/model_epoch_15.t7', 'path to saved answerer model')
cmd:option('-result_path', 'results', 'path to save generated results')

cmd:option('-qFreeze', 0, 'whether to freeze the questioner')

-- optimization params
cmd:option('-batchSize', 15, 'Batch size (number of threads) (Adjust base on GRAM)');
cmd:option('-probSampleSize', 50, 'Number of samples for computing probability');
cmd:option('-learningRate', 1e-3, 'Learning rate');
cmd:option('-dropout', 0, 'Dropout for language model');
cmd:option('-numEpochs', 400, 'Epochs');
cmd:option('-LRateDecay', 10, 'After lr_decay epochs lr reduces to 0.1*lr');
cmd:option('-lrDecayRate', 0.9997592083, 'Decay for learning rate')
cmd:option('-minLRate', 5e-5, 'Minimum learning rate');
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
local opt = cmd:parse(arg);
print(opt)

-- identifier
if opt.identifier == '' then print('NO IDENTIFIER') os.exit() end

-- seed for reproducibility
torch.manualSeed(1234);

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
-- Read saved model and parameters
------------------------------------------------------------------------
local questionerModel = torch.load(opt.load_path_q);
local answererModel = torch.load(opt.load_path_a);

-- transfer all options to model
local questionerModelParams = questionerModel.modelParams;
local answererModelParams = answererModel.modelParams;

opt.img_norm = answererModelParams.img_norm;
opt.model_name_q = questionerModelParams.model_name;
opt.model_name_a = answererModelParams.model_name;
print('Questioner', opt.model_name_q)
print('Answerer', opt.model_name_a)

-- add flags for various configurations
-- additionally check if its imitation of discriminative model
if string.match(opt.model_name_a, 'hist') then opt.useHistory = true; end
if string.match(opt.model_name_a, 'im') then opt.useIm = true; end

local curTime = os.date('*t', os.time());
-- create another folder to avoid clutter
local qModelPath = string.format('models/model-%d-%d-%d-%d:%d:%d-%s_%s/',
                                curTime.month, curTime.day, curTime.year, 
                                curTime.hour, curTime.min, curTime.sec, opt.model_name_q,
                                opt.identifier)
local aModelPath = string.format('models/model-%d-%d-%d-%d:%d:%d-%s_%s/',
                                curTime.month, curTime.day, curTime.year, 
                                curTime.hour, curTime.min, curTime.sec, opt.model_name_a,
                                opt.identifier)

print('Q-Model save path: ' .. qModelPath)
print('A-Model save path: ' .. aModelPath)

if opt.identifier ~= 'test' then
    paths.mkdir(qModelPath)
    -- paths.mkdir(aModelPath)
end
------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initializeQuestioner(opt, {'train', 'val'});
collectgarbage();

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'modelQuestioner'
require 'modelAnswerer'
print('Using models from '.. questionerModelParams.model_name)
print('Using models from '.. answererModelParams.model_name)

local qModel = VisDialQModel(questionerModelParams);
local aModel = VisDialAModel(answererModelParams);

-- copy the weights from loaded model
qModel.wrapperW:copy(questionerModel.modelW);
aModel.wrapperW:copy(answererModel.modelW);

------------------------------------------------------------------------
-- Setting model parameters
------------------------------------------------------------------------
local numThreads = dataloader.numThreads['train'];
-- transfer all options to model
local modelParams = opt;

-- transfer parameters from datalaoder to model
paramNames = {'numTrainThreads', 'numTestThreads', 'numValThreads',
                'vocabSize', 'maxQuesCount', 'maxQuesLen', 'maxAnsLen',
                'numOptions'};
for _, value in pairs(paramNames) do
    modelParams[value] = dataloader[value];
end

-- Iterations per epoch
modelParams.numIterPerEpoch = math.ceil(modelParams.numTrainThreads / opt.batchSize);
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));
------------------------------------------------------------------------
-- Talking
------------------------------------------------------------------------
-- local imgPreds = torch.FloatTensor(numThreads * 10, 4096)
local dialogTable = {}
meanReward = 0;
runningLoss = 0;
runningALoss = 0;
runningLossImg = 0;
runningLossImgRL = 0;

-- SL for `numSLRounds`, RL for `10 - numSLRounds`.
numSLRounds = 8;
maxRounds = 10;

for trainIter = 1, modelParams.numEpochs * modelParams.numIterPerEpoch do
    local inds = torch.LongTensor(opt.batchSize):random(1, numThreads);
    local batch = dataloader:getIndexDataQuestioner(inds, qModel.params, 'train')

    local imgFeats = batch['img_feat']:view(-1, 1, qModel.params.imgFeatureSize)
    imgFeats = imgFeats:repeatTensor(1, maxRounds, 1);
    imgFeats = imgFeats:view(-1, qModel.params.imgFeatureSize);

    -- 10-round SL data
    local batchHist = batch['hist']:view(-1, batch['hist']:size(3)):t()
    local questionIn = batch['question_in']:view(-1, batch['question_in']:size(3)):t();
    local questionOut = batch['question_out']:view(-1, batch['question_out']:size(3)):t();
    local batchQues = utils.rightAlign(batch['question_out'], batch['question_len']-1):view(-1, batch['question_out']:size(3)):t()
    local answerIn = batch['answer_in']:view(-1, batch['answer_in']:size(3)):t()
    local answerOut = batch['answer_out']:view(-1, batch['answer_out']:size(3)):t();

    -- copies with zeroed out RL rounds 
    local ques = torch.Tensor(30, maxRounds * opt.batchSize):zero() -- right-aligned, answerer input, 1st word to ? token
    local quesIn = torch.Tensor(30, maxRounds * opt.batchSize):zero()
    local quesOut = torch.Tensor(30, maxRounds * opt.batchSize):zero()
    local hist = torch.Tensor(40, maxRounds * opt.batchSize):zero()

    local ansIn = torch.Tensor(30, maxRounds * opt.batchSize):zero()
    local ansOut = torch.Tensor(30, maxRounds * opt.batchSize):zero()

    hist[{{40 - batch['hist']:size(3) + 1, 40}, {}}]:copy(batchHist);
    quesIn[{{1, batch['question_in']:size(3)}, {}}]:copy(questionIn);
    quesOut[{{1, batch['question_out']:size(3)}, {}}]:copy(questionOut);
    ques[{{30 - batchQues:size(1) + 1, 30}, {}}]:copy(batchQues);

    ansIn[{{1, batch['answer_in']:size(3)}, {}}]:copy(answerIn);
    ansOut[{{1, batch['answer_out']:size(3)}, {}}]:copy(answerOut);

    -- Make a reinforce update here
    r_t = qModel:annealedReinforceBatched({hist, imgFeats, quesIn, quesOut}, numSLRounds, {batchSize = opt.batchSize, freeze = opt.qFreeze})

    -- table.insert(dialogTable, thread)
    local currentEpoch = trainIter / modelParams.numIterPerEpoch;
    
    if trainIter % 10 == 0 then
        print(string.format('[%s][SL:%d/%d][Epoch:%.02f][Iter:%d][Q Loss: %.02f][A Loss: %.02f][Img Loss: %.02f][Neg R: %.06f][lr: %f]',
                            os.date(), numSLRounds, maxRounds, currentEpoch, trainIter, runningLoss, runningALoss, runningLossImgRL, 100 * meanReward,
                                        qModel.optims.learningRate))
    end

    if trainIter % 500 == 0 then
        print('Evaluating')
        -- Evaluate losses on val
        qModel:evaluate(dataloader, 'val');

        if opt.identifier ~= 'test' then
            print('Saving models')
            local currentEpoch = trainIter / modelParams.numIterPerEpoch;
            -- Saving the models
            torch.save(string.format(qModelPath .. 'model_iter_%d.t7', trainIter),
                                                        {modelW = qModel.wrapperW,
                                                        optims = qModel.optims,
                                                        modelParams = questionerModelParams});
        end
        collectgarbage();
    end

    -- reduce number of SL rounds per epoch
    if currentEpoch ~= 0 and currentEpoch % 1 == 0 then
        numSLRounds = numSLRounds - 1;
        if numSLRounds == 0 then numSLRounds = 1; end
    end

    collectgarbage();
end
