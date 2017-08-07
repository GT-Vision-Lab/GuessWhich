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

-- optimization params
cmd:option('-batchSize', 30, 'Batch size (number of threads) (Adjust base on GRAM)');
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
-- local dataloader_ans = dofile('dataloader.lua')
dataloader:initializeQuestioner(opt, {'train', 'val'});
-- dataloader_ans:initialize(opt, {'train', 'val'});
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

-- path to save the model
-- local modelPath = opt.save_path;
-- creating the directory to save the model
-- paths.mkdir(modelPath);

-- image feature dimension
--modelParams.imgFeatDim = dataset['fv_im']:size(2);
-- Iterations per epoch
modelParams.numIterPerEpoch = modelParams.numTrainThreads
print(string.format('\n%d iter per epoch.', modelParams.numIterPerEpoch));
------------------------------------------------------------------------
-- Talking
------------------------------------------------------------------------
local imgPreds = torch.FloatTensor(numThreads * 10, 4096)
local dialogTable = {}
runningLoss = 0;
runningLossImg = 0;
runningLossImgRL = 0;

-- qModel:evaluate(dataloader, 'val');

for iter = 1, modelParams.numEpochs do
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        local inds = torch.LongTensor(1):fill(convId)
        local batch = dataloader:getIndexDataQuestioner(inds, qModel.params, 'train')

        local imgFeats = batch['img_feat']:view(-1, 1, 4096)
        imgFeats = imgFeats:repeatTensor(1, 10, 1);
        imgFeats = imgFeats:view(-1, 4096);
        local batchHist = batch['hist']:view(-1, batch['hist']:size(3)):t()
        local ques = torch.Tensor(30, 10):zero()
        local quesIn = torch.Tensor(30, 10):zero()
        local ansIn = torch.Tensor(30, 10):zero()
        local hist = torch.Tensor(40, 10):zero()
        hist[{{40 - batch['hist']:size(3) + 1, 40}, {1}}] = batchHist[{{},{1}}]

        local thread = {}
        for round = 1, 10 do
            -- if round == 1 then print(hist[{{}, {1}}]) end
            local qOut = qModel:generateSingleQuestionSample(dataloader, hist, {temperature = 1.0}, round)
            -- local qOut = qModel:generateSingleQuestion(dataloader, hist, {beamSize = 20}, round)

            -- Keeping track of question words
            quesIn[{{}, {round}}] = qOut[1]

            -- Setting up input for answerer
            ques[{{},{round}}] = utils.rightAlign(qOut[1]:view(1, 1, -1), torch.Tensor{qOut[2]}:view(1, 1))
            ques[{{ques:size(1) - qOut[2] + 1}, {round}}]:fill(0)

            local aOut = aModel:generateSingleAnswerSample(dataloader, {hist, imgFeats, ques}, {temperature = 1.0}, round)
            -- local aOut = aModel:generateSingleAnswer(dataloader, {hist, imgFeats, ques}, {beamSize = 20}, round)

            -- Keeping track of answer words
            ansIn[{{}, {round}}] = aOut[1]
            ansIn[{{aOut[2]}, {round}}] = 0

            local histVec = torch.Tensor(40):zero()
            for i = 1, qOut[2] + aOut[2] - 3 do
                if i < qOut[2] then
                    histVec[i] = qOut[1][i+1]
                else
                    if i <= 40 then
                        histVec[i] = aOut[1][i+2-qOut[2]]
                    end
                end
            end
            if round ~= 10 then
                if qOut[2]+aOut[2]-3 > 40 then
                    hist[{{},{round+1}}] = utils.rightAlign(histVec:view(1,1,-1), torch.Tensor{40}:view(1,1))
                else
                    hist[{{},{round+1}}] = utils.rightAlign(histVec:view(1,1,-1), torch.Tensor{qOut[2]+aOut[2]-3}:view(1,1))
                end
            end
            table.insert(thread, {question = qOut[3], answer = aOut[3]})
            print(qOut[3], aOut[3])
            if round == 10 then
                imgPreds[{{(convId - 1) * 10 + 1, convId * 10}, {}}] = qOut[4]:float()
            end
        end
        if convId % 1 == 0 then
            print('One iteration of SL')
            -- Make an SL update here
            qModel:trainIteration(dataloader);
            -- aModel:trainIteration(dataloader_ans);
        end

        -- Make a reinforce update here
        r_t = qModel:reinforce(imgFeats, quesIn, hist, convId)
        -- aModel:reinforce(imgFeats, ques, hist, ansIn, r_t, convId)

        table.insert(dialogTable, thread)
        
        if convId % 10 == 0 then
            local currentEpoch = iter - 1 + convId / modelParams.numIterPerEpoch;
            print(string.format('[%s][Epoch:%.02f][Iter:%d] Q Loss: %.02f ; Img SL Loss: %.02f ; Img RL Loss: %.02f ; lr : %f',
                                os.date(), currentEpoch, convId, runningLoss, runningLossImg, runningLossImgRL,
                                            qModel.optims.learningRate))
        end

        if convId % 100 == 0 then
            print('Evaluating')
            -- Evaluate losses on val
            qModel:evaluate(dataloader, 'val');

            if opt.identifier ~= 'test' then
                print('Saving models')
                -- Saving the models
                torch.save(string.format(qModelPath .. 'model_iter_%d.t7', modelParams.numIterPerEpoch * (iter-1) + convId),
                                                            {modelW = qModel.wrapperW,
                                                            optims = qModel.optims,
                                                            modelParams = questionerModelParams});
            end
        end

        collectgarbage();
    end
end

