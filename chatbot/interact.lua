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
cmd:option('-input_img_h5','/srv/share/abhshkdz/visdial_rl_iccv17/visdial_0.5/data_img.h5','h5file path with image feature')
cmd:option('-input_ques_h5','/srv/share/abhshkdz/visdial_rl_iccv17/visdial_0.5/chat_processed_data.h5','h5file file with preprocessed questions')
cmd:option('-input_ques_opts_h5','/srv/share/abhshkdz/visdial_rl_iccv17/qoptions_cap_qatm1mean_sample25k_visdial_0.5_test.h5','h5file file with preprocessed question options')
cmd:option('-input_json','/srv/share/abhshkdz/visdial_rl_iccv17/visdial_0.5/chat_processed_params.json','json path with info and vocab')
cmd:option('-identifier', 'test', 'default identifier is blank string')

-- cmd:option('-load_path_q', 'models/model-2-20-2017-11:20:55-im-hist-enc-dec-questioner/model_epoch_15.t7', 'path to saved questioner model')
cmd:option('-load_path_q', '/srv/share/abhshkdz/visdial_rl_iccv17/code/abhishek/checkpoints/qbot_hre_qih_sl.t7', 'path to saved questioner model')
cmd:option('-load_path_a', '/srv/share/abhshkdz/visdial_rl_iccv17/code/abhishek/checkpoints/abot_hre_qih_sl.t7', 'path to saved answerer model')
cmd:option('-result_path', 'results', 'path to save generated results')

-- optimization params
cmd:option('-batchSize', 200, 'Batch size (number of threads) (Adjust base on GRAM)');
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

local opt = cmd:parse(arg);
print(opt)

if opt.identifier == '' then print('NEEDS IDENTIFIER') os.exit() end

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
-- prithv1; changing save_path in checkpoints
questionerModelParams['model_name'] = 'im-hist-enc-dec-questioner'
answererModelParams['model_name'] = 'im-hist-enc-dec-answerer'

opt.img_norm = answererModelParams.img_norm;
opt.model_name_q = questionerModelParams.model_name;
opt.model_name_a = answererModelParams.model_name;
print('Questioner', opt.model_name_q)
print('Answerer', opt.model_name_a)

-- add flags for various configurations
-- additionally check if its imitation of discriminative model
if string.match(opt.model_name_a, 'hist') then opt.useHistory = true; end
if string.match(opt.model_name_a, 'im') then opt.useIm = true; end
------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initializeQuestioner(opt, {'test'});
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

-- qModel.wrapper = qModel.wrapper:cuda()

------------------------------------------------------------------------
-- Talking
------------------------------------------------------------------------
local numThreads = dataloader.numThreads['test'];
local imgPreds = torch.FloatTensor(numThreads * 10, 4096)
local dialogTable = {}

for convId = 1, numThreads do
    xlua.progress(convId, numThreads);
    local inds = torch.LongTensor(1):fill(convId)
    local batch = dataloader:getIndexDataQuestioner(inds, qModel.params, 'test')

    local imgFeats = batch['img_feat']:view(-1, 1, 4096)
    imgFeats = imgFeats:repeatTensor(1, 10, 1);
    imgFeats = imgFeats:view(-1, 4096);
    local batchHist = batch['hist']:view(-1, batch['hist']:size(3)):t()
    local ques = torch.Tensor(20, 10):zero()
    local hist = torch.Tensor(40, 10):zero()
    hist[{{40 - batch['hist']:size(3) + 1, 40}, {1}}] = batchHist[{{},{1}}]

    local thread = {}
    for iter = 1, 10 do
        -- local qOut = qModel:generateSingleQuestionSample(dataloader, hist, {temperature = 1.0}, iter)
        local qOut = qModel:generateSingleQuestion(dataloader, hist, {beamSize = 5}, iter)
        ques[{{},{iter}}] = utils.rightAlign(qOut[1]:view(1, 1, -1), torch.Tensor{qOut[2]}:view(1, 1))
        ques[{{ques:size(1) - qOut[2] + 1}, {iter}}]:fill(0)

        -- local aOut = aModel:generateSingleAnswerSample(dataloader, {hist, imgFeats, ques}, {temperature = 1.0}, iter)
        local aOut = aModel:generateSingleAnswer(dataloader, {hist, imgFeats, ques}, {beamSize = 5}, iter)
        local histVec = torch.Tensor(40):zero()
        for i = 1, qOut[2] + aOut[2] - 3 do
            if i < qOut[2] then
                histVec[i] = qOut[1][i+1]
            else
                histVec[i] = aOut[1][i+2-qOut[2]]
            end
        end
        if iter ~= 10 then
            hist[{{},{iter+1}}] = utils.rightAlign(histVec:view(1,1,-1), torch.Tensor{qOut[2]+aOut[2]-3}:view(1,1))
        end
        table.insert(thread, {question = qOut[3], answer = aOut[3]})
        print(qOut[3], aOut[3])
        if iter == 10 then
            imgPreds[{{(convId - 1) * 10 + 1, convId * 10}, {}}] = qOut[4]:float()
            -- os.exit()
        end

    end
    table.insert(dialogTable, thread)

    if convId % 100 == 0 then
        --save the file to json
        local savePath = string.format('%s/%s-interact-%s-results.json', opt.result_path, questionerModelParams.model_name, opt.identifier);
        utils.writeJSON(savePath, dialogTable);
        print('Writing the results to '..savePath);

        local hdf = hdf5.open(string.format('%s/%s-interact-%s-fc7.h5', opt.result_path, questionerModelParams.model_name, opt.identifier), 'w')
        hdf:write('fc7_test', imgPreds)
        hdf:close()
    end
end


local savePath = string.format('%s/%s-interact-%s-results.json', opt.result_path, questionerModelParams.model_name, opt.identifier);
utils.writeJSON(savePath, dialogTable);
print('Writing the results to '..savePath);

local hdf = hdf5.open(string.format('%s/%s-interact-%s-fc7.h5', opt.result_path, questionerModelParams.model_name, opt.identifier), 'w')
hdf:write('fc7_test', imgPreds)
hdf:close()

