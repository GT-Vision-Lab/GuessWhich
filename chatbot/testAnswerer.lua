require 'nn'
require 'nngraph'
require 'io'
require 'rnn'
utils = dofile('utils.lua');

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data/visdial_0.5/data_img.h5','h5file path with image feature')
cmd:option('-input_ques_h5','data/visdial_0.5/chat_processed_data.h5','h5file file with preprocessed questions')
cmd:option('-input_json','data/visdial_0.5/chat_processed_params.json','json path with info and vocab')

cmd:option('-load_path', 'models/model-2-14-2017-22:43:51-im-hist-enc-dec/model_epoch_20.t7', 'path to saved model')
cmd:option('-result_path', 'results', 'path to save generated results')

-- optimization params
cmd:option('-batchSize', 200, 'Batch size (number of threads) (Adjust base on GRAM)');
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

local opt = cmd:parse(arg);
print(opt)

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
local savedModel = torch.load(opt.load_path);

-- transfer all options to model
local modelParams = savedModel.modelParams;
opt.img_norm = modelParams.img_norm;
opt.model_name = modelParams.model_name;
print(opt.model_name)

-- add flags for various configurations
-- additionally check if its imitation of discriminative model
if string.match(opt.model_name, 'hist') then
    opt.useHistory = true;
    if string.match(opt.model_name, 'disc') then
        opt.separateCaption = true;
    end
end
if string.match(opt.model_name, 'im') then opt.useIm = true; end
------------------------------------------------------------------------
-- Loading dataset
------------------------------------------------------------------------
local dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, {'test'});
collectgarbage();

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'modelAnswerer'
print('Using models from '..modelParams.model_name)
local svqaModel = VisDialAModel(modelParams);

-- copy the weights from loaded model
svqaModel.wrapperW:copy(savedModel.modelW);

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
-- validation accuracy
print('Evaluation..')
svqaModel:retrieve(dataloader, 'test');
os.exit()

---[[
print('Generating answers...')
-- local answers = svqaModel:generateAnswers(dataloader, 'test', {sample = false});
local answers = svqaModel:generateAnswersBeamSearch(dataloader, 'test', {});

--save the file to json
local savePath = string.format('%s/%s-results.json', opt.result_path, modelParams.model_name);
utils.writeJSON(savePath, answers);
print('Writing the results to '..savePath);
-- --]]

--svqaModel:visualizeAttention(dataloader, 'val', genParams);
