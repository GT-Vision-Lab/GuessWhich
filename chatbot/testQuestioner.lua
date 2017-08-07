-- Code to test svqa mode
-- author: satwik kottur
-------------------------------------------------------------------------------
-- S-VQA training model using VQA
require 'nn'
require 'nngraph'
require 'io'
require 'rnn'
require 'MaskTime'
utils = dofile('utils.lua');

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the s-vqa model for retrieval')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data/visdial_0.5/data_img.h5','h5file path with image feature')
cmd:option('-input_ques_h5','data/visdial_0.5/chat_processed_data.h5','h5file file with preprocessed questions')
cmd:option('-input_ques_opts_h5','utils/working/qoptions_cap_qatm1_img_sample10k_visdial_0.5_test_vulcan.h5','h5file file with preprocessed question options')
cmd:option('-input_json','data/visdial_0.5/chat_processed_params.json','json path with info and vocab')

cmd:option('-load_path', 'models/model-2-20-2017-11:20:55-im-hist-enc-dec-questioner/model_epoch_20.t7', 'path to saved model')
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
dataloader:initializeQuestioner(opt, {'test'});
collectgarbage();

------------------------------------------------------------------------
-- Setup the model
------------------------------------------------------------------------
require 'modelQuestioner'
print('Using models from '..modelParams.model_name)
local svqaModel = VisDialQModel(modelParams);

-- copy the weights from loaded model
svqaModel.wrapperW:copy(savedModel.modelW);

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
-- validation accuracy
print('Evaluation..')
svqaModel:retrieve(dataloader, 'test');
-- svqaModel:evaluate(dataloader, 'test');
os.exit()

print('Generating questions...')
local res = svqaModel:generateQuestionsBeamSearch(dataloader, 'test', {})

--save the file to json
local savePath = string.format('%s/%s-results.json', opt.result_path, modelParams.model_name);
utils.writeJSON(savePath, res[1]);
print('Writing the results to '..savePath);
-- --]]

-- questions[2] is images
print(res[2]:size())
local hdf = hdf5.open(string.format('%s/%s-fc7.h5', opt.result_path, modelParams.model_name), 'w')
hdf:write('fc7_test', res[2])
hdf:close()

--svqaModel:visualizeAttention(dataloader, 'val', genParams);
