--[[
    History will not have <START>, only <END>
]]
require 'nn'
require 'nngraph'
require 'cjson'
require 'rnn'
require 'MaskTime'
require 'modelQuestioner'
require 'modelAnswerer'
utils = dofile('utils.lua')

local TorchModel = torch.class('SLConversationModel')

function TorchModel:__init(inputJson, qBotpath, aBotpath, gpuid, backend, imfeatpath)
    -- Load the image features
    print(imfeatpath)
    self.imfeats = torch.load(imfeatpath)
    print(self.imfeats)
    print(#self.imfeats)

    -- Model paths
    self.qBotpath = qBotpath
    self.aBotpath = aBotpath
    self.gpuid = gpuid
    self.backend = backend

    -- Create options table to initialize dataloader
    self.opt = {}
    self.opt['input_json'] = inputJson
    self.dataloader = dofile('dataloader.lua')
    self.dataloader:initialize(self.opt)

    -- Initial seeds
    torch.manualSeed(1234)
    if self.gpuid >= 0 then
        require 'cutorch'
        require 'cunn'
        if self.backend == 'cudnn' then require 'cudnn' end
        cutorch.setDevice(1)
        cutorch.manualSeed(1234)
        torch.setdefaulttensortype('torch.CudaTensor')
    else
        torch.setdefaulttensortype('torch.DoubleTensor')
    end

    -- Load Questioner and Answerer model
    self.questionerModel = torch.load(qBotpath)
    self.answererModel = torch.load(aBotpath)

    -- transfer all options to model
    self.questionerModelParams = self.questionerModel.modelParams
    self.answererModelParams = self.answererModel.modelParams

    -- changing savepath in checkpoints
    self.questionerModelParams['model_name'] = 'im-hist-enc-dec-questioner'
    self.answererModelParams['model_name'] = 'im-hist-enc-dec-answerer'

    -- Print Questioner and Answerer
    print('Questioner', self.questionerModelParams.model_name)
    print('Answerer', self.answererModelParams.model_name)

    -- Add flags for various configurations
    if string.match(self.questionerModelParams.model_name, 'hist') then self.questionerModelParams.useHistory = true; end
    if string.match(self.answererModelParams.model_name, 'hist') then self.answererModelParams.useHistory = true; end
    if string.match(self.answererModelParams.model_name, 'im') then self.answererModelParams.useIm = true; end

    -- Setup both Qbot and Abot
    print('Using models from'.. self.questionerModelParams.model_name)
    print('Using models from'.. self.answererModelParams.model_name)
    self.qModel = VisDialQModel(self.questionerModelParams)
    self.aModel = VisDialAModel(self.answererModelParams)

    -- copy weights from loaded model
    self.qModel.wrapperW:copy(self.questionerModel.modelW)
    self.aModel.wrapperW:copy(self.answererModel.modelW)

    -- set models to evaluate mode
    self.qModel.wrapper:evaluate()
    self.aModel.wrapper:evaluate()
end

--[[
    ABot method implementation is exactly similar to the Visual Dialog Model
    Need to clarify the left/right alignment of questions/etc
]]

function TorchModel:abot(imgId, history, question)
    -- Get image-feature
    local imgFeat = self.imfeats[imgId]
    imgFeat = torch.repeatTensor(imgFeat, 10, 1)
    -- Concatenate history
    local history_concat = ''
    for i=1, #history do
        history_concat = history_concat .. history[i] .. ' |||| '
    end
    -- -- Remove <START> from history
    -- history_concat = history_concat:gsub('<START>','')
    -- if history_concat ~= '' then
    --  history_concat = history_concat .. ' <END>'
    -- end
    -- get pre-processed QA+Hist
    local cmd = 'python prepro_ques.py -question "' .. question .. '" -history "' .. history_concat .. '"'
    os.execute(cmd)
    local file = io.open('ques_feat.json', 'r')
    if file then
        json_f = file:read('*a')
        qh_feats = cjson.decode(json_f)
        file:close()
    end
    -- Get question vector
    local ques_vector = utils.wordsToId(qh_feats.question, self.dataloader.word2ind, 20)
    -- Get history Tensor and hist_len vector
    local hist_tensor = torch.LongTensor(10, 40):zero()
    local hist_len = torch.zeros(10)
    for i=1, #qh_feats.history do
        hist_tensor[i] = utils.wordsToId(qh_feats.history[i], self.dataloader.word2ind, 40)
        hist_len[i] = hist_tensor[i][hist_tensor[i]:ne(0)]:size(1)
    end
    -- Get question Tensor
    local ques_tensor = torch.LongTensor(10, 20):zero()
    local ques_len = torch.zeros(10)
    for i=1, #qh_feats.questions do
        ques_tensor[i] = utils.wordsToId(qh_feats.questions[i], self.dataloader.word2ind, 20)
        ques_len[i] = ques_tensor[i][ques_tensor[i]:ne(0)]:size(1)
    end
    -- Parameter for generating answers
    local iter = #qh_feats.questions + 1
    ques_tensor[iter] = ques_vector
    -- Right align the questions
    -- ques_tensor = utils.rightAlign(ques_tensor, ques_len)
    -- Right align the history
    -- hist_tensor = utils.rightAlign(hist_tensor, hist_len)
    -- Transpose the question and history
    ques_tensor = ques_tensor:t()
    hist_tensor = hist_tensor:t()
    -- Shift to GPU
    if self.gpuid >= 0 then
        ques_tensor = ques_tensor:cuda()
        hist_tensor = hist_tensor:cuda()
        imgFeat = imgFeat:cuda()
    end
    -- Generate answer; returns a table :-> {ansWords, aLen, ansText}
    local ans_struct = self.aModel:generateSingleAnswer(self.dataloader, {hist_tensor, imgFeat, ques_tensor}, {beamSize = 5}, iter)
    -- Use answer-text to concatenate things to show at subject's end
    local answer = ans_struct[3]
    local result = {}
    result['answer'] = answer
    result['question'] = question
    if history_concat == '||||' then
        history_concat = ''
    end
    result['history'] = history_concat .. question .. ' ' .. answer
    result['history'] = string.gsub(result['history'], '<START>', '')
    result['input_img'] = imgId
    return result
end