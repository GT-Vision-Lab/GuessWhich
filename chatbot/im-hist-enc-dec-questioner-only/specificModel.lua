-- Implements methods for specific type of model
-- Need following methods, in a table
-- a. buildSpecificModels(self, params)
--      Used to build the particular model
-- b. forwardBackward(self, batch)
--      Used while performing training
-- c. retrieveBatch(self, batch)
--      Used to perform retrieval on a batch
local specificModel = {};
local utils = require 'utils';

function specificModel:buildSpecificModel(params);
    -- build the model - encoder, decoder and answerNet
    local lm;
    local modelFile = string.format('%s/%s', params.model_name,
                                            params.languageModel);
    lm = require(modelFile);
    enc, dec = lm.buildModel(params);
    self.forwardConnect = lm.forwardConnect;
    self.backwardConnect = lm.backwardConnect;

    return enc, dec;
end

function specificModel:forwardBackward(batch, onlyForward);
    local onlyForward = onlyForward or false;
    -- local batchQues = batch['ques_fwd'];
    local batchHist = batch['hist'];
    local questionIn = batch['question_in'];
    local questionOut = batch['question_out'];
    local imgFeats = batch['img_feat'];

    -- resize to treat all rounds similarly
    -- transpose for timestep first
    -- batchQues = batchQues:view(-1, batchQues:size(3)):t();
    batchHist = batchHist:view(-1, batchHist:size(3)):t();
    questionIn = questionIn:view(-1, questionIn:size(3)):t();
    questionOut = questionOut:view(-1, questionOut:size(3)):t();

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    -- print('batchQues', #batchQues)
    -- print('imgFeats', #imgFeats)
    -- print('batchHist', #batchHist)
    -- print(batchQues)
    -- print(questionIn)
    -- print(questionOut)
    -- os.exit()
    local encOut = self.encoder:forward({batchHist}); 
    -- print(#encOut[1])
    -- print(encOut[1]:mean())
    -- print('encOut', encOut)
    -- os.exit()
    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, batchHist:size(1));
    local decOut = self.decoder:forward(questionIn);
    local seqLoss = self.criterion:forward(decOut, questionOut);
    local imgLoss = 0 -- 5 * self.img_criterion:forward(encOut[2], imgFeats)
    -- print('seqLoss', seqLoss);
    -- print('imgLoss', imgLoss)

    -- return only if forward is needed
    if onlyForward == true then goto continue; end

    -- backward pass
    do
    local gradCriterionOut = self.criterion:backward(decOut, questionOut);
    -- local gradImgCriterionOut = 5 * self.img_criterion:backward(encOut[2], imgFeats)
    self.decoder:backward(questionIn, gradCriterionOut);
    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({batchHist}, gradDecOut)
    end

    ::continue::
    collectgarbage()
    return {seqLoss, imgLoss};
end

function specificModel:retrieveBatch(batch);
    local batchQues = batch['ques_fwd'];
    local batchHist = batch['hist'];
    local answerIn = batch['answer_in'];
    local answerOut = batch['answer_out'];
    local optionIn = batch['option_in'];
    local optionOut = batch['option_out'];
    local gtPosition = batch['answer_ind']:view(-1, 1);
    local imgFeats = batch['img_feat'];

    -- resize to treat all rounds similarly
    -- transpose for time step first
    batchQues = batchQues:view(-1, batchQues:size(3)):t();
    batchHist = batchHist:view(-1, batchHist:size(3)):t();
    answerIn = answerIn:view(-1, answerIn:size(3)):t();
    answerOut = answerOut:view(-1, answerOut:size(3)):t();
    optionIn = optionIn:view(-1, optionIn:size(3), optionIn:size(4));
    optionOut = optionOut:view(-1, optionOut:size(3), optionOut:size(4));
    optionIn = optionIn:transpose(1, 2):transpose(2, 3);
    optionOut = optionOut:transpose(1, 2):transpose(2, 3);

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    local encOut = self.encoder:forward({batchQues, batchHist, imgFeats}); 
    local batchSize = batchQues:size(2);
    -- tensor holds the likelihood for all the options
    local optionLhood = torch.Tensor(self.params.numOptions, batchSize);

    -- repeat for each option and get gt rank
    for opId = 1, self.params.numOptions do
        -- forward connect encoder and decoder
        self.forwardConnect(encOut, self.encoder, self.decoder, batchQues:size(1));

        local curOptIn = optionIn[opId];
        local curOptOut = optionOut[opId];

        local decOut = self.decoder:forward(curOptIn);
        -- compute the probabilities for each answer, based on its tokens
        optionLhood[opId] = utils.computeLhood(curOptOut, decOut);
    end

    -- return the ranks for this batch
    return utils.computeRanks(optionLhood:t(), gtPosition);
end

-- code for answer generation
function specificModel:encoderPass(batch)
    -- local batchQues = batch['ques_fwd'];
    -- local batchHist = batch['hist'];
    -- local imgFeats = batch['img_feat'];
    local batchHist = batch['hist'];
    local questionIn = batch['question_in'];
    local questionOut = batch['question_out'];
    local imgFeats = batch['img_feat'];

    -- resize to treat all rounds similarly
    -- transpose for timestep first
    -- batchQues = batchQues:view(-1, batchQues:size(3)):t();
    -- batchHist = batchHist:view(-1, batchHist:size(3)):t();
    batchHist = batchHist:view(-1, batchHist:size(3)):t();
    questionIn = questionIn:view(-1, questionIn:size(3)):t();
    questionOut = questionOut:view(-1, questionOut:size(3)):t();

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- process the image features based on the question (replicate features)
    -- imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    -- imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    -- imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    local encOut = self.encoder:forward({batchHist}); 
    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, batchHist:size(1));
end

return specificModel;
