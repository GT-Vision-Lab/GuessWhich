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

function specificModel:buildSpecificModel(params)
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

function specificModel:forwardBackward(batch, onlyForward)
    local onlyForward = onlyForward or false;
    local batchQues = batch['ques_fwd'];
    local batchHist = batch['hist'];
    local answerIn = batch['answer_in'];
    local answerOut = batch['answer_out'];
    local imgFeats = batch['img_feat'];

    -- resize to treat all rounds similarly
    -- transpose for timestep first
    batchQues = batchQues:view(-1, batchQues:size(3)):t();
    batchHist = batchHist:view(-1, batchHist:size(3)):t();
    answerIn = answerIn:view(-1, answerIn:size(3)):t();
    answerOut = answerOut:view(-1, answerOut:size(3)):t();

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    -- print('batchQues', #batchQues)
    -- print('imgFeats', #imgFeats)
    -- print('batchHist', #batchHist)
    local encOut = self.encoder:forward({batchQues, batchHist, imgFeats}); 
    -- print('encOut', #encOut)
    -- os.exit()
    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, batchQues:size(1));
    local decOut = self.decoder:forward(answerIn);
    local curLoss = self.criterion:forward(decOut, answerOut);

    -- return only if forward is needed

    -- backward pass
    if onlyForward ~= true then
        do
        local gradCriterionOut = self.criterion:backward(decOut, answerOut);
        self.decoder:backward(answerIn, gradCriterionOut);
        --backward connect decoder and encoder
        local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
        self.encoder:backward({batchQues, batchHist, imgFeats}, gradDecOut)
        end
    end

    return curLoss;
end

function specificModel:retrieveBatch(batch)
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
    local batchQues = batch['ques_fwd'];
    local batchHist = batch['hist'];
    local imgFeats = batch['img_feat'];

    -- resize to treat all rounds similarly
    -- transpose for timestep first
    batchQues = batchQues:view(-1, batchQues:size(3)):t();
    batchHist = batchHist:view(-1, batchHist:size(3)):t();

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    local encOut = self.encoder:forward({batchQues, batchHist, imgFeats}); 
    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, batchQues:size(1));
    return encOut
end

function specificModel:forwardBackwardReinforce(imgFeats, ques, hist, ansIn, r_t, params)
    assert(imgFeats) -- make sure this is not `nil`, this is GT

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({ques, hist, imgFeats}); 

    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, ques:size(1));
    local decOut = self.decoder:forward(ansIn);

    -- compute loss/reward per round
    local numRounds = 10; -- TODO increase a round
    local gradInput = torch.Tensor(30, numRounds, 7547):zero();

    -- compute RL gradients
    local maxAnsLen = 30;
    for i = 1, numRounds do
        for j = 1, maxAnsLen-1 do
            if ansIn[j][i] ~= 0 and ansIn[j+1][i] == 0 then
                gradInput[j][i][7547] = r_t[i]
            elseif ansIn[j+1][i] ~= 0 then
                gradInput[j][i][ansIn[j+1][i]] = r_t[i]
            end
        end
    end

    -- backprop!
    self.decoder:backward(ansIn, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({ques, hist, imgFeats}, gradDecOut)
    collectgarbage()
end

function specificModel:forwardBackwardAnnealedReinforceBatched(batch, r_t, params)
    local hist = batch[1]
    local imgFeats = batch[2]
    local ques = batch[3]
    local ansIn = batch[4]
    local ansOut = batch[5]

    assert(imgFeats) -- make sure this is not `nil`, this is GT

    local numSLRounds = params.numSLRounds

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({ques, hist, imgFeats});

    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, ques:size(1));
    local decOut = self.decoder:forward(ansIn);

    local maxRounds = 10;
    local numRounds = params.batchSize * maxRounds; -- TODO increase a round
    local gradInput = torch.Tensor(30, numRounds, 7547):zero();

    if numSLRounds ~= 0 then
        local SLRoundInds, SLRoundIndsIdx = torch.LongTensor(numSLRounds * params.batchSize):zero(), 1;
        for i = 1, numRounds do
            if i % maxRounds ~= 0 and i % maxRounds <= numSLRounds then
                SLRoundInds[SLRoundIndsIdx] = i
                SLRoundIndsIdx = SLRoundIndsIdx + 1
            end
        end

        -- compute SL gradients
        local seqLoss = self.criterion:forward(decOut:index(2, SLRoundInds), ansOut:index(2, SLRoundInds))
        gradInput:indexCopy(2, SLRoundInds, self.criterion:backward(decOut:index(2, SLRoundInds), ansOut:index(2, SLRoundInds)))
    end

    -- compute RL gradients
    local maxAnsLen = 30;
    for i = 1, numRounds do
        if i % maxRounds > numSLRounds then
            for j = 1, maxAnsLen-1 do
                if ansIn[j][i] ~= 0 and ansIn[j+1][i] == 0 then
                    gradInput[j][i][7547] = r_t[i]
                elseif ansIn[j+1][i] ~= 0 then
                    gradInput[j][i][ansIn[j+1][i]] = r_t[i]
                end
            end
        end
    end

    -- backprop!
    self.decoder:backward(ansIn, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({ques, hist, imgFeats}, gradDecOut)
    collectgarbage()
    return seqLoss
end

function specificModel:multitaskReinforceForwardBackward(batch, r_t, params)
    local hist = batch[1]
    local imgFeats = batch[2]
    local ques = batch[3]
    local ansIn = batch[4]
    local ansOut = batch[5]
    local ansSample = batch[6]:t()

    assert(imgFeats) -- make sure this is not `nil`, this is GT

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({ques, hist, imgFeats});

    -- forward connect encoder and decoder
    self.forwardConnect(encOut, self.encoder, self.decoder, ques:size(1));
    local decOut = self.decoder:forward(ansIn);

    local maxRounds = 10;
    local numRounds = params.batchSize * maxRounds; -- TODO increase a round
    local gradInput = torch.Tensor(30, numRounds, 7547):zero();

    -- compute SL gradients
    local seqLoss = self.criterion:forward(decOut, ansOut)
    local gradInputSL = self.criterion:backward(decOut, ansOut)

    -- compute RL gradients
    local gradInputRL = torch.Tensor(30, numRounds, 7547):zero();
    local maxAnsLen = 30;
    for i = 1, numRounds do
        for j = 2, maxAnsLen-1 do
            if ansSample[j][i] ~= 0 then
                gradInputRL[j-1][i][ansSample[j][i]] = r_t[i]
            end
        end
    end

    -- print(params)
    gradInput = gradInputSL + params.lambda * gradInputRL;

    -- backprop!
    self.decoder:backward(ansIn, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({ques, hist, imgFeats}, gradDecOut)
    collectgarbage()
    return seqLoss
end
return specificModel;
