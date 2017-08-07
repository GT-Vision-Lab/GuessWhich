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
    params.languageModel = 'lstm';
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

    local encOut = self.encoder:forward({batchHist}); 

    -- forward connect encoder and decoder
    self.forwardConnect(encOut[1], self.encoder, self.decoder, batchHist:size(1));
    local decOut = self.decoder:forward(questionIn);
    local seqLoss = self.criterion:forward(decOut, questionOut);
    local imgLoss = 1000 * self.img_criterion:forward(encOut[2], imgFeats)
    -- print('seqLoss', seqLoss);
    -- print('imgLoss', imgLoss)

    -- return only if forward is needed

    -- backward pass
    if onlyForward ~= true then
        do
            local gradCriterionOut = self.criterion:backward(decOut, questionOut);
            local gradImgCriterionOut = 1000 * self.img_criterion:backward(encOut[2], imgFeats)
            self.decoder:backward(questionIn, gradCriterionOut);
            --backward connect decoder and encoder
            local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
            self.encoder:backward({batchHist}, {gradDecOut, gradImgCriterionOut})
        end
    end

    collectgarbage()
    return {seqLoss, imgLoss};
end

function specificModel:retrieveBatch(batch)
    local batchHist = batch['hist'];
    local questionIn = batch['question_in'];
    local questionOut = batch['question_out'];
    local imgFeats = batch['img_feat'];
    local optionIn = batch['option_in'];
    local optionOut = batch['option_out'];
    local gtPosition = batch['question_ind']:view(-1, 1);
    local imgFeats = batch['img_feat'];

    batchHist = batchHist:view(-1, batchHist:size(3)):t();
    questionIn = questionIn:view(-1, questionIn:size(3)):t();
    questionOut = questionOut:view(-1, questionOut:size(3)):t();

    optionIn = optionIn:view(-1, optionIn:size(3), optionIn:size(4));
    optionOut = optionOut:view(-1, optionOut:size(3), optionOut:size(4));
    optionIn = optionIn:transpose(1, 2):transpose(2, 3);
    optionOut = optionOut:transpose(1, 2):transpose(2, 3);

    -- process the image features based on the question (replicate features)
    imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);

    -- forward pass
    local encOut = self.encoder:forward({batchHist}); 
    local batchSize = batchHist:size(2);
    -- tensor holds the likelihood for all the options
    local optionLhood = torch.Tensor(self.params.numOptions, batchSize);

    -- repeat for each option and get gt rank
    for opId = 1, self.params.numOptions do
        -- forward connect encoder and decoder
        self.forwardConnect(encOut[1], self.encoder, self.decoder, batchHist:size(1));

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

    -- -- resize to treat all rounds similarly
    -- -- transpose for timestep first
    -- batchQues = batchQues:view(-1, batchQues:size(3)):t();
    -- batchHist = batchHist:view(-1, batchHist:size(3)):t();

    -- -- process the image features based on the question (replicate features)
    -- imgFeats = imgFeats:view(-1, 1, self.params.imgFeatureSize);
    -- imgFeats = imgFeats:repeatTensor(1, self.params.maxQuesCount, 1);
    -- imgFeats = imgFeats:view(-1, self.params.imgFeatureSize);
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
    local encOut = self.encoder:forward({batchHist}); 
    -- forward connect encoder and decoder
    self.forwardConnect(encOut[1], self.encoder, self.decoder, batchHist:size(1));
    return encOut
end

function specificModel:forwardBackwardReinforce(imgFeats, ques, hist, params)
    assert(imgFeats) -- make sure this is not `nil`, this is GT

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({hist})

    self.forwardConnect(encOut[1], self.encoder, self.decoder, hist:size(1));
    local decOut = self.decoder:forward(ques);

    -- compute loss/reward per round
    local numRounds = 10; -- TODO increase a round
    local imgLoss, r_t = torch.Tensor(numRounds), torch.Tensor(numRounds):zero()
    local gradInput = torch.Tensor(30, numRounds, 7547):zero();

    for i = 1, numRounds do
        imgLoss[i] = self.img_criterion:forward(encOut[2][i]:squeeze(), imgFeats[i]:squeeze())
        if i > 1 then
            r_t[i-1] = imgLoss[i] - imgLoss[i-1]
            -- r_t[i-1] = r_t[i-1] * 1e-3 -- self.optims.learningRate;
        end
    end

    -- clamping rewards
    -- torch.clamp(r_t, -0.0001, 0.0001)

    local maxQuesLen = 30;
    for i = 1, numRounds do
        for j = 1, maxQuesLen-1 do
            if ques[j][i] == 4640 and ques[j+1][i] == 0 then
                gradInput[j][i][7547] = r_t[i];
            elseif ques[j+1][i] ~= 0 then
                gradInput[j][i][ques[j+1][i]] = r_t[i]
            end
        end
    end

    local gradImgInput;
    if params ~= nil and params.freezeImg == true then
        imgLoss = self.img_criterion:forward(encOut[2], encOut[2]) -- frozen image head
        gradImgInput = self.img_criterion:backward(encOut[2], encOut[2])
    else
        imgLoss = self.img_criterion:forward(encOut[2], imgFeats)
        gradImgInput = 1000 * self.img_criterion:backward(encOut[2], imgFeats)
    end

    -- backprop!
    self.decoder:backward(ques, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({hist}, {gradDecOut, gradImgInput})

    imgLoss = self.img_criterion:forward(encOut[2], imgFeats) -- return this anyway
    collectgarbage()
    return {1000.0 * imgLoss, r_t}
end

function specificModel:forwardBackwardAnnealedReinforce(batch, params)
    local hist = batch[1];
    local imgFeats = batch[2];
    local quesIn = batch[3];
    local quesOut = batch[4];

    assert(imgFeats) -- make sure this is not `nil`, this is GT

    local numSLRounds = params.numSLRounds;

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({hist})

    self.forwardConnect(encOut[1], self.encoder, self.decoder, hist:size(1));
    local decOut = self.decoder:forward(quesIn);

    -- compute loss/reward per round
    local numRounds = 10; -- TODO increase a round
    local imgLoss, r_t = torch.Tensor(numRounds), torch.Tensor(numRounds):zero()
    local gradInput = torch.Tensor(30, numRounds, 7547):zero()

    for i = 1, numRounds do
        imgLoss[i] = self.img_criterion:forward(encOut[2][i]:squeeze(), imgFeats[i]:squeeze())
        if i > 1 then
            r_t[i-1] = imgLoss[i] - imgLoss[i-1]
            -- r_t[i-1] = r_t[i-1] * 1e-3 -- self.optims.learningRate;
        end
    end

    -- clamping rewards
    -- torch.clamp(r_t, -0.001, 0.00001)

    -- compute SL gradients
    local seqLoss = self.criterion:forward(decOut[{{}, {1,numSLRounds}, {}}], quesOut[{{}, {1,numSLRounds}}])
    gradInput[{{}, {1, numSLRounds}, {1, 7547}}] = self.criterion:backward(decOut[{{}, {1, numSLRounds}, {}}], quesOut[{{}, {1,numSLRounds}}])

    -- compute RL gradients
    local maxQuesLen = 30;
    for i = numSLRounds+1, numRounds do
        for j = 1, maxQuesLen-1 do
            if quesIn[j][i] == 4640 and quesIn[j+1][i] == 0 then
                gradInput[j][i][7547] = r_t[i];
            elseif quesIn[j+1][i] ~= 0 then
                gradInput[j][i][quesIn[j+1][i]] = r_t[i]
            end
        end
    end

    local gradImgInput;
    if params ~= nil and params.freezeImg == true then
        imgLoss = self.img_criterion:forward(encOut[2], encOut[2]) -- frozen image head
        gradImgInput = self.img_criterion:backward(encOut[2], encOut[2])
    else
        imgLoss = self.img_criterion:forward(encOut[2], imgFeats)
        gradImgInput = 1000 * self.img_criterion:backward(encOut[2], imgFeats)
    end

    -- backprop!
    self.decoder:backward(quesIn, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({hist}, {gradDecOut, gradImgInput})

    imgLoss = self.img_criterion:forward(encOut[2], imgFeats) -- return this anyway
    collectgarbage()
    return {seqLoss, 1000.0 * imgLoss, r_t}
end

function specificModel:forwardBackwardAnnealedReinforceBatched(batch, params)
    local hist = batch[1];
    local imgFeats = batch[2];
    local quesIn = batch[3];
    local quesOut = batch[4];

    assert(imgFeats) -- make sure this is not `nil`, this is GT

    local numSLRounds = params.numSLRounds;

    -- encoder forward pass, just sanity checking
    local encOut = self.encoder:forward({hist})

    self.forwardConnect(encOut[1], self.encoder, self.decoder, hist:size(1));
    local decOut = self.decoder:forward(quesIn);

    -- compute loss/reward per round
    local maxRounds = 10;
    local numRounds = params.batchSize * maxRounds; -- TODO increase a round
    local imgLoss, r_t = torch.Tensor(numRounds):zero(), torch.Tensor(numRounds):zero()
    local gradInput = torch.Tensor(30, numRounds, 7547):zero()

    for i = 1, numRounds do
        imgLoss[i] = self.img_criterion:forward(encOut[2][i]:squeeze(), imgFeats[i]:squeeze())
        if i % maxRounds ~= 1 then -- ignore last round
            r_t[i-1] = imgLoss[i] - imgLoss[i-1]
            -- r_t[i-1] = r_t[i-1] * 1e-3 -- self.optims.learningRate;
        end
    end

    -- clamping rewards
    -- torch.clamp(r_t, -0.001, 0.00001)

    local SLRoundInds, SLRoundIndsIdx = torch.LongTensor(numSLRounds * params.batchSize):zero(), 1;
    if numSLRounds ~= 0 then
        for i = 1, numRounds do
            if i % maxRounds ~= 0 and i % maxRounds <= numSLRounds then
                SLRoundInds[SLRoundIndsIdx] = i
                SLRoundIndsIdx = SLRoundIndsIdx + 1
            end
        end
    end

    -- compute SL gradients
    local seqLoss = 0;
    if numSLRounds ~= 0 then
        seqLoss = self.criterion:forward(decOut:index(2, SLRoundInds), quesOut:index(2, SLRoundInds))
        gradInput:indexCopy(2, SLRoundInds, self.criterion:backward(decOut:index(2, SLRoundInds), quesOut:index(2, SLRoundInds)))
    end

    -- compute RL gradients
    local maxQuesLen = 30;
    for i = 1, numRounds do
        if i % maxRounds > numSLRounds then
            for j = 1, maxQuesLen-1 do
                if quesIn[j][i] == 4640 and quesIn[j+1][i] == 0 then
                    gradInput[j][i][7547] = r_t[i];
                elseif quesIn[j+1][i] ~= 0 then
                    gradInput[j][i][quesIn[j+1][i]] = r_t[i]
                end
            end
        end
    end

    local gradImgInput;
    if params ~= nil and params.freezeImg == true then
        imgLoss = self.img_criterion:forward(encOut[2], encOut[2]) -- frozen image head
        gradImgInput = self.img_criterion:backward(encOut[2], encOut[2])
    else
        imgLoss = self.img_criterion:forward(encOut[2], imgFeats)
        gradImgInput = 1000 * self.img_criterion:backward(encOut[2], imgFeats)
    end

    -- backprop!
    self.decoder:backward(quesIn, gradInput);

    --backward connect decoder and encoder
    local gradDecOut = self.backwardConnect(self.encoder, self.decoder);
    self.encoder:backward({hist}, {gradDecOut, gradImgInput})

    imgLoss = self.img_criterion:forward(encOut[2], imgFeats) -- return this anyway
    collectgarbage()
    return {seqLoss, 1000.0 * imgLoss, r_t}
end
return specificModel;
