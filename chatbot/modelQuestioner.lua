-- abstract class for models
require 'optim_updates'
require 'xlua'
require 'hdf5'
require 'ReplaceZero'

local utils = require 'utils'

local VisDialQModel = torch.class('VisDialQModel');

-- initialize
function VisDialQModel:__init(params)
    print('Setting up SVQA model..\n');
    self.params = params;

    -- build the model - encoder, decoder and answerNet
    local modelFile = string.format('%s/specificModel.lua', params.model_name);
    local model = dofile(modelFile);
    enc, dec = model:buildSpecificModel(params);
    -- add methods from specific model
    local methods = {'forwardConnect', 'backwardConnect',
                    'forwardBackward', 'retrieveBatch', 'encoderPass',
                    'forwardBackwardReinforce', 'forwardBackwardAnnealedReinforce',
                    'forwardBackwardAnnealedReinforceBatched'};
    for key, value in pairs(methods) do self[value] = model[value]; end

    -- print models
    print('Encoder:\n'); print(enc);
    -- print('Decoder:\n'); print(dec);

    -- criterion
    self.criterion = nn.ClassNLLCriterion();
    self.criterion.sizeAverage = false;
    self.criterion = nn.SequencerCriterion(
                                nn.MaskZeroCriterion(self.criterion, 1));

    -- fc7 regression criterion
    self.img_criterion = nn.MSECriterion();

    -- wrap the models
    self.wrapper = nn.Sequential():add(enc):add(dec);

    -- initialize weights
    self.wrapper = require('weight-init')(self.wrapper, 'xavier');
    -- ship to gpu if necessary
    if params.gpuid >= 0 then
        print('Shifting to cuda..')
        self.wrapper = self.wrapper:cuda();
        self.criterion = self.criterion:cuda();
    end

    self.encoder = self.wrapper:get(1);
    self.decoder = self.wrapper:get(2);
    self.wrapperW, self.wrapperdW = self.wrapper:getParameters();

    self.wrapper:training();

    -- setup the optimizer
    self.optims = {};
    self.optims.learningRate = params.learningRate;
end

-------------------------------------------------------------------------------
-- One iteration of training -- forward and backward pass
function VisDialQModel:trainIteration(dataloader)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    -- grab a training batch
    local batch = dataloader:getTrainBatchQuestioner(self.params);

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackward(batch);
    -- print(curLoss)
    -- count the number of tokens
    local numTokens = torch.sum(batch['question_out']:gt(0));

    local seqLoss = curLoss[1] / numTokens
    local imgLoss = curLoss[2]

    -- update the running average of loss
    if runningLoss > 0 then
        runningLoss = 0.95 * runningLoss + 0.05 * seqLoss;
        runningLossImg = 0.95 * runningLossImg + 0.05 * imgLoss;
    else
        runningLoss = seqLoss
        runningLossImg = imgLoss
    end
    -- clamp gradients
    self.wrapperdW:clamp(-5.0, 5.0);

    -- update parameters
    --rmsprop(self.wrapperW, self.wrapperdW, self.optims);
    adam(self.wrapperW, self.wrapperdW, self.optims);

    -- decay learning rate, if needed
    if self.optims.learningRate > self.params.minLRate then
        self.optims.learningRate = self.optims.learningRate *
                                        self.params.lrDecayRate;
    end
end

-- One iteration of reinforce -- forward and backward pass
function VisDialQModel:reinforce(imgFeats, ques, hist, iter)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    params = {freezeImg = false, curriculum = false, iter = iter}

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackwardReinforce(imgFeats, ques, hist, params);
    local imgLoss = curLoss[1]
    local r_t = curLoss[2]

    -- update the running average of loss
    if runningLossImgRL > 0 then
        runningLossImgRL = 0.95 * runningLossImgRL + 0.05 * imgLoss;
    else
        runningLossImgRL = imgLoss
    end

    -- clamp gradients
    self.wrapperdW:clamp(-5.0, 5.0);

    -- update parameters
    adam(self.wrapperW, self.wrapperdW, self.optims);

    -- decay learning rate, if needed
    if self.optims.learningRate > self.params.minLRate then
        self.optims.learningRate = self.optims.learningRate *
                                        self.params.lrDecayRate;
    end

    collectgarbage()
    -- return r_t, we can pass this to answerer
    return r_t
end

-- SL for t rounds, RL for n - t rounds
function VisDialQModel:annealedReinforce(batch, numSLRounds, iter)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    params = {freezeImg = false, curriculum = false, iter = iter, numSLRounds = numSLRounds}

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackwardAnnealedReinforce(batch, params);

    local numTokens = torch.sum(batch[4]:gt(0));
    local seqLoss = curLoss[1] / numTokens;
    local imgLoss = curLoss[2]
    local r_t = curLoss[3]

    -- update the running average of loss
    if runningLossImgRL > 0 then
        runningLoss = 0.95 * runningLoss + 0.05 * seqLoss;
        runningLossImgRL = 0.95 * runningLossImgRL + 0.05 * imgLoss;
        meanReward = 0.95 * meanReward + 0.05 * r_t[{{numSLRounds+1, 10}}]:mean()
    else
        runningLoss = seqLoss
        runningLossImgRL = imgLoss
        meanReward = r_t[{{numSLRounds+1, 10}}]:mean();
    end

    -- clamp gradients
    self.wrapperdW:clamp(-5.0, 5.0);

    -- update parameters
    adam(self.wrapperW, self.wrapperdW, self.optims);

    -- decay learning rate, if needed
    if self.optims.learningRate > self.params.minLRate then
        self.optims.learningRate = self.optims.learningRate *
                                        self.params.lrDecayRate;
    end

    collectgarbage()
    -- return r_t, we can pass this to answerer
    return r_t
end
---------------------------------------------------------------------
-- SL for t rounds, RL for n - t rounds
function VisDialQModel:annealedReinforceBatched(batch, numSLRounds, opt)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    params = {freezeImg = false, curriculum = false, iter = opt.iter, batchSize = opt.batchSize, numSLRounds = numSLRounds}

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackwardAnnealedReinforceBatched(batch, params);

    local numTokens = torch.sum(batch[4]:gt(0));
    local seqLoss = 0
    if curLoss[1] ~= nil then
        seqLoss = curLoss[1] / numTokens;
    end
    local imgLoss = curLoss[2]
    local r_t = curLoss[3]

    -- update the running average of loss
    if runningLossImgRL > 0 then
        runningLoss = 0.95 * runningLoss + 0.05 * seqLoss;
        runningLossImgRL = 0.95 * runningLossImgRL + 0.05 * imgLoss;
        meanReward = 0.95 * meanReward + 0.05 * torch.sum(r_t) / (opt.batchSize * (10 - numSLRounds - 1))
    else
        runningLoss = seqLoss
        runningLossImgRL = imgLoss
        meanReward = torch.sum(r_t) / (opt.batchSize * (10 - numSLRounds - 1))
    end

    -- clamp gradients
    self.wrapperdW:clamp(-5.0, 5.0);

    -- update parameters
    if opt.freeze == 0 then
        adam(self.wrapperW, self.wrapperdW, self.optims);

        -- decay learning rate, if needed
        if self.optims.learningRate > self.params.minLRate then
            self.optims.learningRate = self.optims.learningRate *
                                            self.params.lrDecayRate;
        end
    end

    collectgarbage()
    -- return r_t, we can pass this to answerer
    return r_t
end
-- SL for t rounds, RL for n - t rounds
function VisDialQModel:multitaskReinforce(batch, opt)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    local imgFeats = batch[1]:cuda()
    local imgPreds = batch[2]:cuda()

    local maxRounds = 10
    local numRounds = opt.batchSize * maxRounds
    local imgLoss, r_t = torch.Tensor(numRounds):zero(), torch.Tensor(numRounds):zero()

    for i = 1, numRounds do
        imgLoss[i] = self.img_criterion:forward(imgPreds[i]:squeeze(), imgFeats[i]:squeeze())
        if i % maxRounds ~= 1 then
            r_t[i-1] = (imgLoss[i] - imgLoss[i-1])
        end
    end

    imgLoss = 1000 * self.img_criterion:forward(imgPreds, imgFeats)

    -- update the running average of loss
    if runningLossImg > 0 then
        runningLossImg = 0.95 * runningLossImg + 0.05 * imgLoss;
        meanReward = 0.95 * meanReward + 0.05 * torch.sum(r_t) / (opt.batchSize * 10)
    else
        runningLossImg = imgLoss
        meanReward = torch.sum(r_t) / (opt.batchSize * 10)
    end

    collectgarbage()

    -- return r_t, we can pass this to answerer
    return r_t
end
-- validation performance on test/val
function VisDialQModel:evaluate(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local imgLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];
    local numTokens = 0;
    local batchSize = 0;
    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a validation batch
        local batch, nextStartId
                        = dataloader:getTestBatchQuestioner(startId, self.params, dtype);
        -- count the number of tokens
        numTokens = numTokens + torch.sum(batch['question_out']:gt(0));
        -- forward pass to compute loss
        local loss = self:forwardBackward(batch, true);
        curLoss = curLoss + loss[1];
        imgLoss = imgLoss + loss[2];
        -- curLoss = curLoss + self:forwardBackward(batch, true);
        startId = nextStartId;
        if batchSize == 0 then batchSize = batch['question_out']:size(1) end
    end

    -- print the results
    curLoss = curLoss / numTokens;
    imgLoss = imgLoss / numThreads * batchSize;
    print(string.format('\n%s\tLoss: %f\tPerplexity: %f\tImg Loss: %f\n', dtype,
                        curLoss, math.exp(curLoss), imgLoss));

    -- change back to training
    self.wrapper:training();
    collectgarbage();
end
---------------------------------------------------------------------
-- retrieval performance on test/val
function VisDialQModel:retrieve(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];
    -- numThreads = 3000
    print('numThreads', numThreads)

    -- ranks for the given datatype
    local ranks = torch.Tensor(numThreads, self.params.maxQuesCount);
    ranks:fill(self.params.numOptions + 1);
    print('ranks', #ranks)

    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a validation batch
        local batch, nextStartId =
                        dataloader:getTestBatchQuestioner(startId, self.params, dtype);

        -- Call retrieve function for specific model, and store ranks
        ranks[{{startId, nextStartId - 1}, {}}] = self:retrieveBatch(batch);
        startId = nextStartId;
    end

    -- if lambdaRange is given
    torch.save('ranks_questioner.t7', ranks:float())
    print('saved to ranks_questioner.t7')

    print(string.format('\n%s - Retrieval:', dtype))
    utils.processRanks(ranks);

    -- os.exit()

    -- change back to training
    self.wrapper:training();

    -- collect garbage
    collectgarbage();
end

-- generating questions
function VisDialQModel:generateQuestions(dataloader, dtype, genParams)
    local sampleOutput = genParams.sample or false;
    local temperature = genParams.temperature or 1.0;
    local maxTimeSteps = genParams.maxTimeSteps or self.params.maxAnsLen;
    local numConvs = genParams.numConvs or 2;

    -- change the model to evaluate model
    self.wrapper:evaluate();

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['?'];
    local numThreads = dataloader.numThreads[dtype];
    -- ABHSHKDZ
    -- numThreads = 500

    local answerTable = {};
    local answerWts = {};

    -- Tensor to save predicted fc7s
    local img_preds = torch.FloatTensor(numThreads * 10, 4096)

    -- generate for the first few questions
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        -- print the conversation id
        --print(string.format('Conversation: %d', convId))
        local inds = torch.LongTensor(1):fill(convId);
        local batch = dataloader:getIndexDataQuestioner(inds, self.params, dtype);
        local numQues = batch['question_in']:size(1) * batch['question_in']:size(2);

        -- one pass through the encoder
        -- check for additional attributes to save
        local res = self:encoderPass(batch);
        local encOut = res[1]
        local img_fc7 = res[2]
        img_preds[{{(convId - 1) * 10 + 1, convId * 10}, {}}] = img_fc7:float()

        -- start generation for each time step
        local answerIn = torch.Tensor(1, numQues):fill(startToken);
        local answer = {answerIn:t():double()};
        -- print(self.encoder.modules)
        -- for i = 1, #self.encoder.modules do
            -- print(i, self.encoder.modules[i])
        -- end
        -- memory-im-hist, 22 is maskSoftmax
        -- local att_weights = self.encoder.modules[22].output:doublequestion_in
        -- memory-hist, 18 is maskSoftmax
        -- local att_weights = self.encoder.modules[18].output:double()

        -- self.decoder:remember()

        for timeStep = 1, maxTimeSteps do
            -- one pass through the decoder
            local decOut = self.decoder:forward(answerIn):squeeze();
            -- connect decoder to itself
            self:decoderConnect(self.decoder);

            -- get the max or sample
            if sampleOutput == true then
                -- sampling with temperature
                nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);
            else
                -- take the max value
                _, nextToken = torch.topk(decOut, 1, true);
            end
            --nextToken = nextToken[1];

            -- add to the answer, exit if endToken
            table.insert(answer, nextToken:double());
            -- cannot break until all batches have end tokens
            -- just keep generating until end and curate later
            --if nextToken == endToken then break end
            answerIn:copy(nextToken);
        end

        -- self.decoder:forget()

        -- join all the outputs
        answer = nn.JoinTable(-1):forward(answer);

        -- convert each question and answer to text
        local threadAnswers = {};
        for quesId = 1, self.params.maxQuesCount do
            local quesWords = batch['hist'][{{1}, {quesId}, {}}]:squeeze():double();
            local ansWords = answer[{{quesId}, {}}]:squeeze();

            local quesText = utils.idToWords(quesWords, dataloader.ind2word);
            local ansText = utils.idToWords(ansWords, dataloader.ind2word);

            -- local questionAttWeight = nn.SplitTable(1):forward(att_weights[quesId]);

            -- table.insert(threadAnswers, {question=quesText, answer=ansText, attention=questionAttWeight});
            table.insert(threadAnswers, {history=quesText, question=ansText})
            -- print(string.format('H-%d: %s\nQ-%d: %s', quesId, quesText, quesId,
                                                                    -- ansText))
        end
        table.insert(answerTable, threadAnswers);
        -- check if there is additional save
        -- if modelSave then table.insert(answerWts, modelSave); end
    end
    -- save if non-empty
    -- if #answerWts > 0 then
        -- -- join all the answer weights
        -- local attWts = nn.JoinTable(1):forward(answerWts);
        -- local filePt = hdf5.open('results/im-hist-enc-dec-att-wts.h5', 'w');
        -- filePt:write('weights', attWts:double());
        -- filePt:close();
    -- end

    -- going back to training mode
    self.wrapper:training();

    -- return answerTable;
    return {answerTable, img_preds};
end
-- connecting decoder to itself
function VisDialQModel:decoderConnect(dec, seqLen)
    for ii = 1, #dec.rnnLayers do
        dec.rnnLayers[ii].userPrevOutput = dec.rnnLayers[ii].output[1]
        dec.rnnLayers[ii].userPrevCell = dec.rnnLayers[ii].cell[1]
    end

    -- if language model is lstm
    -- if self.params.languageModel == 'lstm' then
        -- for ii = 1, #dec.rnnLayers do
            -- dec.rnnLayers[ii].userPrevCell = dec.rnnLayers[ii].cell[1]
        -- end
    -- end
end

function VisDialQModel:encodeSentence(txt, word2ind)
    txt = string.lower(txt)
    local len = 0
    local words = torch.Tensor(self.params.maxAnsLen):zero()
    for i in string.gmatch(txt, "%S+") do
        len = len + 1
        if word2ind[i] ~= nil then
            words[len] = word2ind[i]
        else
            words[len] = word2ind['UNK']
        end
        if len == words:size(1) then break end
    end
    return utils.rightAlign(words:view(1, 1, -1), torch.Tensor{len}:view(1, 1))
end

-- sampling
function VisDialQModel:generateSingleQuestionSample(dataloader, hist, params, iter)
    local sampleOutput = params.sample or false;
    local temperature = params.temperature or 1.0;
    local maxTimeSteps = params.maxTimeSteps or 30;
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['?'];

    local out = self.encoder:forward({hist})
    local encOut = out[1]
    local img_fc7 = out[2]
    self.forwardConnect(encOut, self.encoder, self.decoder, hist:size(1))

    -- start generation for each time step
    local numHist = 10;
    local questionIn = torch.Tensor(1, numHist):fill(startToken);
    local question = {questionIn:t():double()};
    -- print(#answerIn)

    local qLen = 1
    local quesWords = torch.Tensor(maxTimeSteps):zero()
    quesWords[1] = startToken
    for timeStep = 1, maxTimeSteps do
        -- one pass through the decoder
        local decOut = self.decoder:forward(questionIn):squeeze();
        -- connect decoder to itself
        self:decoderConnect(self.decoder);

        -- sampling with temperature
        nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);
        -- take the max value
        -- _, nextToken = torch.topk(decOut, 1, true);

        -- add to the answer, exit if endToken
        table.insert(question, nextToken:double());
        if nextToken == nil or nextToken:size(1) ~= 10 or nextToken:size(2) ~= 1 then print(nextToken) end
        quesWords[qLen+1] = nextToken[iter][1]
        qLen = qLen + 1
        questionIn:copy(nextToken);
        if nextToken[iter][1] == endToken then break end
        if qLen == maxTimeSteps then break end
    end

    -- join all the outputs
    question = nn.JoinTable(-1):forward(question);

    local encInSeq = hist[{{}, {iter}}]
    local histWords = encInSeq:double():squeeze()
    -- local quesWords = question[{{iter},{}}]:squeeze();
    -- local qLen = finishBeams[1].length;

    local histText = utils.idToWords(histWords, dataloader.ind2word);
    local quesText = utils.idToWords(quesWords, dataloader.ind2word);

    -- for i = 1, beamSize do print(utils.idToWords(finishBeams[i].beam:squeeze(), dataloader.ind2word)) end
    -- if iter == 1 then print(histText) end
    self.wrapper:training();
    return {quesWords:double(), qLen, quesText, img_fc7:float()}
end

function VisDialQModel:generateSingleQuestionSampleBatched(dataloader, hist, params, iter)
    local sampleOutput = params.sample or false;
    local temperature = params.temperature or 1.0;
    local maxTimeSteps = params.maxTimeSteps or 30;
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['?'];

    local out = self.encoder:forward({hist})
    local encOut = out[1]
    local img_fc7 = out[2]
    self.forwardConnect(encOut, self.encoder, self.decoder, hist:size(1))

    -- start generation for each time step
    local numHist = hist:size(2);
    local questionIn = torch.Tensor(1, numHist):fill(startToken);
    local question = {questionIn:t():double()};
    -- print(#answerIn)

    local qLen = torch.Tensor(params.batchSize):fill(1);
    local quesWords = torch.Tensor(params.batchSize, maxTimeSteps):zero();
    quesWords[{{}, {1}}]:fill(startToken);
    local qEnd = torch.Tensor(params.batchSize):zero();
    for timeStep = 1, maxTimeSteps do
        -- one pass through the decoder
        local decOut = self.decoder:forward(questionIn):squeeze();
        -- connect decoder to itself
        self:decoderConnect(self.decoder);

        -- sampling with temperature
        nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);
        -- take the max value
        -- _, nextToken = torch.topk(decOut, 1, true);

        -- add to the answer, exit if endToken
        table.insert(question, nextToken:double());
        -- if nextToken == nil or nextToken:size(1) ~= 10 or nextToken:size(2) ~= 1 then print(nextToken) end
        for i = 1, params.batchSize do
            if qEnd[i] ~= 1 then
                quesWords[i][qLen[i]+1] = nextToken[10 * (i-1) + iter][1]
                qLen[i] = qLen[i]  + 1;
            end
            if nextToken[10 * (i-1) + iter][1] == endToken then qEnd[i] = 1 end
        end
        questionIn:copy(nextToken);
        if torch.sum(qEnd)  == params.batchSize then break end
        if timeStep == maxTimeSteps-1 then break end
    end

    -- join all the outputs
    question = nn.JoinTable(-1):forward(question);

    local quesText = {};
    for i = 1, params.batchSize do
        local encInSeq = hist[{{}, {10 * (i-1) + iter}}]
        local histWords = encInSeq:double():squeeze()

        local histText = utils.idToWords(histWords, dataloader.ind2word);
        table.insert(quesText, utils.idToWords(quesWords[i], dataloader.ind2word))
    end

    self.wrapper:training();
    return {quesWords, qLen, quesText, img_fc7:float()}
end
-- beam search to generate a single question
function VisDialQModel:generateSingleQuestion(dataloader, hist, params, iter)
    params = params or {};

    local beamSize = params.beamSize or 5;
    local beamLen = params.beamLen or 20;
    local iter = iter or 1
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['?'];
    local out = self.encoder:forward({hist})
    local encOut = out[1]
    local img_fc7 = out[2]
    self.forwardConnect(encOut, self.encoder, self.decoder, hist:size(1))

    -- beams
    local beams = torch.LongTensor(beamLen, beamSize):zero():cuda();

    -- initial hidden states for the beam at current round of dialog
    local hiddenBeams = {};
    for level = 1, #self.encoder.histLayers do
        if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
        hiddenBeams[level]['output'] = self.encoder.histLayers[level].output[hist:size(1)][iter];
        hiddenBeams[level]['cell'] = self.encoder.histLayers[level].cell[hist:size(1)][iter];
        if level == #self.encoder.histLayers then
            hiddenBeams[#self.encoder.histLayers]['output'] = encOut[iter]
        end
        hiddenBeams[level]['output'] = torch.repeatTensor(hiddenBeams[level]['output'], beamSize, 1);
        hiddenBeams[level]['cell'] = torch.repeatTensor(hiddenBeams[level]['cell'], beamSize, 1);
    end
    -- hiddenBeams[]['cell'] is beam_nums x 512
    -- hiddenBeams[]['output'] is beam_nums x 512

    -- for first step, initialize with start symbols
    beams[1] = dataloader.word2ind['<START>'];
    scores = torch.DoubleTensor(beamSize):zero();
    finishBeams = {}; -- accumulate beams that are done

    for step = 2, beamLen do

        -- candidates for the current iteration
        cands = {};

        -- if step == 2, explore only one beam (all are <START>)
        local exploreSize = (step == 2) and 1 or beamSize;

        -- first copy the hidden states to the decoder
        for level = 1, #self.encoder.histLayers do
            self.decoder.rnnLayers[level].userPrevOutput = hiddenBeams[level]['output']
            self.decoder.rnnLayers[level].userPrevCell = hiddenBeams[level]['cell']
        end

        -- decoder forward pass
        decOut = self.decoder:forward(beams[{{step-1}}]);
        decOut = decOut:squeeze(); -- decOut is beam_nums x vocab_size

        -- iterate separately for each possible word of beam
        for wordId = 1, exploreSize do
            local curHidden = {};
            for level = 1, #self.decoder.rnnLayers do
                if curHidden[level] == nil then curHidden[level] = {} end
                curHidden[level]['output'] = self.decoder.rnnLayers[level].output[{{1},{wordId}}]:clone():squeeze(); -- rnnLayers[].output is 1 x beam_nums x 512
                curHidden[level]['cell'] = self.decoder.rnnLayers[level].cell[{{1},{wordId}}]:clone():squeeze();
            end

            sampleWords = false;
            if sampleWords == true then
                -- sample and get top probabilities
                topInd = torch.multinomial(torch.exp(decOut[wordId] / temperature), beamSize);
                topProb = decOut[wordId]:index(1, topInd);
            else
                -- sort and get the top probabilities
                if beamSize == 1 then
                    topProb, topInd = torch.topk(decOut, beamSize, true);
                else
                    topProb, topInd = torch.topk(decOut[wordId], beamSize, true);
                end
            end

            for candId = 1, beamSize do
                local candBeam = beams[{{}, {wordId}}]:clone();
                -- get the updated cost for each explored candidate, pool
                candBeam[step] = topInd[candId];
                if topInd[candId] == endToken then
                    -- table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = scores[wordId] + topProb[candId]});
                    -- Length penalty; encourages longer responses
                    table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = (scores[wordId] * (step-1) + topProb[candId])/step});
                    -- Length + diversity
                    -- table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = (scores[wordId] * (step-1) + topProb[candId] - 10.0 * candId)/step});
                else
                    -- table.insert(cands, {score = scores[wordId] + topProb[candId],
                    table.insert(cands, {score = (scores[wordId] * (step-1) + topProb[candId])/step,
                    -- table.insert(cands, {score = (scores[wordId] * (step-1) + topProb[candId] - 10.0 * candId)/step,
                                            beam = candBeam,
                                            hidden = curHidden});
                end
            end
        end

        -- sort the candidates and stick to beam size
        table.sort(cands, function (a, b) return a.score > b.score; end);

        for candId = 1, math.min(#cands, beamSize) do
            beams[{{}, {candId}}] = cands[candId].beam;

            --recursive copy
            for level = 1, #self.decoder.rnnLayers do
                hiddenBeams[level]['output'][candId] = cands[candId].hidden[level]['output']:clone();
                hiddenBeams[level]['cell'][candId] = cands[candId].hidden[level]['cell']:clone();
            end

            scores[candId] = cands[candId].score;
        end
    end
    table.sort(finishBeams, function (a, b) return a.score > b.score; end);

    if finishBeams[1] == nil then
        table.insert(finishBeams, {beam = cands[1].beam:double():squeeze(), length = 20})
    end

    local encInSeq = hist[{{}, {iter}}]
    local histWords = encInSeq:double():squeeze()
    local quesWords = finishBeams[1].beam:squeeze();
    local qLen = finishBeams[1].length;

    local histText = utils.idToWords(histWords, dataloader.ind2word);
    local quesText = utils.idToWords(quesWords, dataloader.ind2word);

    -- for i = 1, beamSize do print(utils.idToWords(finishBeams[i].beam:squeeze(), dataloader.ind2word)) end
    -- if iter == 1 then print(histText) end
    self.wrapper:training();
    return {quesWords, qLen, quesText, img_fc7:float()}
end

-- Satwik's beam search code
function VisDialQModel:generateQuestionsBeamSearch(dataloader, dtype, params)
    -- setting the options for beams search
    params = params or {};

    -- sample or take max
    local sampleWords = params.sampleWords or false;
    local temperature = params.temperature or 0.8;
    local beamSize = params.beamSize or 5;
    local beamLen = params.beamLen or 20;

    -- endToken index
    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['?'];
    local numThreads = dataloader.numThreads[dtype];

    -- Tensor to save predicted fc7s
    local img_preds = torch.FloatTensor(numThreads * 10, 4096)

    local answerTable = {}
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        self.wrapper:evaluate()

        local inds = torch.LongTensor(1):fill(convId);
        local batch = dataloader:getIndexDataQuestioner(inds, self.params, dtype);
        local numQues = batch['question_in']:size(1) * batch['question_in']:size(2);

        local out = self:encoderPass(batch)
        local encOut = out[1]
        local img_fc7 = out[2]
        img_preds[{{(convId - 1) * 10 + 1, convId * 10}, {}}] = img_fc7:float()
        local threadAnswers = {}

        -- do it for each example now
        for iter = 1, 10 do
            local encInSeq = batch['hist']:view(-1, batch['hist']:size(3)):t();
            encInSeq = encInSeq[{{},{iter}}]:squeeze():float()

            -- beams
            local beams = torch.LongTensor(beamLen, beamSize):zero():cuda();

            -- initial hidden states for the beam at current round of dialog
            local hiddenBeams = {};
            for level = 1, #self.encoder.histLayers do
                if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
                hiddenBeams[level]['output'] = self.encoder.histLayers[level].output[batch['hist']:size(3)][iter];
                hiddenBeams[level]['cell'] = self.encoder.histLayers[level].cell[batch['hist']:size(3)][iter];
                if level == #self.encoder.histLayers then
                    hiddenBeams[#self.encoder.histLayers]['output'] = encOut[iter]
                end
                hiddenBeams[level]['output'] = torch.repeatTensor(hiddenBeams[level]['output'], beamSize, 1);
                hiddenBeams[level]['cell'] = torch.repeatTensor(hiddenBeams[level]['cell'], beamSize, 1);
            end
            -- hiddenBeams[]['cell'] is beam_nums x 512
            -- hiddenBeams[]['output'] is beam_nums x 512

            -- for first step, initialize with start symbols
            beams[1] = dataloader.word2ind['<START>'];
            scores = torch.DoubleTensor(beamSize):zero();
            finishBeams = {}; -- accumulate beams that are done

            for step = 2, beamLen do

                -- candidates for the current iteration
                cands = {};

                -- if step == 2, explore only one beam (all are <START>)
                local exploreSize = (step == 2) and 1 or beamSize;

                -- first copy the hidden states to the decoder
                for level = 1, #self.encoder.histLayers do
                    self.decoder.rnnLayers[level].userPrevOutput = hiddenBeams[level]['output']
                    self.decoder.rnnLayers[level].userPrevCell = hiddenBeams[level]['cell']
                end

                -- decoder forward pass
                decOut = self.decoder:forward(beams[{{step-1}}]);
                decOut = decOut:squeeze(); -- decOut is beam_nums x vocab_size

                -- iterate separately for each possible word of beam
                for wordId = 1, exploreSize do
                    local curHidden = {};
                    for level = 1, #self.decoder.rnnLayers do
                        if curHidden[level] == nil then curHidden[level] = {} end
                        curHidden[level]['output'] = self.decoder.rnnLayers[level].output[{{1},{wordId}}]:clone():squeeze(); -- rnnLayers[].output is 1 x beam_nums x 512
                        curHidden[level]['cell'] = self.decoder.rnnLayers[level].cell[{{1},{wordId}}]:clone():squeeze();
                    end

                    sampleWords = false;
                    if sampleWords == true then
                        -- sample and get top probabilities
                        topInd = torch.multinomial(torch.exp(decOut[wordId] / temperature), beamSize);
                        topProb = decOut[wordId]:index(1, topInd);
                    else
                        -- sort and get the top probabilities
                        if beamSize == 1 then
                            topProb, topInd = torch.topk(decOut, beamSize, true);
                        else
                            topProb, topInd = torch.topk(decOut[wordId], beamSize, true);
                        end
                    end

                    for candId = 1, beamSize do
                        local candBeam = beams[{{}, {wordId}}]:clone();
                        -- get the updated cost for each explored candidate, pool
                        candBeam[step] = topInd[candId];
                        if topInd[candId] == endToken then
                            table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = scores[wordId] + topProb[candId]});
                        else
                            table.insert(cands, {score = scores[wordId] + topProb[candId],
                                                    beam = candBeam,
                                                    hidden = curHidden});
                        end
                    end
                end

                -- sort the candidates and stick to beam size
                table.sort(cands, function (a, b) return a.score > b.score; end);

                for candId = 1, math.min(#cands, beamSize) do
                    beams[{{}, {candId}}] = cands[candId].beam;

                    --recursive copy
                    for level = 1, #self.decoder.rnnLayers do
                        hiddenBeams[level]['output'][candId] = cands[candId].hidden[level]['output']:clone();
                        hiddenBeams[level]['cell'][candId] = cands[candId].hidden[level]['cell']:clone();
                    end

                    scores[candId] = cands[candId].score;
                end
            end

            table.sort(finishBeams, function (a, b) return a.score > b.score; end);

            local histWords = encInSeq:double():squeeze()
            local quesWords = finishBeams[1].beam:squeeze();

            local histText = utils.idToWords(histWords, dataloader.ind2word);
            local quesText = utils.idToWords(quesWords, dataloader.ind2word);

            table.insert(threadAnswers, {history=histText, question=quesText})

            -- for step = 1, encInSeq:size(1) do
                -- if encInSeq[step] > 0 then
                    -- io.write(dataloader.ind2word[encInSeq[step]]..' ')
                -- end
            -- end
            -- io.write('\n')

            -- -- spit out all the finished beams, and also the top beams
            -- print('Finished:', iter)
            -- for candId = 1, math.min(#finishBeams, beamSize) do
                -- local string = '';
                -- for step = 1, finishBeams[candId].length do
                    -- local ind = finishBeams[candId].beam[step];
                    -- string = string .. ' ' .. dataloader.ind2word[ind];
                -- end
                -- print(string, finishBeams[candId].score)
            -- end

            -- print('Unfinished:')
            -- for candId = 1, beamSize do
                -- for step = 1, beamLen do
                    -- io.write(dataloader.ind2word[beams[step][candId] ]..' ')
                -- end
                -- io.write('\n')
            -- end
        end
        self.wrapper:training()
        table.insert(answerTable, threadAnswers)
    end
    return {answerTable, img_preds}
end

return VisDialQModel;
