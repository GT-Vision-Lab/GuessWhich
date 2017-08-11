-- abstract class for models
require 'optim_updates'
require 'xlua'
require 'hdf5'

local utils = require 'utils'

local VisDialAModel = torch.class('VisDialAModel');

-- initialize
function VisDialAModel:__init(params)
    print('Setting up SVQA model..\n');
    self.params = params;
    -- print(params)
    -- build the model - encoder, decoder and answerNet
    local modelFile = string.format('%s/specificModel.lua', params.model_name);
    local model = dofile(modelFile);
    enc, dec = model:buildSpecificModel(params);
    -- add methods from specific model
    local methods = {'forwardConnect', 'backwardConnect',
                    'forwardBackward', 'retrieveBatch', 'encoderPass'};
    for key, value in pairs(methods) do self[value] = model[value]; end

    print('Encoder:\n'); print(enc);

    -- criterion
    self.criterion = nn.ClassNLLCriterion();
    self.criterion.sizeAverage = false;
    self.criterion = nn.SequencerCriterion(
                                nn.MaskZeroCriterion(self.criterion, 1));

    -- wrap the models
    self.wrapper = nn.Sequential():add(enc):add(dec);

    -- initialize weights
    self.wrapper = require('weight-init')(self.wrapper, 'xavier');
    -- ship to gpu if necessary
    if params.gpuid >= 0 then
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
function VisDialAModel:trainIteration(dataloader)
    -- clear the gradients
    self.wrapper:zeroGradParameters();

    -- grab a training batch
    local batch = dataloader:getTrainBatch(self.params);

    -- call the internal function for model specific actions
    local curLoss = self:forwardBackward(batch);
    -- count the number of tokens
    local numTokens = torch.sum(batch['answer_out']:gt(0));

    -- update the running average of loss
    if runningLoss > 0 then
        runningLoss = 0.95 * runningLoss + 0.05 * curLoss/numTokens;
    else
        runningLoss = curLoss/numTokens;
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
end

---------------------------------------------------------------------
-- validation performance on test/val
function VisDialAModel:evaluate(dataloader, dtype)
    -- change to evaluate mode
    self.wrapper:evaluate();

    local curLoss = 0;
    local startId = 1;
    local numThreads = dataloader.numThreads[dtype];

    local numTokens = 0;
    while startId <= numThreads do
        -- print progress
        xlua.progress(startId, numThreads);

        -- grab a validation batch
        local batch, nextStartId
                        = dataloader:getTestBatch(startId, self.params, dtype);
        -- count the number of tokens
        numTokens = numTokens + torch.sum(batch['answer_out']:gt(0));
        -- forward pass to compute loss
        curLoss = curLoss + self:forwardBackward(batch, true);
        startId = nextStartId;
    end

    -- print the results
    curLoss = curLoss / numTokens;
    print(string.format('\n%s\tLoss: %f\t Perplexity: %f\n', dtype,
                        curLoss, math.exp(curLoss)));

    -- change back to training
    self.wrapper:training();
end

---------------------------------------------------------------------
-- retrieval performance on test/val
function VisDialAModel:retrieve(dataloader, dtype)
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
                        dataloader:getTestBatch(startId, self.params, dtype);

        -- Call retrieve function for specific model, and store ranks
        ranks[{{startId, nextStartId - 1}, {}}] = self:retrieveBatch(batch);
        startId = nextStartId;
    end

    -- if lambdaRange is given
    torch.save('ranks_answerer.t7', ranks:float())
    print('saved to ranks_answerer.t7')

    print(string.format('\n%s - Retrieval:', dtype))
    utils.processRanks(ranks);

    -- change back to training
    self.wrapper:training();

    -- collect garbage
    collectgarbage();
end

---------------------------------------------------------------------
-- generating answers for questions
function VisDialAModel:generateAnswers(dataloader, dtype, genParams)
    local sampleOutput = genParams.sample;
    local temperature = genParams.temperature or 1.0;
    local maxTimeSteps = genParams.maxTimeSteps or self.params.maxAnsLen;
    local numConvs = genParams.numConvs or 2;

    -- change the model to evaluate model
    self.wrapper:evaluate();

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];
    local numThreads = dataloader.numThreads[dtype];
    -- ABHSHKDZ
    -- numThreads = 500

    local answerTable = {};
    local answerWts = {};
    -- generate for the first few questions
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        -- print the conversation id
        --print(string.format('Conversation: %d', convId))
        local inds = torch.LongTensor(1):fill(convId);
        local batch = dataloader:getIndexData(inds, self.params, dtype);
        local numQues = batch['ques_fwd']:size(1) * batch['ques_fwd']:size(2);

        -- one pass through the encoder
        -- check for additional attributes to save
        local modelSave = self:encoderPass(batch);

        -- start generation for each time step
        local answerIn = torch.Tensor(1, numQues):fill(startToken);
        local answer = {answerIn:t():double()};

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

        -- join all the outputs
        answer = nn.JoinTable(-1):forward(answer);

        -- convert each question and answer to text
        local threadAnswers = {};
        for quesId = 1, self.params.maxQuesCount do
            local quesWords = batch['ques_fwd'][{{1}, {quesId}, {}}]:squeeze():double();
            local ansWords = answer[{{quesId}, {}}]:squeeze();

            local quesText = utils.idToWords(quesWords, dataloader.ind2word);
            local ansText = utils.idToWords(ansWords, dataloader.ind2word);

            table.insert(threadAnswers, {question=quesText, answer=ansText});
        end
        table.insert(answerTable, threadAnswers);
        -- check if there is additional save
        if modelSave then table.insert(answerWts, modelSave); end
    end
    -- save if non-empty
    if #answerWts > 0 then
        -- join all the answer weights
        local attWts = nn.JoinTable(1):forward(answerWts);
        local filePt = hdf5.open('results/im-hist-enc-dec-att-wts.h5', 'w');
        filePt:write('weights', attWts:double());
        filePt:close();
    end

    -- going back to training mode
    self.wrapper:training();

    return answerTable;
end

-- connecting decoder to itself
function VisDialAModel:decoderConnect(dec, seqLen)
    for ii = 1, #dec.rnnLayers do
        dec.rnnLayers[ii].userPrevOutput = dec.rnnLayers[ii].output[1]
    end

    -- if language model is lstm
    if self.params.languageModel == 'lstm' then
        for ii = 1, #dec.rnnLayers do
            dec.rnnLayers[ii].userPrevCell = dec.rnnLayers[ii].cell[1]
        end
    end
end

-- sampling
function VisDialAModel:generateSingleAnswerSample(dataloader, batch, params, iter)
    local sampleOutput = params.sample or false;
    local temperature = params.temperature or 1.0;
    local maxTimeSteps = params.maxTimeSteps or 30;
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];

    local encOut = self.encoder:forward({batch[3], batch[1], batch[2]})
    self.forwardConnect(encOut, self.encoder, self.decoder, batch[3]:size(1))

    -- start generation for each time step
    local numQues = 10;
    local answerIn = torch.Tensor(1, numQues):fill(startToken);
    local answer = {answerIn:t():double()};
    -- print(#answerIn)

    local aLen = 1
    local ansWords = torch.Tensor(maxTimeSteps):zero()
    ansWords[1] = startToken
    for timeStep = 1, maxTimeSteps do
        -- one pass through the decoder
        local decOut = self.decoder:forward(answerIn):squeeze();
        -- connect decoder to itself
        self:decoderConnect(self.decoder);

        -- sampling with temperature
        nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);
        -- take the max value
        -- _, nextToken = torch.topk(decOut, 1, true);

        -- add to the answer, exit if endToken
        table.insert(answer, nextToken:double());
        ansWords[aLen+1] = nextToken[iter][1]
        aLen = aLen + 1
        answerIn:copy(nextToken);
        if nextToken[iter][1] == endToken then break end
        if aLen == maxTimeSteps then break end
    end

    -- join all the outputs
    answer = nn.JoinTable(-1):forward(answer);

    local encInSeq = batch[3][{{},{iter}}]
    local quesWords = encInSeq:double():squeeze()

    local quesText = utils.idToWords(quesWords, dataloader.ind2word);
    local ansText = utils.idToWords(ansWords, dataloader.ind2word);

    self.wrapper:training();
    return {ansWords, aLen, ansText}
end

-- sampling
function VisDialAModel:generateSingleAnswerSampleBatched(dataloader, batch, params, iter)
    local sampleOutput = params.sample or false;
    local temperature = params.temperature or 1.0;
    local maxTimeSteps = params.maxTimeSteps or 30;
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];

    local encOut = self.encoder:forward({batch[3], batch[1], batch[2]})
    self.forwardConnect(encOut, self.encoder, self.decoder, batch[3]:size(1))

    -- start generation for each time step
    local numQues = batch[1]:size(2);
    local answerIn = torch.Tensor(1, numQues):fill(startToken);
    local answer = {answerIn:t():double()};
    -- print(#answerIn)

    local aLen = torch.Tensor(params.batchSize):fill(1);
    local ansWords = torch.Tensor(params.batchSize, maxTimeSteps):zero()
    ansWords[{{}, {1}}]:fill(startToken);
    local aEnd = torch.Tensor(params.batchSize):zero()
    for timeStep = 1, maxTimeSteps do
        -- one pass through the decoder
        local decOut = self.decoder:forward(answerIn):squeeze();
        -- connect decoder to itself
        self:decoderConnect(self.decoder);

        -- sampling with temperature
        nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);
        -- take the max value
        -- _, nextToken = torch.topk(decOut, 1, true);

        -- add to the answer, exit if endToken
        table.insert(answer, nextToken:double());
        for i = 1, params.batchSize do
            if aEnd[i] ~= 1 then
                ansWords[i][aLen[i]+1] = nextToken[10 * (i-1) + iter][1]
                aLen[i] = aLen[i] + 1;
            end
            if nextToken[10 * (i-1) + iter][1] == endToken then aEnd[i] = 1 end
        end
        answerIn:copy(nextToken);
        if torch.sum(aEnd) == params.batchSize then break end
        if timeStep == maxTimeSteps-1 then break end
    end

    -- join all the outputs
    answer = nn.JoinTable(-1):forward(answer);

    local ansText = {};
    local quesText = {};
    for i = 1, params.batchSize do
        local encInSeq = batch[3][{{},{10 * (i-1) + iter}}]
        local quesWords = encInSeq:double():squeeze()

        table.insert(quesText, utils.idToWords(quesWords, dataloader.ind2word));
        table.insert(ansText, utils.idToWords(ansWords[i], dataloader.ind2word));
    end

    self.wrapper:training();
    return {ansWords, aLen, ansText, quesText}
end

-- full sampling. returns batch_size x max_rounds tensor
function VisDialAModel:generateSingleAnswerSampleBatchedFull(dataloader, batch, params)
    local sampleOutput = params.sample or false;
    local temperature = params.temperature or 1.0;
    local maxTimeSteps = params.maxTimeSteps or 30;
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];

    local encOut = self.encoder:forward({batch[3], batch[1], batch[2]})
    self.forwardConnect(encOut, self.encoder, self.decoder, batch[3]:size(1))

    -- start generation for each time step
    local numQues = batch[1]:size(2);
    local answerIn = torch.Tensor(1, numQues):fill(startToken);
    local answer = {answerIn:t():double()};

    local aLen = torch.Tensor(params.batchSize * 10):fill(1);
    local ansWords = torch.Tensor(params.batchSize * 10, maxTimeSteps):zero()
    ansWords[{{}, {1}}]:fill(startToken);
    local aEnd = torch.Tensor(params.batchSize * 10):zero()
    for timeStep = 1, maxTimeSteps do
        -- one pass through the decoder
        local decOut = self.decoder:forward(answerIn):squeeze();
        -- connect decoder to itself
        self:decoderConnect(self.decoder);

        -- sampling with temperature
        nextToken = torch.multinomial(torch.exp(decOut / temperature), 1);

        -- add to the answer, exit if endToken
        table.insert(answer, nextToken:double());
        for i = 1, params.batchSize * 10 do
            if aEnd[i] ~= 1 then
                ansWords[i][aLen[i]+1] = nextToken[i][1]
                aLen[i] = aLen[i] + 1;
            end
            if nextToken[i][1] == endToken then aEnd[i] = 1 end
        end
        answerIn:copy(nextToken);
        if torch.sum(aEnd) == params.batchSize * 10 then break end
        if timeStep == maxTimeSteps-1 then break end
    end

    -- join all the outputs
    answer = nn.JoinTable(-1):forward(answer);

    local ansText = {};
    local quesText = {};
    for i = 1, params.batchSize * 10 do
        local encInSeq = batch[3][{{},{i}}]
        local quesWords = encInSeq:double():squeeze()

        table.insert(quesText, utils.idToWords(quesWords, dataloader.ind2word));
        table.insert(ansText, utils.idToWords(ansWords[i], dataloader.ind2word));
    end

    self.wrapper:training();
    return {ansWords, aLen, ansText, quesText}
end

-- beam search to generate a single answer
function VisDialAModel:generateSingleAnswer(dataloader, batch, params, iter)
    params = params or {};

    local beamSize = params.beamSize or 5;
    local beamLen = params.beamLen or 20;
    local iter = iter or 1
    self.wrapper:evaluate()

    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];

    local encOut = self.encoder:forward({batch[3], batch[1], batch[2]})
    self.forwardConnect(encOut, self.encoder, self.decoder, batch[3]:size(1))

    -- beams
    local beams = torch.LongTensor(beamLen, beamSize):zero():cuda();

    -- initial hidden states for the beam at current round of dialog
    local hiddenBeams = {};
    for level = 1, #self.encoder.rnnLayers do
        if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
        hiddenBeams[level]['output'] = self.encoder.rnnLayers[level].output[batch[3]:size(1)][iter];
        hiddenBeams[level]['cell'] = self.encoder.rnnLayers[level].cell[batch[3]:size(1)][iter];
        if level == #self.encoder.rnnLayers then
            hiddenBeams[#self.encoder.rnnLayers]['output'] = encOut[iter]
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
        for level = 1, #self.encoder.rnnLayers do
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
                    table.insert(finishBeams, {beam = candBeam:double():squeeze(), length = step, score = (scores[wordId] * (step-1) + topProb[candId]) / step});
                else
                    -- table.insert(cands, {score = scores[wordId] + topProb[candId],
                                            -- beam = candBeam,
                                            -- hidden = curHidden});
                    table.insert(cands, {score = (scores[wordId] * (step-1) + topProb[candId]) / step,
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

    local encInSeq = batch[3][{{},{iter}}]
    local quesWords = encInSeq:double():squeeze()
    local ansWords = finishBeams[1].beam:squeeze();
    local aLen = finishBeams[1].length;

    local quesText = utils.idToWords(quesWords, dataloader.ind2word);
    local ansText = utils.idToWords(ansWords, dataloader.ind2word);

    self.wrapper:training();
    return {ansWords, aLen, ansText}
end

-- Satwik's beam search code
function VisDialAModel:generateAnswersBeamSearch(dataloader, dtype, params)
    -- setting the options for beams search
    params = params or {};

    -- sample or take max
    -- local sampleWords = params.sampleWords or false;
    -- local temperature = params.temperature or 0.8;
    local beamSize = params.beamSize or 5;
    local beamLen = params.beamLen or 20;

    -- endToken index
    local startToken = dataloader.word2ind['<START>'];
    local endToken = dataloader.word2ind['<END>'];
    local numThreads = dataloader.numThreads[dtype];
    print('numThreads', numThreads)

    local answerTable = {}
    for convId = 1, numThreads do
        xlua.progress(convId, numThreads);
        self.wrapper:evaluate()

        local inds = torch.LongTensor(1):fill(convId);
        local batch = dataloader:getIndexData(inds, self.params, dtype);
        local numQues = batch['ques_fwd']:size(1) * batch['ques_fwd']:size(2);

        local encOut = self:encoderPass(batch)
        local threadAnswers = {}

        -- do it for each example now
        for iter = 1, 10 do
            local encInSeq = batch['ques_fwd']:view(-1, batch['ques_fwd']:size(3)):t();
            encInSeq = encInSeq[{{},{iter}}]:squeeze():float()

            -- beams
            local beams = torch.LongTensor(beamLen, beamSize):zero():cuda();

            -- initial hidden states for the beam at current round of dialog
            local hiddenBeams = {};
            for level = 1, #self.encoder.rnnLayers do
                if hiddenBeams[level] == nil then hiddenBeams[level] = {} end
                hiddenBeams[level]['output'] = self.encoder.rnnLayers[level].output[batch['ques_fwd']:size(3)][iter];
                hiddenBeams[level]['cell'] = self.encoder.rnnLayers[level].cell[batch['ques_fwd']:size(3)][iter];
                if level == #self.encoder.rnnLayers then
                    hiddenBeams[#self.encoder.rnnLayers]['output'] = encOut[iter]
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
                for level = 1, #self.encoder.rnnLayers do
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

            local quesWords = encInSeq:double():squeeze()
            local ansWords = finishBeams[1].beam:squeeze();

            local quesText = utils.idToWords(quesWords, dataloader.ind2word);
            local ansText = utils.idToWords(ansWords, dataloader.ind2word);

            table.insert(threadAnswers, {question=quesText, answer=ansText})

        end
        self.wrapper:training()
        table.insert(answerTable, threadAnswers)
    end
    return answerTable
end


return VisDialAModel;
