-- script to load the dataloader for meta-rnn
require 'hdf5'
require 'xlua'
local utils = require 'utils'

local dataloader = {};

-- read the data
-- params: object itself, command line options, 
--          subset of data to load (train, test, val)
function dataloader:initialize(opt, subsets)
    -- read additional info like dictionary, etc
    print('DataLoader loading h5 file: ', opt.input_json)
    info = utils.readJSON(opt.input_json);
    for key, value in pairs(info) do dataloader[key] = value; end

    -- add <START> and <END> to vocabulary
    count = 0;
    for _ in pairs(dataloader['word2ind']) do count = count + 1; end
    dataloader['word2ind']['<START>'] = count + 1;
    dataloader['word2ind']['<END>'] = count + 2;
    count = count + 2;
    dataloader.vocabSize = count;
    print(string.format('Vocabulary size (with <START>,<END>): %d\n', count));

    -- construct ind2word
    local ind2word = {};
    for word, ind in pairs(dataloader['word2ind']) do
        ind2word[ind] = word;
    end
    dataloader['ind2word'] = ind2word;

    -- read questions, answers and options
    -- print('DataLoader loading h5 file: ', opt.input_ques_h5)
    -- local quesFile = hdf5.open(opt.input_ques_h5, 'r');

    -- print('DataLoader loading h5 file: ', opt.input_img_h5)
    -- local imgFile = hdf5.open(opt.input_img_h5, 'r');
    -- number of threads
    -- self.numThreads = {};

    -- for _, dtype in pairs(subsets) do
        -- read question related information
        -- self[dtype..'_ques'] = quesFile:read('ques_'..dtype):all();
        -- self[dtype..'_ques_len'] = quesFile:read('ques_length_'..dtype):all();
        -- self[dtype..'_ques_count'] = quesFile:read('ques_count_'..dtype):all();

        -- read answer related information
        -- self[dtype..'_ans'] = quesFile:read('ans_'..dtype):all();
        -- self[dtype..'_ans_len'] = quesFile:read('ans_length_'..dtype):all();
        -- self[dtype..'_ans_ind'] = quesFile:read('ans_index_'..dtype):all():long();

        -- read image list, if image features are needed
        -- if opt.useIm then
            -- print('Reading image features ..')
            -- local imgFeats = imgFile:read('/images_'..dtype):all();

            -- Normalize the image feature(if needed)
            -- if opt.img_norm == 1 then
                -- print('Normalizing image features..')
                -- local nm = torch.sqrt(torch.sum(torch.cmul(imgFeats, imgFeats), 2));
                -- imgFeats = torch.cdiv(imgFeats, nm:expandAs(imgFeats)):float();
            -- end
            -- self[dtype..'_img_fv'] = imgFeats;
            -- TODO: make it 1 indexed in processing code
            -- currently zero indexed, adjust manually
            -- self[dtype..'_img_pos'] = quesFile:read('img_pos_'..dtype):all():long();
            -- self[dtype..'_img_pos'] = self[dtype..'_img_pos'] + 1;
        -- end

        -- print information for data type
        -- print(string.format('%s:\n\tNo. of threads: %d\n\tNo. of rounds: %d'..
                            -- '\n\tMax ques len: %d'..'\n\tMax ans len: %d\n',
                                -- dtype, self[dtype..'_ques']:size(1), 
                                        -- self[dtype..'_ques']:size(2),
                                        -- self[dtype..'_ques']:size(3),
                                        -- self[dtype..'_ans']:size(2)));
        -- record some stats
        -- if dtype == 'train' then
            -- self.numTrainThreads = self['train_ques']:size(1);
            -- self.numThreads['train'] = self.numTrainThreads;
        -- end
        -- if dtype == 'test' then
            -- self.numTestThreads = self['test_ques']:size(1);
            -- self.numThreads['test'] = self.numTestThreads;
        -- end
        -- if dtype == 'val' then
            -- self.numValThreads = self['val_ques']:size(1);
            -- self.numThreads['val'] = self.numValThreads;
        -- end

        -- record the options only for test and val
        -- if dtype == 'val' or dtype == 'test' then
            -- self[dtype..'_opt'] = quesFile:read('opt_'..dtype):all():long();
            -- self[dtype..'_opt_len'] = quesFile:read('opt_length_'..dtype):all();
            -- self[dtype..'_opt_list'] = quesFile:read('opt_list_'..dtype):all();
            -- self[dtype..'_opt_prob'] = torch.Tensor(self[dtype..'_opt_len']:size());
        -- end

        -- assume similar stats across multiple data subsets
        -- maximum number of questions per image, ideally 10
        -- self.maxQuesCount = self[dtype..'_ques']:size(2);
        -- maximum length of question
        -- self.maxQuesLen = self[dtype..'_ques']:size(3);
        -- maximum length of answer
        -- self.maxAnsLen = self[dtype..'_ans']:size(2);
        -- number of options, if read
        -- if self[dtype..'_opt'] then
            -- self.numOptions = self[dtype..'_opt']:size(3);
        -- end

        -- if history is needed
        -- if opt.useHistory then
            -- self[dtype..'_cap'] = quesFile:read('cap_'..dtype):all():long();
            -- self[dtype..'_cap_len'] = quesFile:read('cap_length_'..dtype):all();
        -- end
    -- end
    -- done reading, close files
    -- quesFile:close();
    -- imgFile:close();

    -- take desired flags/values from opt
    -- self.useBi = opt.useBi;
    -- self.useHistory = opt.useHistory;
    -- self.useIm = opt.useIm;
    -- self.separateCaption = opt.separateCaption;
    -- self.maxHistoryLen = opt.maxHistoryLen or 60;

    -- prepareDataset for training
    -- for _, dtype in pairs(subsets) do self:prepareDataset(dtype); end
end

-- read the data
-- params: object itself, command line options, 
--          subset of data to load (train, test, val)
function dataloader:initializeQuestioner(opt, subsets)
    -- read additional info like dictionary, etc
    print('DataLoader loading h5 file: ', opt.input_json)
    info = utils.readJSON(opt.input_json);
    for key, value in pairs(info) do dataloader[key] = value; end

    -- add <START> and <END> to vocabulary
    count = 0;
    for _ in pairs(dataloader['word2ind']) do count = count + 1; end
    dataloader['word2ind']['<START>'] = count + 1;
    dataloader['word2ind']['<END>'] = count + 2;
    count = count + 2;
    dataloader.vocabSize = count;
    print(string.format('Vocabulary size (with <START>,<END>): %d\n', count));

    -- construct ind2word
    local ind2word = {};
    for word, ind in pairs(dataloader['word2ind']) do
        ind2word[ind] = word;
    end
    dataloader['ind2word'] = ind2word;

    -- read questions, answers and options
    print('DataLoader loading h5 file: ', opt.input_ques_h5)
    local quesFile = hdf5.open(opt.input_ques_h5, 'r');

    print('DataLoader loading h5 file: ', opt.input_img_h5)
    local imgFile = hdf5.open(opt.input_img_h5, 'r');

    print('Dataloader loading question options h5: ', opt.input_ques_opts_h5)
    local quesOptsFile = hdf5.open(opt.input_ques_opts_h5, 'r');

    -- number of threads
    self.numThreads = {};

    for _, dtype in pairs(subsets) do
        -- read question related information
        self[dtype..'_ques'] = quesFile:read('ques_'..dtype):all();
        self[dtype..'_ques_len'] = quesFile:read('ques_length_'..dtype):all();
        self[dtype..'_ques_count'] = quesFile:read('ques_count_'..dtype):all();
        if dtype == 'test' then
            self[dtype..'_ques_ind'] = quesOptsFile:read('ques_index_'..dtype):all():long();
        end

        -- read answer related information
        self[dtype..'_ans'] = quesFile:read('ans_'..dtype):all();
        self[dtype..'_ans_len'] = quesFile:read('ans_length_'..dtype):all();
        self[dtype..'_ans_ind'] = quesFile:read('ans_index_'..dtype):all():long();

        -- read image list, if image features are needed
        if opt.useIm then
            print('Reading image features ..')
            local imgFeats = imgFile:read('/images_'..dtype):all();

            -- Normalize the image feature(if needed)
            -- NO
            -- opt.img_norm = 0
            if opt.img_norm == 1 then
                print('Normalizing image features..')
                local nm = torch.sqrt(torch.sum(torch.cmul(imgFeats, imgFeats), 2));
                imgFeats = torch.cdiv(imgFeats, nm:expandAs(imgFeats)):float();
            end
            self[dtype..'_img_fv'] = imgFeats;
            -- TODO: make it 1 indexed in processing code
            -- currently zero indexed, adjust manually
            self[dtype..'_img_pos'] = quesFile:read('img_pos_'..dtype):all():long();
            self[dtype..'_img_pos'] = self[dtype..'_img_pos'] + 1;
        end

        -- print information for data type
        print(string.format('%s:\n\tNo. of threads: %d\n\tNo. of rounds: %d'..
                            '\n\tMax ques len: %d'..'\n\tMax ans len: %d\n',
                                dtype, self[dtype..'_ques']:size(1), 
                                        self[dtype..'_ques']:size(2),
                                        self[dtype..'_ques']:size(3),
                                        self[dtype..'_ans']:size(2)));
        -- record some stats
        if dtype == 'train' then
            self.numTrainThreads = self['train_ques']:size(1);
            self.numThreads['train'] = self.numTrainThreads;
        end
        if dtype == 'test' then
            self.numTestThreads = self['test_ques']:size(1);
            self.numThreads['test'] = self.numTestThreads;
        end
        if dtype == 'val' then
            self.numValThreads = self['val_ques']:size(1);
            self.numThreads['val'] = self.numValThreads;
        end

        -- record the options only for test and val
        -- if dtype == 'val' or dtype == 'test' then
        if dtype == 'test' then
            self[dtype..'_opt'] = quesOptsFile:read('ques_opt_'..dtype):all():long();
            self[dtype..'_opt_len'] = quesOptsFile:read('ques_opt_length_'..dtype):all();
            self[dtype..'_opt_list'] = quesOptsFile:read('ques_opt_list_'..dtype):all();
            -- self[dtype..'_opt_prob'] = torch.Tensor(self[dtype..'_opt_len']:size());
        end

        -- assume similar stats across multiple data subsets
        -- maximum number of questions per image, ideally 10
        self.maxQuesCount = self[dtype..'_ques']:size(2);
        -- maximum length of question
        self.maxQuesLen = self[dtype..'_ques']:size(3);
        -- maximum length of answer
        self.maxAnsLen = self[dtype..'_ans']:size(2);
        -- number of options, if read
        if self[dtype..'_opt'] then
            self.numOptions = self[dtype..'_opt']:size(3);
        end

        -- if history is needed
        if opt.useHistory then
            self[dtype..'_cap'] = quesFile:read('cap_'..dtype):all():long();
            self[dtype..'_cap_len'] = quesFile:read('cap_length_'..dtype):all();
        end
    end
    -- done reading, close files
    quesFile:close();
    imgFile:close();

    -- take desired flags/values from opt
    self.useBi = opt.useBi;
    self.useHistory = opt.useHistory;
    self.useIm = opt.useIm;
    self.separateCaption = opt.separateCaption;
    self.maxHistoryLen = opt.maxHistoryLen or 60;

    -- prepareDataset for training
    for _, dtype in pairs(subsets) do self:prepareDatasetQuestioner(dtype); end
end

-- method to prepare questions and answers for retrieval
-- questions : right align
-- answers : prefix with <START> and <END>
function dataloader:prepareDataset(dtype)
    -- right align the questions
    print('Right aligning questions: '..dtype);
    self[dtype..'_ques_fwd'] = utils.rightAlign(self[dtype..'_ques'],
                                            self[dtype..'_ques_len']);

    -- if bidirectional model is needed store backward
    if self.useBi then 
        self[dtype..'_ques_bwd'] = nn.SeqReverseSequence(3)
                                        :forward(self[dtype..'_ques']:double());
        -- convert back to LongTensor
        self[dtype..'_ques_bwd'] = self[dtype..'_ques_bwd']:long();
    end
     
    -- if separate captions are needed
    if self.separateCaption then self:processHistoryCaption(dtype);
    else if self.useHistory then self:processHistory(dtype); end 
    end
    -- prefix options with <START> and <END>, if not train
    if dtype ~= 'train' then self:processOptions(dtype); end
    -- process answers
    self:processAnswers(dtype);
end

-- method to prepare questions and answers for retrieval
-- questions : right align
-- answers : prefix with <START> and <END>
function dataloader:prepareDatasetQuestioner(dtype)
    -- right align the questions
    print('Right aligning questions: '..dtype);
    self[dtype..'_ques_fwd'] = utils.rightAlign(self[dtype..'_ques'],
                                            self[dtype..'_ques_len']);
    -- if bidirectional model is needed store backward
    if self.useBi then 
        self[dtype..'_ques_bwd'] = nn.SeqReverseSequence(3)
                                        :forward(self[dtype..'_ques']:double());
        -- convert back to LongTensor
        self[dtype..'_ques_bwd'] = self[dtype..'_ques_bwd']:long();
    end
     
    -- if separate captions are needed
    if self.separateCaption then self:processHistoryCaption(dtype);
    else if self.useHistory then self:processHistory(dtype); end 
    end
    -- prefix options with <START> and <END>, if not train
    if dtype == 'test' then self:processOptionsQuestioner(dtype); end
    -- process answers
    self:processAnswers(dtype);
    self:processAnswersQuestioner(dtype);
end

-- process answers
function dataloader:processAnswers(dtype)
    --prefix answers with <START>, <END>; adjust answer lengths
    local answers = self[dtype..'_ans'];
    local ansLen = self[dtype..'_ans_len'];

    local numConvs = answers:size(1);
    local numRounds = answers:size(2);
    local maxAnsLen = answers:size(3);

    local decodeIn = torch.LongTensor(numConvs, numRounds, maxAnsLen+1):zero();
    local decodeOut = torch.LongTensor(numConvs, numRounds, maxAnsLen+1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, {}, 1}] = self.word2ind['<START>'];

    -- go over each answer and modify
    local endTokenId = self.word2ind['<END>'];
    for thId = 1, numConvs do
        for roundId = 1, numRounds do
            local length = ansLen[thId][roundId];

            -- only if nonzero
            if length > 0 then
                --decodeIn[thId][roundId][1] = startToken;
                decodeIn[thId][roundId][{{2, length + 1}}]
                                = answers[thId][roundId][{{1, length}}];
                decodeOut[thId][roundId][{{1, length}}]
                                = answers[thId][roundId][{{1, length}}];
                decodeOut[thId][roundId][length+1] = endTokenId;
            else
                print(string.format('Warning: empty answer at (%d %d %d)',
                                                    thId, roundId, length))
            end
        end
    end

    self[dtype..'_ans_len'] = self[dtype..'_ans_len'] + 1;
    self[dtype..'_ans_in'] = decodeIn;
    self[dtype..'_ans_out'] = decodeOut;
end
    
-- process answers
function dataloader:processAnswersQuestioner(dtype)
    --prefix answers with <START>, <END>; adjust answer lengths
    local questions = self[dtype..'_ques'];
    local quesLen = self[dtype..'_ques_len'];

    local numConvs = questions:size(1);
    local numRounds = questions:size(2);
    local maxQuesLen = questions:size(3);

    local decodeIn = torch.LongTensor(numConvs, numRounds, maxQuesLen+1):zero();
    local decodeOut = torch.LongTensor(numConvs, numRounds, maxQuesLen+1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, {}, 1}] = self.word2ind['<START>'];

    -- go over each answer and modify
    local endTokenId = self.word2ind['<END>'];
    for thId = 1, numConvs do
        for roundId = 1, numRounds do
            local length = quesLen[thId][roundId];

            -- only if nonzero
            if length > 0 then
                --decodeIn[thId][roundId][1] = startToken;
                decodeIn[thId][roundId][{{2, length + 1}}]
                                = questions[thId][roundId][{{1, length}}];
                            
                decodeOut[thId][roundId][{{1, length}}]
                                = questions[thId][roundId][{{1, length}}];
                decodeOut[thId][roundId][length+1] = endTokenId;
            else
                print(string.format('Warning: empty question at (%d %d %d)',
                                                    thId, roundId, length))
            end
        end
    end

    self[dtype..'_ques_len'] = self[dtype..'_ques_len'] + 1;
    self[dtype..'_ques_in'] = decodeIn;
    self[dtype..'_ques_out'] = decodeOut;
end

-- process caption as history
function dataloader:processHistory(dtype)
    local captions = self[dtype..'_cap'];
    local questions = self[dtype..'_ques'];
    local quesLen = self[dtype..'_ques_len'];
    local capLen = self[dtype..'_cap_len'];
    local maxQuesLen = questions:size(3);

    local answers = self[dtype..'_ans'];
    local ansLen = self[dtype..'_ans_len'];
    local numConvs = answers:size(1);
    local numRounds = answers:size(2);
    local maxAnsLen = answers:size(3);

    -- chop off caption to maxQuesLen
    local history = torch.LongTensor(numConvs, numRounds, 
                                    maxQuesLen+maxAnsLen):zero();
    local histLen = torch.LongTensor(numConvs, numRounds):zero();

    -- go over each question and append it with answer
    for thId = 1, numConvs do
        local lenC = capLen[thId];
        for roundId = 1, numRounds do
            local lenH; -- length of history
            if roundId == 1 then
                -- first round has caption as history
                history[thId][roundId]
                            = captions[thId][{{1, maxQuesLen + maxAnsLen}}];
                lenH = math.min(lenC, maxQuesLen + maxAnsLen);
            else
                -- other rounds have previous Q + A as history
                local lenQ = quesLen[thId][roundId-1];
                local lenA = ansLen[thId][roundId-1];

                if lenQ > 0 then
                    history[thId][roundId][{{1, lenQ}}]
                            = questions[thId][roundId-1][{{1, lenQ}}];
                end
                if lenA > 0 then
                    history[thId][roundId][{{lenQ + 1, lenQ + lenA}}]
                                = answers[thId][roundId-1][{{1, lenA}}];
                end
                lenH = lenA + lenQ;
            end
            -- save the history length
            histLen[thId][roundId] = lenH;
        end
    end

    -- right align history and then save
    print('Right aligning history: '..dtype);
    self[dtype..'_hist'] = utils.rightAlign(history, histLen);
    self[dtype..'_hist_len'] = histLen;
end

-- process history and captions separately
function dataloader:processHistoryCaption(dtype)
    local questions = self[dtype..'_ques'];
    local quesLen = self[dtype..'_ques_len'];
    local maxQuesLen = questions:size(3);

    local answers = self[dtype..'_ans'];
    local ansLen = self[dtype..'_ans_len'];
    local numConvs = answers:size(1);
    local numRounds = answers:size(2);
    local maxAnsLen = answers:size(3);

    -- chop off caption to maxQuesLen
    local history = torch.LongTensor(numConvs, numRounds, 
                    (self.maxQuesCount - 1) * (maxQuesLen + maxAnsLen)):zero();
    local histLen = torch.LongTensor(numConvs, numRounds):zero();

    -- go over each question and append it with answer
    for thId = 1, numConvs do
        -- current round as history for next rounds
        local runHistLen = 0;
        for prevId = 1, numRounds-1 do
            -- current Q and A as history
            local lenQ = quesLen[thId][prevId];
            local lenA = ansLen[thId][prevId];
            local curHistLen = lenA + lenQ;
            local curHistory = torch.LongTensor(curHistLen);
            curHistory[{{1, lenQ}}] = questions[thId][prevId][{{1, lenQ}}];
            curHistory[{{lenQ + 1, curHistLen}}] = answers[thId][prevId][{{1, lenA}}];

            for roundId = prevId + 1, numRounds do
                history[thId][roundId][{{runHistLen+1, runHistLen+curHistLen}}]
                                                                = curHistory;
            end

            -- increase the running count of history length
            runHistLen = runHistLen + curHistLen;
            histLen[thId][prevId + 1] = runHistLen;
        end
    end

    -- right align history and then save
    print('Right aligning history: '..dtype);
    self[dtype..'_hist'] = utils.rightAlign(history, histLen);
    self[dtype..'_hist_len'] = histLen;

    -- right align captions and then save
    print('Right aligning captions: '..dtype);
    self[dtype..'_cap'] = utils.rightAlign(self[dtype..'_cap'],
                                            self[dtype..'_cap_len']);
end

-- process options
function dataloader:processOptions(dtype)
    local lengths = self[dtype..'_opt_len'];
    local answers = self[dtype..'_ans'];
    local maxAnsLen = answers:size(3);
    local answers = self[dtype..'_opt_list'];
    local numConvs = answers:size(1);

    local ansListLen = answers:size(1);
    local decodeIn = torch.LongTensor(ansListLen, maxAnsLen + 1):zero();
    local decodeOut = torch.LongTensor(ansListLen, maxAnsLen + 1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, 1}] = self.word2ind['<START>'];

    -- go over each answer and modify
    local endTokenId = self.word2ind['<END>'];
    for id = 1, ansListLen do
        -- print progress for number of images
        if id % 100 == 0 then
            xlua.progress(id, numConvs);
        end
        local length = lengths[id];

        -- only if nonzero
        if length > 0 then
            decodeIn[id][{{2, length + 1}}] = answers[id][{{1, length}}];
                        
            decodeOut[id][{{1, length}}] = answers[id][{{1, length}}];
            decodeOut[id][length + 1] = endTokenId;
        else
            print(string.format('Warning: empty answer for %s at %d',
                                                            dtype, id))
        end
    end

    self[dtype..'_opt_len'] = self[dtype..'_opt_len'] + 1;
    self[dtype..'_opt_in'] = decodeIn;
    self[dtype..'_opt_out'] = decodeOut;

    collectgarbage();
end

-- process options
function dataloader:processOptionsQuestioner(dtype)
    local lengths = self[dtype..'_opt_len'];
    local questions = self[dtype..'_ques'];
    local maxQuesLen = questions:size(3);
    local questions = self[dtype..'_opt_list'];
    local numConvs = questions:size(1);

    local quesListLen = questions:size(1);
    local decodeIn = torch.LongTensor(quesListLen, maxQuesLen + 1):zero();
    local decodeOut = torch.LongTensor(quesListLen, maxQuesLen + 1):zero();

    -- decodeIn begins with <START>
    decodeIn[{{}, 1}] = self.word2ind['<START>'];

    -- go over each question and modify
    local endTokenId = self.word2ind['<END>'];
    for id = 1, quesListLen do
        -- print progress for number of images
        if id % 100 == 0 then
            xlua.progress(id, numConvs);
        end
        local length = lengths[id];

        -- only if nonzero
        if length > 0 then
            decodeIn[id][{{2, length + 1}}] = questions[id][{{1, length}}];
            decodeOut[id][{{1, length}}] = questions[id][{{1, length}}];
            decodeOut[id][length + 1] = endTokenId;
        else
            print(string.format('Warning: empty question for %s at %d',
                                                            dtype, id))
        end
    end

    self[dtype..'_opt_len'] = self[dtype..'_opt_len'] + 1;
    self[dtype..'_opt_in'] = decodeIn;
    self[dtype..'_opt_out'] = decodeOut;

    collectgarbage();
end

-- method to grab the next training batch
function dataloader.getTrainBatch(self, params, batchSize)
    local size = batchSize or params.batchSize;
    local inds = torch.LongTensor(size):random(1, params.numTrainThreads);

    -- Index question, answers, image features for batch
    return self:getIndexData(inds, params, 'train');
    --output['batch_ques'], output['answer_in'], output['answer_out'];
end

-- method to grab the next training batch
function dataloader.getTrainBatchQuestioner(self, params, batchSize)
    local size = batchSize or params.batchSize;
    local inds = torch.LongTensor(size):random(1, params.numTrainThreads);

    -- Index question, answers, image features for batch
    return self:getIndexDataQuestioner(inds, params, 'train');
    --output['batch_ques'], output['answer_in'], output['answer_out'];
end

-- method to grab the next test/val batch, for evaluation of a given size
function dataloader.getTestBatch(self, startId, params, dtype)
    local batchSize = params.batchSize * 4;
    -- get the next start id and fill up current indices till then
    local nextStartId;
    if dtype == 'val' then
        nextStartId = math.min(self.numValThreads+1, startId + batchSize);
    end
    if dtype == 'test' then
        nextStartId = math.min(self.numTestThreads+1, startId + batchSize);
    end

    -- dumb way to get range (complains if cudatensor is default)
    local inds = torch.LongTensor(nextStartId - startId);
    for ii = startId, nextStartId - 1 do inds[ii - startId + 1] = ii; end
    --local inds = torch.range(startId, nextStartId - 1):long();

    -- Index question, answers, image features for batch
    local batchOutput = self:getIndexData(inds, params, dtype);
    local optionOutput = self:getIndexOption(inds, params, dtype);

    -- merge both the tables and return
    for key, value in pairs(optionOutput) do batchOutput[key] = value; end

    return batchOutput, nextStartId;
end

-- method to grab the next test/val batch, for evaluation of a given size
function dataloader.getTestBatchQuestioner(self, startId, params, dtype)
    local batchSize = params.batchSize * 4;
    -- get the next start id and fill up current indices till then
    local nextStartId;
    if dtype == 'val' then
        nextStartId = math.min(self.numValThreads+1, startId + batchSize);
    end
    if dtype == 'test' then
        nextStartId = math.min(self.numTestThreads+1, startId + batchSize);
    end

    -- dumb way to get range (complains if cudatensor is default)
    local inds = torch.LongTensor(nextStartId - startId);
    for ii = startId, nextStartId - 1 do inds[ii - startId + 1] = ii; end
    --local inds = torch.range(startId, nextStartId - 1):long();

    -- Index question, answers, image features for batch
    local batchOutput = self:getIndexDataQuestioner(inds, params, dtype);
    if dtype == 'test' then
        local optionOutput = self:getIndexOptionQuestioner(inds, params, dtype);

        -- merge both the tables and return
        for key, value in pairs(optionOutput) do batchOutput[key] = value; end
    end

    return batchOutput, nextStartId;
end

-- get batch from data subset given the indices
function dataloader.getIndexData(self, inds, params, dtype)
    -- get the question lengths
    local batchQuesLen = self[dtype..'_ques_len']:index(1, inds);
    local maxQuesLen = torch.max(batchQuesLen);
    -- get questions
    local quesFwd = self[dtype..'_ques_fwd']:index(1, inds)
                                            [{{}, {}, {-maxQuesLen, -1}}];
    local quesBwd;
    if self.useBi then
        quesBwd = self[dtype..'_ques_bwd']:index(1, inds)
                                            [{{}, {}, {-maxQuesLen, -1}}];
    end

    local history;
    if self.useHistory then
        local batchHistLen = self[dtype..'_hist_len']:index(1, inds);
        local maxHistLen = math.min(torch.max(batchHistLen), self.maxHistoryLen);
        history = self[dtype..'_hist']:index(1, inds)
                                    [{{}, {}, {-maxHistLen, -1}}];
    end

    local caption;
    if self.separateCaption then
        local batchCapLen = self[dtype..'_cap_len']:index(1, inds);
        local maxCapLen = torch.max(batchCapLen);
        caption = self[dtype..'_cap']:index(1, inds)[{{}, {-maxCapLen, -1}}];
    end

    local imgFeats;
    if self.useIm then
        local imgInds = self[dtype..'_img_pos']:index(1, inds);
        imgFeats = self[dtype..'_img_fv']:index(1, imgInds);
    end
    
    -- get the answer lengths
    local batchAnsLen = self[dtype..'_ans_len']:index(1, inds);
    local maxAnsLen = torch.max(batchAnsLen);
    -- answer labels (decode input and output)
    local answerIn = self[dtype..'_ans_in']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerOut = self[dtype..'_ans_out']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerInd = self[dtype..'_ans_ind']:index(1, inds);

    local output = {};
    if params.gpuid >= 0 then
        -- TODO: instead store everything on gpu to save time
        output['ques_fwd'] = quesFwd:cuda();
        output['answer_in'] = answerIn:cuda();
        output['answer_out'] = answerOut:cuda();
        output['answer_ind'] = answerInd:cuda();
        if quesBwd then output['ques_bwd'] = quesBwd:cuda(); end
        if history then output['hist'] = history:cuda(); end
        if caption then output['cap'] = caption:cuda(); end
        if imgFeats then output['img_feat'] = imgFeats:cuda(); end
    else
        output['ques_fwd'] = quesFwd:contiguous();
        output['answer_in'] = answerIn:contiguous();
        output['answer_out'] = answerOut:contiguous();
        output['answer_ind'] = answerInd:contiguous();
        if quesBwd then output['ques_bwd'] = quesBwd:contiguous(); end
        if history then output['hist'] = history:contiguous(); end
        if caption then output['cap'] = caption:contiguous(); end
        if imgFeats then output['img_feat'] = imgFeats:contiguous(); end
    end

    return output;
end

-- get batch from data subset given the indices
function dataloader.getIndexDataQuestioner(self, inds, params, dtype)
    -- get the question lengths
    local batchQuesLen = self[dtype..'_ques_len']:index(1, inds);
    local maxQuesLen = torch.max(batchQuesLen) - 1;
    -- get questions
    local quesFwd = self[dtype..'_ques_fwd']:index(1, inds)
                                            [{{}, {}, {-maxQuesLen, -1}}];
    local quesBwd;
    if self.useBi then
        quesBwd = self[dtype..'_ques_bwd']:index(1, inds)
                                            [{{}, {}, {1, maxQuesLen}}];
    end

    local history;
    if self.useHistory then
        local batchHistLen = self[dtype..'_hist_len']:index(1, inds);
        local maxHistLen = math.min(torch.max(batchHistLen), self.maxHistoryLen);
        history = self[dtype..'_hist']:index(1, inds)
                                    [{{}, {}, {-maxHistLen, -1}}];
    end

    local caption;
    if self.separateCaption then
        local batchCapLen = self[dtype..'_cap_len']:index(1, inds);
        local maxCapLen = torch.max(batchCapLen);
        caption = self[dtype..'_cap']:index(1, inds)[{{}, {-maxCapLen, -1}}];
    end

    local imgFeats;
    if self.useIm then
        local imgInds = self[dtype..'_img_pos']:index(1, inds);
        imgFeats = self[dtype..'_img_fv']:index(1, imgInds);
    end
    
    -- get the question lengths
    local batchQuesLen = self[dtype..'_ques_len']:index(1, inds);
    local maxQuesLen = torch.max(batchQuesLen);
    -- answer labels (decode input and output)
    local questionIn = self[dtype..'_ques_in']
                                :index(1, inds)[{{}, {}, {1, maxQuesLen}}];
    local questionOut = self[dtype..'_ques_out']
                                :index(1, inds)[{{}, {}, {1, maxQuesLen}}];
    if dtype == 'test' then
        questionInd = self[dtype..'_ques_ind']:index(1, inds);
    end

    -- get the answer lengths
    local batchAnsLen = self[dtype..'_ans_len']:index(1, inds);
    local maxAnsLen = torch.max(batchAnsLen);
    -- answer labels (decode input and output)
    local answerIn = self[dtype..'_ans_in']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerOut = self[dtype..'_ans_out']
                                :index(1, inds)[{{}, {}, {1, maxAnsLen}}];
    local answerInd = self[dtype..'_ans_ind']:index(1, inds);

    local output = {};
    if params.gpuid >= 0 then
        -- TODO: instead store everything on gpu to save time
        output['ques_fwd'] = quesFwd:cuda();
        output['question_len'] = batchQuesLen:cuda();
        output['question_in'] = questionIn:cuda();
        output['question_out'] = questionOut:cuda();
        output['answer_in'] = answerIn:cuda();
        output['answer_out'] = answerOut:cuda();
        output['answer_ind'] = answerInd:cuda();
        if dtype == 'test' then
            output['question_ind'] = questionInd:cuda();
        end
        if quesBwd then output['ques_bwd'] = quesBwd:cuda(); end
        if history then output['hist'] = history:cuda(); end
        if caption then output['cap'] = caption:cuda(); end
        if imgFeats then output['img_feat'] = imgFeats:cuda(); end
    else
        output['ques_fwd'] = quesFwd:contiguous();
        output['question_in'] = questionIn:contiguous();
        output['question_out'] = questionOut:contiguous();
        output['question_ind'] = questionInd:contiguous();
        if quesBwd then output['ques_bwd'] = quesBwd:contiguous(); end
        if history then output['hist'] = history:contiguous(); end
        if caption then output['cap'] = caption:contiguous(); end
        if imgFeats then output['img_feat'] = imgFeats:contiguous(); end
    end

    return output;
end

-- get batch from options given the indices
function dataloader.getIndexOption(self, inds, params, dtype)
    local optionIn, optionOut, optionProb, answerProb;

    local optInds = self[dtype..'_opt']:index(1, inds);
    local indVector = optInds:view(-1);

    local batchOptLen = self[dtype..'_opt_len']:index(1, indVector);
    local maxOptLen = torch.max(batchOptLen);

    optionIn = self[dtype..'_opt_in']:index(1, indVector);
    optionIn = optionIn:view(optInds:size(1), optInds:size(2),
                                            optInds:size(3), -1);
    optionIn = optionIn[{{}, {}, {}, {1, maxOptLen}}];

    optionOut = self[dtype..'_opt_out']:index(1, indVector);
    optionOut = optionOut:view(optInds:size(1), optInds:size(2),
                                            optInds:size(3), -1);
    optionOut = optionOut[{{}, {}, {}, {1, maxOptLen}}];

    if self[dtype..'_opt_prob'] then
        optionProb = self[dtype..'_opt_prob']:index(1, indVector);
        optionProb = optionProb:viewAs(optInds);

        -- also get the answer probabilities
        local answerInds = self[dtype..'_ans_ind']:index(1, inds);
        indVector = answerInds:view(-1);
        answerProb = self[dtype..'_opt_prob']:index(1, indVector);
        answerProb = answerProb:viewAs(answerInds);
    end

    local output = {};
    if params.gpuid >= 0 then
        output['option_in'] = optionIn:cuda();
        output['option_out'] = optionOut:cuda();
        if optionProb then
            output['option_prob'] = optionProb:cuda();
            output['answer_prob'] = answerProb:cuda();
        end
    else
        output['option_in'] = optionIn:contiguous();
        output['option_out'] = optionOut:contiguous();
        if optionProb then
            output['option_prob'] = optionProb:contiguous();
            output['answer_prob'] = answerProb:contiguous();
        end
    end

    return output;
end

-- get batch from options given the indices
function dataloader.getIndexOptionQuestioner(self, inds, params, dtype)
    local optionIn, optionOut, optionProb, answerProb;

    local optInds = self[dtype..'_opt']:index(1, inds);
    local indVector = optInds:view(-1);

    local batchOptLen = self[dtype..'_opt_len']:index(1, indVector);
    local maxOptLen = torch.max(batchOptLen);

    optionIn = self[dtype..'_opt_in']:index(1, indVector);
    optionIn = optionIn:view(optInds:size(1), optInds:size(2),
                                            optInds:size(3), -1);
    optionIn = optionIn[{{}, {}, {}, {1, maxOptLen}}];

    optionOut = self[dtype..'_opt_out']:index(1, indVector);
    optionOut = optionOut:view(optInds:size(1), optInds:size(2),
                                            optInds:size(3), -1);
    optionOut = optionOut[{{}, {}, {}, {1, maxOptLen}}];

    if self[dtype..'_opt_prob'] then
        optionProb = self[dtype..'_opt_prob']:index(1, indVector);
        optionProb = optionProb:viewAs(optInds);

        -- also get the answer probabilities
        local answerInds = self[dtype..'_ans_ind']:index(1, inds);
        indVector = answerInds:view(-1);
        answerProb = self[dtype..'_opt_prob']:index(1, indVector);
        answerProb = answerProb:viewAs(answerInds);
    end

    local output = {};
    if params.gpuid >= 0 then
        output['option_in'] = optionIn:cuda();
        output['option_out'] = optionOut:cuda();
        if optionProb then 
            output['option_prob'] = optionProb:cuda();
            output['answer_prob'] = answerProb:cuda();
        end
    else
        output['option_in'] = optionIn:contiguous();
        output['option_out'] = optionOut:contiguous();
        if optionProb then
            output['option_prob'] = optionProb:contiguous();
            output['answer_prob'] = answerProb:contiguous();
        end
    end

    return output;
end

return dataloader;
