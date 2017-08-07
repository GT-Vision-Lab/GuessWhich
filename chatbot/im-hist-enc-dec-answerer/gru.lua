-- gru based models
local gru = {};

function gru.buildModel(params)
    -- return encoder, nil, decoder
    return gru:EncoderNet(params), gru:DecoderNet(params);
end

function gru.EncoderNet(self, params)
    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- encoder network
    local enc = nn.Sequential();

    -- create the two branches
    local concat = nn.ConcatTable();

    -- word branch, along with embedding layer
    self.wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);
    local wordBranch = nn.Sequential():add(nn.SelectTable(1)):add(self.wordEmbed);

    -- make clones for embed layer
    local qEmbedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    local hEmbedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');

    -- create two branches
    local histBranch = nn.Sequential()
                            :add(nn.SelectTable(3))
                            :add(hEmbedNet);
    enc.histLayers = {};
    -- number of layers to read the history
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize 
                                    or params.rnnHiddenSize;
        enc.histLayers[layer] = nn.SeqGRU(inputSize, params.rnnHiddenSize); 
        enc.histLayers[layer]:maskZero();

        histBranch:add(enc.histLayers[layer]);
    end
    histBranch:add(nn.Select(1, -1));

    -- image branch
    -- embedding for images
    local imgPar = nn.ParallelTable()
                        :add(nn.Identity())
                        :add(nn.Sequential()
                                :add(nn.Dropout(0.5))
                                :add(nn.Linear(params.imgFeatureSize,
                                                params.imgEmbedSize)));
    -- select words and image only
    local imageBranch = nn.Sequential()
                            :add(nn.NarrowTable(1, 2)) 
                            :add(imgPar)
                            :add(nn.MaskTime(params.imgEmbedSize));

    -- add concatTable and join
    concat:add(wordBranch)
    concat:add(imageBranch)
    concat:add(histBranch)
    enc:add(concat);

    -- another concat table
    local concat2 = nn.ConcatTable();

    -- select words + image, and history
    local wordImageBranch = nn.Sequential()
                                :add(nn.NarrowTable(1, 2))
                                :add(nn.JoinTable(-1))

    -- language model
    enc.rnnLayers = {};
    for layer = 1, params.numLayers do
        local inputSize = (layer==1) and (params.imgEmbedSize+params.embedSize)
                                    or params.rnnHiddenSize;
        enc.rnnLayers[layer] = nn.SeqGRU(inputSize, params.rnnHiddenSize);
        enc.rnnLayers[layer]:maskZero();

        wordImageBranch:add(enc.rnnLayers[layer]);
    end
    wordImageBranch:add(nn.Select(1, -1));

    -- add both the branches (wordImage, select history) to concat2
    concat2:add(wordImageBranch):add(nn.SelectTable(3));
    enc:add(concat2);
    -- join both the tensors
    enc:add(nn.JoinTable(-1));
    --change the view of the data
    enc:add(nn.View(params.maxQuesCount, -1, 2*params.rnnHiddenSize))
    enc:add(nn.SeqGRU(2*params.rnnHiddenSize, params.rnnHiddenSize))
    enc:add(nn.View(-1, params.rnnHiddenSize))

    return enc;
end

function gru.AnswerNet(params, dropout)
    return metarnn[params.modelName](params, dropout);
    --return metarnn.MetaRNNVanilla(params, dropout);
end

function gru.DecoderNet(self, params)
    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- decoder network
    local dec = nn.Sequential();
    -- use the same embedding for both encoder and decoder gru
    local embedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    dec:add(embedNet);

    dec.rnnLayers = {};
    -- check if decoder has different hidden size
    local hiddenSize = (params.ansHiddenSize ~= 0) and params.ansHiddenSize
                                            or params.rnnHiddenSize;
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize or hiddenSize;
        dec.rnnLayers[layer] = nn.SeqGRU(inputSize, hiddenSize);
        dec.rnnLayers[layer]:maskZero();

        dec:add(dec.rnnLayers[layer]);
    end
    dec:add(nn.Sequencer(nn.MaskZero(
                            nn.Linear(hiddenSize, params.vocabSize), 1)))
    dec:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))
    
    return dec;
end
-------------------------------------------------------------------------------
-- transfer the hidden state from encoder to decoder
function gru.forwardConnect(encOut, dec)
    for ii = 1, #dec.rnnLayers do
        dec.rnnLayers[ii].userPrevOutput = encOut;
    end
end

-- transfer gradients from decoder to encoder
function gru.backwardConnect(dec)
    for ii = 1, #dec.rnnLayers do
        return dec.rnnLayers[ii].userGradPrevOutput;
    end
end

return gru;
