-- lstm based models
local lstm = {};

function lstm.buildModel(params)
    -- return encoder, nil, decoder
    return lstm:EncoderNet(params), lstm:DecoderNet(params);
end

function lstm.EncoderNet(self, params)
    local dropout = params.dropout or 0.2;
    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- encoder network
    local enc = nn.Sequential();

    -- create the two branches
    local concat = nn.ConcatTable();

    -- word branch, along with embedding layer
    self.wordEmbed = nn.LookupTableMaskZero(params.vocabSize, params.embedSize);
    -- local wordBranch = nn.Sequential():add(nn.SelectTable(1)):add(self.wordEmbed);
    -- -- add additional dropout after embedding
    -- --wordBranch:add(nn.Dropout(0.5));

    -- language model
    -- enc.rnnLayers = {};
    -- for layer = 1, params.numLayers do
        -- local inputSize = (layer==1) and (params.embedSize)
                                    -- or params.rnnHiddenSize;
        -- enc.rnnLayers[layer] = nn.SeqLSTM(inputSize, params.rnnHiddenSize);
        -- enc.rnnLayers[layer]:maskZero();

        -- wordBranch:add(enc.rnnLayers[layer]);
    -- end
    -- wordBranch:add(nn.Select(1, -1));

    -- make clones for embed layer
    -- local qEmbedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    local hEmbedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');

    -- create two branches
    local histBranch = nn.Sequential()
                            :add(nn.SelectTable(1))
                            :add(hEmbedNet);
    enc.histLayers = {};
    -- number of layers to read the history
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize 
                                    or params.rnnHiddenSize;
        enc.histLayers[layer] = nn.SeqLSTM(inputSize, params.rnnHiddenSize); 
        enc.histLayers[layer]:maskZero();

        histBranch:add(enc.histLayers[layer]);
    end
    histBranch:add(nn.Select(1, -1));

    -- image branch
    -- embedding for images
    -- local imgPar = nn.ParallelTable()
                        -- :add(nn.Identity())
                        -- :add(nn.Sequential()
                                -- :add(nn.Dropout(0.5))
                                -- :add(nn.Linear(params.imgFeatureSize,
                                                -- params.imgEmbedSize)));
    -- select words and image only
    -- local imageBranch = nn.Sequential()
                            -- :add(nn.SelectTable(2))
                            -- :add(nn.Dropout(0.5))
                            -- :add(nn.Linear(params.imgFeatureSize, params.imgEmbedSize))
                            -- :add(nn.NarrowTable(1, 2)) 
                            -- :add(imgPar)
                            -- :add(nn.MaskTime(params.imgEmbedSize));

    -- add concatTable and join
    -- concat:add(wordBranch)
    concat:add(histBranch)
    -- concat:add(imageBranch)
    enc:add(concat);

    -- another concat table
    -- local concat2 = nn.ConcatTable();

    -- select words + history + image
    -- local wordHistImageBranch = nn.Sequential()
                                -- :add(nn.NarrowTable(1, 2))
                                -- :add(nn.JoinTable(-1))

    -- -- add both the branches (wordImage, select history) to concat2
    -- concat2:add(wordHistImageBranch):add(nn.SelectTable(1));
    -- enc:add(concat2);
    enc:add(nn.JoinTable(1, 1))
    -- -- join both the tensors
    -- enc:add(nn.JoinTable(-1));
    --change the view of the data
    -- always split it back wrt batch size and then do transpose
    enc:add(nn.View(-1, params.maxQuesCount, params.rnnHiddenSize));
    enc:add(nn.Transpose({1, 2}));
    enc:add(nn.View(params.maxQuesCount, -1, params.rnnHiddenSize))
    enc:add(nn.SeqLSTM(params.rnnHiddenSize, params.rnnHiddenSize))
    enc:add(nn.Transpose({1, 2}));
    enc:add(nn.View(-1, params.rnnHiddenSize))

    local concat3 = nn.ConcatTable();

    concat3:add(nn.Identity())
    concat3:add(nn.Sequential()
        :add(nn.Dropout(0.5))
        :add(nn.Linear(params.rnnHiddenSize, params.rnnHiddenSize))
        :add(nn.ReLU())
        :add(nn.Dropout(0.5))
        :add(nn.Linear(params.rnnHiddenSize, params.imgFeatureSize))
        -- :add(nn.Normalize(2))
    )

    enc:add(concat3)

    return enc;
end

function lstm.DecoderNet(self, params)
    local dropout = params.dropout or 0.2;
    -- Use `nngraph`
    nn.FastLSTM.usenngraph = true;

    -- decoder network
    local dec = nn.Sequential();
    -- use the same embedding for both encoder and decoder lstm
    local embedNet = self.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias');
    dec:add(embedNet);
    -- add additional dropout after embedding
    --dec:add(nn.Dropout(0.5));

    dec.rnnLayers = {};
    -- check if decoder has different hidden size
    local hiddenSize = (params.ansHiddenSize ~= 0) and params.ansHiddenSize
                                            or params.rnnHiddenSize;
    for layer = 1, params.numLayers do
        local inputSize = (layer == 1) and params.embedSize or hiddenSize;
        dec.rnnLayers[layer] = nn.SeqLSTM(inputSize, hiddenSize);
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
function lstm.forwardConnect(encOut, enc, dec, seqLen)
    for ii = 1, #enc.histLayers do
        -- print('ii', ii)
        dec.rnnLayers[ii].userPrevOutput = enc.histLayers[ii].output[seqLen];
        dec.rnnLayers[ii].userPrevCell = enc.histLayers[ii].cell[seqLen];
    end

    -- last layer gets output gradients
    dec.rnnLayers[#enc.histLayers].userPrevOutput = encOut;
end

-- transfer gradients from decoder to encoder
function lstm.backwardConnect(enc, dec)
    -- borrow gradients from decoder
    for ii = 1, #dec.rnnLayers do
        enc.histLayers[ii].userNextGradCell = dec.rnnLayers[ii].userGradPrevCell;
        enc.histLayers[ii].gradPrevOutput = dec.rnnLayers[ii].userGradPrevOutput;
    end

    -- return the gradients for the last layer
    return dec.rnnLayers[#enc.histLayers].userGradPrevOutput;
end
return lstm;
