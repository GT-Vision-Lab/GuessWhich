cmd = torch.CmdLine()
cmd:text('Train the s-vqa model for retrieval')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data/visdial_0.5/data_img.h5','h5file path with image feature')
cmd:option('-input_ques_h5','data/visdial_0.5/chat_processed_data.h5','h5file file with preprocessed questions')
cmd:option('-input_ques_opts_h5','utils/working/qoptions_cap_qatm1mean_sample25k_visdial_0.5_test.h5','h5file file with preprocessed question options')
cmd:option('-input_json','data/visdial_0.5/chat_processed_params.json','json path with info and vocab')

cmd:option('-save_path', 'models/', 'path to save the model and checkpoints')
cmd:option('-model_name', 'im-hist-enc-dec-answerer', 'Name of the model to use for answering')

cmd:option('-img_norm', 1, 'normalize the image feature. 1=yes, 0=no')
cmd:option('-load_path_a', '', 'path to saved answerer model')

-- model params
cmd:option('-metaHiddenSize', 100, 'Size of the hidden layer for meta-rnn');
cmd:option('-multiEmbedSize', 1024, 'Size of multimodal embedding for q+i')
cmd:option('-imgEmbedSize', 300, 'Size of the multimodal embeddings');
cmd:option('-imgFeatureSize', 4096, 'Size of the deep image feature');
cmd:option('-embedSize', 300, 'Size of input word embeddings')
cmd:option('-rnnHiddenSize', 512, 'Size of the hidden language rnn in each layer')
cmd:option('-ansHiddenSize', 0, 'Size of the hidden language rnn in each layer for answers')
cmd:option('-maxHistoryLen', 60, 'Maximum history to consider when using appended qa pairs');
cmd:option('-numLayers', 2, 'number of the rnn layer')
cmd:option('-languageModel', 'lstm', 'rnn to use for language model, lstm | gru')
cmd:option('-bidirectional', 0, 'Bi-directional language model')
cmd:option('-metric', 'llh', 'Metric to use for retrieval, llh | mi')
cmd:option('-lambda', '1.0', 'Factor for marginalized probability for mi metric')

-- optimization params
cmd:option('-batchSize', 30, 'Batch size (number of threads) (Adjust base on GRAM)');
cmd:option('-probSampleSize', 50, 'Number of samples for computing probability');
cmd:option('-learningRate', 1e-3, 'Learning rate');
cmd:option('-dropout', 0, 'Dropout for language model');
cmd:option('-numEpochs', 400, 'Epochs');
cmd:option('-LRateDecay', 10, 'After lr_decay epochs lr reduces to 0.1*lr');
cmd:option('-lrDecayRate', 0.9997592083, 'Decay for learning rate')
cmd:option('-minLRate', 5e-5, 'Minimum learning rate');
cmd:option('-gpuid', 0, 'GPU id to use')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

local opts = cmd:parse(arg);

-- if save path is not given, use default..time
-- get the current time
local curTime = os.date('*t', os.time());
-- create another folder to avoid clutter
local modelPath = string.format('models/model-%d-%d-%d-%d:%d:%d-%s/',
                                curTime.month, curTime.day, curTime.year, 
                                curTime.hour, curTime.min, curTime.sec, opts.model_name)
if opts.save_path == 'models/' then opts.save_path = modelPath end;
-- add useMI flag if the metric is mutual information
if opts.metric == 'mi' then opts.useMI = true; end
if opts.bidirectional == 0 then opts.useBi = nil; else opts.useBi = true; end
-- additionally check if its imitation of discriminative model
if string.match(opts.model_name, 'hist') then 
    opts.useHistory = true;
    if string.match(opts.model_name, 'disc') then
        opts.separateCaption = true;
    end
end
if string.match(opts.model_name, 'im') then opts.useIm = true; end

return opts;
