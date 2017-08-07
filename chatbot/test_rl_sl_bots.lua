require 'nn'
require 'cjson'
require 'evaluate_chatbot'

cmd = torch.CmdLine()
cmd:text('Testing class-based instantiation of Chatbot models')
cmd:text('Options')
cmd:option('-input_json','/srv/share/abhshkdz/visdial_rl_iccv17/visdial_0.5/chat_processed_params.json','json path with info and vocab')
cmd:option('-load_path_q', '/srv/share/abhshkdz/visdial_rl_iccv17/code/abhishek/checkpoints/qbot_hre_qih_sl.t7', 'path to saved questioner model')
cmd:option('-load_path_a', '/srv/share/abhshkdz/visdial_rl_iccv17/code/abhishek/checkpoints/abot_hre_qih_sl.t7', 'path to saved answerer model')
cmd:option('-gpuid', 12, 'GPU to use')
cmd:option('-backend', 'cudnn', 'Backend to use')
opt = cmd:parse(arg)

-- Create image feature tensor randomly
-- Image-IDs range from 1-100
im_feats = {}
for i=1, 100 do
	table.insert(im_feats, torch.rand(4096))
end

torch.save('random_imfeats.t7', im_feats)

-- Instantiate model-class
local model_class = ConversationModel(opt.input_json, opt.load_path_q, opt.load_path_a, opt.gpuid, opt.backend, 'random_imfeats.t7')


-- Test ABot model
local abot_result = model_class:ABot(10, '', 'What is the man doing?')
print(abot_result)

-- Create history table
hist = {abot_result['history']}
-- Test QBot model
local qbot_result = model_class:QBot(hist, 2)
print(qbot_result)
-- print(qbot_result['predicted_fc7'])