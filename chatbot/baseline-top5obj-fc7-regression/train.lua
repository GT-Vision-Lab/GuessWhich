-- code to train an fc7 regressor from top-k category labels

require 'nn'
require 'optim_updates'

require 'hdf5'
require 'xlua'

require 'cunn'
require 'cutorch'

cmd = torch.CmdLine()

cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_img.h5','h5file path with image feature')
cmd:option('-input_json','data/visdial_0.5/chat_processed_params.json','json path with info and vocab')

cmd:option('-save_path', 'checkpoints/', 'path to save the model and checkpoints')
cmd:option('-model_name', 'top5-labels-fc7-regression', 'Name of the model to use for answering')

cmd:option('-img_norm', 0, 'normalize the image feature. 1=yes, 0=no')
cmd:option('-softmax', 0, 'pass scores through softmax. 1=yes, 0=no')
cmd:option('-topk', 5, 'topk labels to consider')
cmd:option('-hidden_size', 1000, 'hidden layer size')

-- optimization params
cmd:option('-batch_size', 30, 'batch size');
cmd:option('-learning_rate', 1e-4, 'learning rate');
cmd:option('-dropout', 0, 'dropout for language model');
cmd:option('-max_epochs', 100, 'max epochs');
cmd:option('-gpuid', 0, 'GPU id to use');
cmd:option('-backend', 'cudnn', 'nn|cudnn');

local opts = cmd:parse(arg);

-- create checkpoint folder
local cur_time = os.date('*t', os.time());
if opts.model_name ~= 'test' then
    model_path = string.format('%s/model-%d-%d-%d-%d:%d:%d-%s',
                                    opts.save_path, cur_time.month, cur_time.day, cur_time.year, 
                                    cur_time.hour, cur_time.min, cur_time.sec, opts.model_name)
    -- paths.mkdir(model_path)
end

-- Loading data
local img_data = hdf5.open(opts.input_img_h5, 'r')
train_data = {}
train_data['scores'] = img_data:read('images_train'):all():cuda()
train_data['fc7'] = img_data:read('images_train_fc7'):all():cuda()

test_data = {}
test_data['scores'] = img_data:read('images_val'):all():cuda()
test_data['fc7'] = img_data:read('images_val_fc7'):all():cuda()

print('Train feat size: ' .. train_data['scores']:size(1))
print('Test feat size: ' .. test_data['scores']:size(1))

if opts.softmax == 1 then
    print('Softmaxing...')
    train_data['scores'] = nn.SoftMax():cuda():forward(train_data['scores'])
    test_data['scores'] = nn.SoftMax():cuda():forward(test_data['scores'])
end

if opts.img_norm == 1 then
    print('Normalizing...')
    train_data['fc7'] = nn.Normalize(2):cuda():forward(train_data['fc7'])
    test_data['fc7'] = nn.Normalize(2):cuda():forward(test_data['fc7'])
end

function get_batch(dtype, start_id)
    local batch = {}
    if dtype == 'train' then
        local inds = torch.LongTensor(opts.batch_size):random(1, train_data['scores']:size(1))
        batch['img_in'] = train_data['scores']:index(1, inds)
        batch['img_out'] = train_data['fc7']:index(1, inds)
    elseif dtype == 'test' then
        local end_id = math.min(start_id + opts.batch_size - 1, test_data['scores']:size(1))
        local inds = torch.LongTensor(end_id - start_id + 1)
        for ii = start_id, end_id do inds[ii - start_id + 1] = ii end
        batch['img_in'] = test_data['scores']:index(1, inds)
        batch['img_out'] = test_data['fc7']:index(1, inds)
    end
    return batch
end

function get_bow(inds)
    local out = torch.Tensor(inds:size(1), 1000):zero()
    for i = 1, inds:size(1) do
        for j = 1, opts.topk do
            out[i][inds[i][j]] = 1
        end
    end
    return out:cuda()
end

function get_bow_prob(inds, prob)
    local out = torch.Tensor(inds:size(1), 1000):zero()
    for i = 1, inds:size(1) do
        for j = 1, opts.topk do
            out[i][inds[i][j]] = prob[i][j]
        end
    end
    return out:cuda()
end

num_train = math.ceil(train_data['scores']:size(1) / opts.batch_size)
num_test = math.ceil(test_data['scores']:size(1) / opts.batch_size)

print('No. train batches: ' .. num_train)
print('No. test batches: ' .. num_test)

-- define model
model = nn.Sequential()
            :add(nn.Linear(1000, opts.hidden_size))
            :add(nn.ReLU())
            :add(nn.Dropout(0.5))
            :add(nn.Linear(opts.hidden_size, 4096))

criterion = nn.MSECriterion()
-- weight init
model = require('weight-init')(model, 'xavier');

model:cuda()
criterion:cuda()

print(model)
print(criterion)

params, grad_params = model:getParameters();
optim_params = {learningRate = opts.learning_rate}
running_loss = 0
multiplier = 1
best_test_loss = 10000

print('Training...')
for epoch = 1, opts.max_epochs do
    for iter = 1, num_train do
        local current_epoch = epoch - 1 + (iter / num_train)
        model:training()
        model:zeroGradParameters()
        -- forward pass
        local batch = get_batch('train')
        local prob, idx = torch.topk(batch['img_in'], opts.topk, true) -- true is for largest elements
        -- local input = get_bow_prob(idx, prob)
        local input = get_bow(idx)
        local output = model:forward(input)
        local loss = multiplier * criterion:forward(output, batch['img_out'])
        if running_loss > 0 then
            running_loss = 0.95 * running_loss + 0.05 * loss
        else
            running_loss = loss
        end
        if iter % 100 == 0 then
            print(string.format('[TRAIN][Epoch:%.02f][Loss:%.05f]', current_epoch, running_loss))
        end
        -- backward pass
        local doutput = multiplier * criterion:backward(output, batch['img_out'])
        local dinput = model:backward(input, doutput)
        grad_params:clamp(-5.0, 5.0);
        -- parameter update
        adam(params, grad_params, optim_params)
    end
    if epoch % 4 == 0 then
        model:evaluate()
        local loss_test = 0;
        local start_id = 1
        for test_iter = 1, num_test do
            local batch = get_batch('test', start_id)
            local prob, idx = torch.topk(batch['img_in'], opts.topk, true) -- true is for largest element
            -- local input = get_bow_prob(idx, prob)
            local input = get_bow(idx)
            local output = model:forward(input)
            loss_test = loss_test + multiplier * criterion:forward(output, batch['img_out'])
            start_id = start_id + opts.batch_size;
        end
        loss_test = loss_test / num_test;
        print(string.format('[TEST][Epoch:%.02f][Loss:%.05f]', epoch, loss_test))
        if loss_test < best_test_loss and opts.model_name ~= 'test' then
            best_test_loss = loss_test
            print('Saving model and predictions...')
            local hdf = hdf5.open(string.format('results/%s.h5', opts.model_name, epoch, loss_test), 'w')
            local fc7_preds = torch.Tensor(test_data['fc7']:size(1), 4096)
            local start_id = 1
            for test_iter = 1, num_test do
                local end_id = math.min(start_id + opts.batch_size - 1, test_data['scores']:size(1))
                local batch = get_batch('test', start_id)
                local prob, idx = torch.topk(batch['img_in'], opts.topk, true) -- true is for largest elements
                -- local input = get_bow_prob(idx, prob)
                local input = get_bow(idx)
                local output = model:forward(input)
                fc7_preds[{{start_id, end_id}, {}}] = output:float()
                start_id = start_id + opts.batch_size
            end
            hdf:write('fc7_test', fc7_preds)
            hdf:close()

            -- torch.save(string.format('%s/epoch-%.02f_loss-%.02f.t7', model_path, epoch, loss_test), {model = model, opts = opts, params = params})
        end
    end
end
