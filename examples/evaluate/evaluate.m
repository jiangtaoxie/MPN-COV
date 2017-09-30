% EVALUATE  
% This script can evaluate models using 1-crop or 10-crop method
% We noticed that using the image-resize function matlab built-in
% can improve preformance slightly than vl_imreadjpeg(version:beta22)
% Created by Jiangtao Xie
run(fullfile(fileparts(mfilename('fullpath')),'..', '..', 'matlab', 'vl_setupnn.m')) ;
model_type = 'MPN-COV-ResNet-50';
dataset = 'ILSVRC2012'; % dataset ILSVRC2012
evaluate_mode = 'val';
%% image preprocess setting details
opts.scale = 256;        % [256 384 512]
opts.method = '1-crop';  %  '1-crop' or '10-crop'
opts.crop_size = 224;    %  crop size
opts.averageImage = [124.2441 116.5862 103.4211]; % averageImage
%% load Model
switch dataset
    case 'ILSVRC2012'
        dataDir = fullfile(vl_rootnn,'data',dataset,'images');
    otherwise
        
end
switch model_type
    case 'MPN-COV-ResNet-50'
        modelPath = fullfile(vl_rootnn,'pretrained', 'imagenet-MPN-COV-ResNet-50-dag.mat') ;
    case 'MPN-COV-ResNet-101'
        modelPath = fullfile(vl_rootnn,'pretrained', 'imagenet-MPN-COV-ResNet-101-dag.mat') ;
    case 'MPN-COV-AlexNet'
        modelPath = fullfile(vl_rootnn,'pretrained', 'imagenet-MPN-COV-AlexNet.mat') ;
    case 'MPN-COV-VGG-M'
        modelPath = fullfile(vl_rootnn,'pretrained', 'imagenet-MPN-COV-VGG-M.mat') ;
    case 'MPN-COV-VGG-16'
        modelPath = fullfile(vl_rootnn,'pretrained', 'imagenet-MPN-COV-VGG-16.mat') ;
    otherwise
end
display_time = 1;
gpus = [1];   % choose GPU id, not support Multi GPUs currently.
numGpus = numel(gpus) ;
imdb = load('imagenet_imdb/imdb.mat');
net = load(modelPath);
if isfield(net,'net')
    net = net.net;
end
%% initialize Network and GPU
isDag = isfield(net, 'params') ;
if isDag 
  opts.networkType = 'dagnn' ;
  net = dagnn.DagNN.loadobj(net) ;
  trainfn = @cnn_train_dag ;
  % Drop existing loss layers
  drop = arrayfun(@(x) isa(x.block,'dagnn.Loss'), net.layers) ;
  for n = {net.layers(drop).name}
    net.removeLayer(n) ;
  end
  % Extract raw predictions from softmax
  sftmx = arrayfun(@(x) isa(x.block,'dagnn.SoftMax'), net.layers) ;
  predVar = 'prediction' ;
  for n = {net.layers(sftmx).name}
    % check if output
    l = net.getLayerIndex(n) ;
    v = net.getVarIndex(net.layers(l).outputs{1}) ;
    if net.vars(v).fanout == 0
      % remove this layer and update prediction variable
      predVar = net.layers(l).inputs{1} ;
      net.removeLayer(n) ;
    end
  end
  % Add custom objective and loss layers on top of raw predictions
  net.addLayer('objective', dagnn.Loss('loss', 'softmaxlog'), ...
               {predVar,'label'}, 'objective') ;
  net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
               {predVar,'label'}, 'top1err') ;
  net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                     'opts', {'topK',5}), ...
               {predVar,'label'}, 'top5err') ;
  % Make sure that the input is called 'input'
  v = net.getVarIndex('data') ;
  if ~isnan(v)
    net.renameVar('data', 'input') ;
  end
  % Swtich to test mode
  net.mode = 'test' ;
  prediction_id = net.getVarIndex('prediction');
  net.vars(prediction_id).precious = 1; 
else
    net = vl_simplenn_tidy(net) ;
    vl_simplenn_display(net);
end
if numGpus > 0
fprintf('%s: resetting GPU\n', mfilename) ;
  disp('Clearing mex files') ;
  clear mex ;
  clear vl_tmove vl_imreadjpeg ;
  if numGpus == 1
    disp(gpuDevice(gpus)) ;
    if isDag
      net.move('gpu') ;
    else
      net = vl_simplenn_move(net, 'gpu') ;
    end
  else
    error('Not support Multi GPUs!');
  end
  opts.useGpu = true;
else
    if isDag
      net.move('cpu');
    else
      net = vl_simplenn_move(net, 'cpu') ;
    end
    opts.useGpu = false;
end
switch evaluate_mode
    case 'val'
        im_id = find(imdb.images.set == 2);
    case 'test'
        im_id = find(imdb.images.set == 3);
end

%% Prediction
t = 0;
error_state = [0 0 0 0 0];
start = 1;
custom_end = length(im_id);  
for i = start : custom_end
    tic;
    imagePath = fullfile(dataDir,imdb.images.name(im_id(i)));
    label  = imdb.images.label(im_id(i));
    data = getImageBatch_custom(opts,imagePath);
    label = repmat(label,[1 size(data,4)]);
    if isDag
        score = predict_dagnn(data,label,net);
    else
        score = predict_simplenn(data,label,net);
    end
    error_current_state = compute_error(score,label(1));
    error_state(1:4)  = error_state(1:4) + error_current_state;error_state(end) = i;
    if(i/display_time == fix(i/display_time))
         fprintf('val %d/%d(%4.2fHz):',i,custom_end,display_time/t);
         fprintf('top1err is %f,top5err is %f,top1err_softmax is %f,top5err_softmax is %f\n',...
                            1-error_state(1)/i,1-error_state(2)/i,1-error_state(3)/i,1-error_state(4)/i);
         t = 0;
    end
    t = t + toc;
end
