function net = MPN_COV_init_resnet_ft(opts, varargin)
%MPN_COV_INIT_RESNET_FT:
% Load the ResNet-MPN-COV model and replace the fc layer
% to a new one
% modified by Peihua Li  for MPN-COV

opts.model = [];
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;


model_path = fullfile(vl_rootnn,'pretrained', ['imagenet-' opts.modelType '-dag.mat']); 
net = load(model_path);
if isfield(net, 'net') ;
   net = net.net ;
end
% Cannot use isa('dagnn.DagNN') because it is not an object yet
isDag = isfield(net, 'params') ;
if isDag
   opts.networkType = 'dagnn' ;
   net = dagnn.DagNN.loadobj(net) ;    
else
   error('In  MPN_COV_init_resnet: invalid  pre-trained model!');
end

switch opts.regu_method
    case  'power'
        keep_idx        = net.getLayerIndex('prediction') ;
        lastAdded.var   = net.layers(keep_idx).inputs;
        lastAdded.depth = net.layers(keep_idx).block.size(3);
        remove_names    = {net.layers(keep_idx : end).name};
        net.removeLayer(remove_names);
        name = 'prediction' ;
        net.addLayer(name, ...
                     dagnn.Conv('size', [1, 1, lastAdded.depth, numel(opts.classNames)]), ...
                     lastAdded.var, ...
                     name, ...
                     {[name, '_f'], [name, '_b']}) ;
        lastAdded.var = name;   
    otherwise
        error('In  MPN_COV_init_resnet: pooling method NOT  supported!');
        %add your custom regu_method here
end
   
net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [448 448 3] ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 448 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;

net.meta.augmentation.keepAspect = true;
net.meta.augmentation.jitterLocation = false ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
 net.meta.augmentation.jitterScale  = [1, 1] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

lr = 0.1 * ones(1,100) * (10 ^ -1.5);
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize   = 120;   
net.meta.trainOpts.numSubBatches = 6;  
net.meta.trainOpts.weightDecay = 0.0001 ;

% Randomly init parameters of layers whose index is from keep_idx to end
for l = keep_idx : numel(net.layers)
  p  = net.getParamIndex(net.layers(l).params);
  params = net.layers(l).block.initParams() ;
  switch net.device
    case 'cpu'
      params = cellfun(@gather, params, 'UniformOutput', false) ;
    case 'gpu'
      params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
  end
  [net.params(p).value] = deal(params{:}) ; 
end

for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.BatchNorm')
       k = net.getParamIndex(net.layers(l).params{3}) ;
       net.params(k).learningRate = 0.3 ;
       net.params(k).epsilon = 1e-5 ;
    end
end

%Increase the learning rate of the FC layer new added 
for l = keep_idx : numel(net.layers)
    p = net.getParamIndex(net.layers(l).params) ;
    for k = 1 : numel(p)
        net.params(p(k)).learningRate = net.params(p(k)).learningRate * 5;
    end
end

% Make sure that the input is called 'input'
v = net.getVarIndex('data') ;
if ~isnan(v)
net.renameVar('data', 'input') ;
end


end
