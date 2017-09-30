function [net, info] = MPN_COV_main_ft(varargin)
%MPN_COV_MAIN_FT
%  This demo demonstrates using the ImageNet pre-trained model
%  on fine-grained datasets.
%  Support Resnet architecture with our MPN_COV method
% modified by Peihua Li  for MPN-COV

run(fullfile(fileparts(mfilename('fullpath')),   '..', '..', 'matlab', 'vl_setupnn.m')) ;

dataset_name = 'cars';  % 'CUB_200_2011', 'fgvc-aircraft-2013b' , 'cars'
opts.dataDir = fullfile(vl_rootnn, 'data', dataset_name) ;
opts.modelType =  'MPN-COV-ResNet-50';          %  'MPN-COV-ResNet-50'  'MPN-COV-ResNet-50'
           

opts.dim = 64;                                  
opts.regu_method = 'power';       
opts.epsilon = 0;                         
opts.alpha = 0.5; 
opts.batchNormalization = true ;    

opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

if  any(strncmp(opts.modelType, {'MPN-COV-ResNet'}, 7)) opts.networkType = 'dagnn'; end

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bn'] ; end
sfx = ['MPN-COV-' sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', [dataset_name '-' sfx]) ;

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12;  
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

fprintf('Current experimental dir is: %s\n', opts.expDir);

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
    switch dataset_name
        case 'CUB_200_2011'
            imdb = cub_get_database(opts.dataDir,false,false);
        case 'fgvc-aircraft-2013b'
            imdb = aircraft_get_database(opts.dataDir,'variant');
        case 'cars'
            imdb = cars_get_database(opts.dataDir,false,false);
        otherwise
             error('Unsupported database ''%s''', dataset_name) ;
    end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set(find(imdb.images.set == 2)) = 1; % Merge train and val
imdb.images.set(find(imdb.images.set == 3)) = 2; 
% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  if numel(train) < 5e5
      images = fullfile(imdb.imageDir, imdb.images.name(train(1:5:end))) ;
  else
      images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  end
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [256 256], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
    if any(strncmp(opts.modelType,  {'MPN-COV-ResNet'}, 7))
          net = MPN_COV_init_resnet_ft(opts, 'averageImage', rgbMean, 'colorDeviation', rgbDeviation, ...
                                         'classNames', imdb.classes.name) ;
          opts.networkType = 'dagnn' ;
    else 
          error('Error--the architecture is not supported yet! \n');
    end
else
  net = opts.network ;
  opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu,...
  'keepAspect',meta.augmentation.keepAspect) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;  
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end

