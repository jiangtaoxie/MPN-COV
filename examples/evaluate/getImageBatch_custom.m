function data = getImageBatch_custom(opts,imagePaths)
% GETIMAGEBATCH_CUSTOM  Load and jitter a batch of images
% Modified by Jiangtao Xie

useGpu        = opts.useGpu;    
method        = opts.method;
scale         = opts.scale;
crop_size     = opts.crop_size;       %  crop size
averageImage  = opts.averageImage;

data = vl_imreadjpeg(imagePaths);
data = cell2mat(data(1)); % batch size == 1
if size(data,3) < 3  % Convert Single Channel Picture to three Channels
    data = repmat(data,[1 1 3]);
end
I = data;data = [];
datasize = [size(I,1) size(I,2)];
scale_rate = scale / min(datasize) ;
I = imresize(I,scale_rate);
I(:,:,1) = I(:,:,1) - averageImage(1);
I(:,:,2) = I(:,:,2) - averageImage(2);
I(:,:,3) = I(:,:,3) - averageImage(3);
H = size(I,1);
W = size(I,2);
crop = crop_size - 1;
left_top     = [1 1 crop crop];
right_top    = [W - crop 1 crop crop];
left_bottom  = [1 H - crop crop crop];
right_bottom = [W - crop H - crop crop crop];
center       = [W/2 - crop/2 H/2 - crop/2 crop crop];
switch method
    case '1-crop'
        data(:,:,:,1) = imcrop(I,center);
        data = single(data);
    case '10-crop'
        data(:,:,:,1) = imcrop(I,left_top);
        data(:,:,:,2) = imcrop(I,right_top);
        data(:,:,:,3) = imcrop(I,left_bottom);
        data(:,:,:,4) = imcrop(I,right_bottom);
        data(:,:,:,5) = imcrop(I,center);
        for i=1:5
            data(:,:,:,i+5) = flip(data(:,:,:,i),2);
        end
        data = single(data);
end
if useGpu  % Convert Cpu data to Gpu data
   data = gpuArray(data);
end   
end
