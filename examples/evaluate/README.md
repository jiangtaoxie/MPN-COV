## Demo code on evaluating a MPN-COV ConvNet
   
This demo contains the code that can evaluate a model for inference with either center crop or ten crop. Note that the main difference from the code provided by MatConvNet is that we RESIZE an image using Matlab [imresize function](http://cn.mathworks.com/help/images/ref/imresize.html); the performance will decrease slightly if the resize function of MatConvNet is used. 

### Usage

#### Tutorial

1. Put the models you download to `matconvnet_root_dir/pretrained`.

2. Modify the `model_type` and `opts.method` in `evaluate.m` to get the results.

3. run `evaluate.m`.

#### Function description



1. `evaluate.m`: main script.

```matlab
   dataset = 'ILSVRC2012';
   model_type = 'model'; % 'MPN-COV-AlexNet','MPN-COV-VGG-M' 'MPN-COV-VGG-16'
                         %  'MPN-COV-ResNet-50' 'MPN-COV-ResNet-101'
   evaluate_mode; % 'val' for validation dataset
   
   opts.scale = 256 % rescale the image's short-side to 256
   opts.method = '1-crop' % '1-crop' or '10-crop'
   opts.crop_size = 224;  % 227 for 'MPN-COV-AlexNet',224 for others 
   opts.averageImage = [124.2441 116.5862 103.4211]; % import from our models
```

2. `predict_dagnn.m`: Make inference for an input image under DagNN framework.

3. `predict_simplenn.m`: Make inference for an input image under SimpleNN framework.

### Change log








