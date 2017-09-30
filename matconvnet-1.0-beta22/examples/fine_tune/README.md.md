## Demo code on finetuning with pretrained MPN-COV ConvNets
   
To show the potential of our method, we make additional experiments on Fine-grained Visual Recognition benchmarks. Specifically, we perform finetuning on three popular benchmarks using MPN-COV ConvNets pre-trained on ImageNet. The results are shown as follows.


### Fine-grained classification results(top-1 accuracy rates, %)

Network     |[Birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |[Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) |[Aircrafts](http://www.robots.ox.ac.uk/~vgg/data/oid/) 
---|:---:|:---:|:---:
MPN-COV-ResNet-50        |**87.6** |**92.9** |**90.5**
B-CNN(VGG-M+VGG-D)[[1]](#1-t-y-lin-a-roychowdhury-and-s-maji-bilinear-cnn-models-for-fine-grained-visual-recognition-ieee-tpami-2017)    |84.1 |91.3 |86.6
Improved B-CNN(VGG-D)[[2]](#2-t-y-lin-and-s-majiimproved-bilinear-pooling-with-cnns-in-bmvc-2017) |85.8 |92.0 |88.5

  - The results are obtained by finetuning the MPN-COV ConvNets pretrained on ImageNet on the target fine-grained benchmarks; neither bounding boxes nor part information are used anywhere. The code to reproduce the results is released [here](https://github.com/jiangtaoxie/demo/tree/master/fine-tune).
  - We compare our results with closely related method, i.e., Bilinear CNN (B-CNN)[[1]](#1-t-y-lin-a-roychowdhury-and-s-maji-bilinear-cnn-models-for-fine-grained-visual-recognition-ieee-tpami-2017) and the improved B-CNN[[2]](#2-t-y-lin-and-s-majiimproved-bilinear-pooling-with-cnns-in-bmvc-2017) .


### Usage

#### Function descriptions

1. `MPN_COV_main_ft.m`: The main function of this finetuning demo.

```matlab
   dataset_name = 'name'; % Birds: 'CUB_200_2011' Aircrafts:'fgvc-aircraft-2013b'
                          % Cars: 'cars'
   opts.modelType = 'model'; % pre-trained model: 'MPN-COV-ResNet-50' 
                             % or 'MPN-COV-ResNet-101'
   opts.alpha = '0.5' % 0.5 is better
```

2. `MPN_COV_init_resnet_ft.m`: Initialize, for finetuning, with pre-trained MPN-COV ConvNets under ResNet architecture using DagNN, and set the hyper-parameters involved in training.

3. `***_get_database.m` : These functions are duplicated from [bcnn-package](https://bitbucket.org/tsungyu/bcnn.git) to creat imdb for different databases.
#### Reproduce our Results
The experimental settings are presented in the following for the three benchmarks, i.e., Birds, Cars and Aircrafts, respectively. Note that, except the following parameters, all the other parameters are the same as the settings in the file `MPN_COV_init_resnet_ft.m`
##### Birds

```matlab
%% MPN_COV_init_resnet_ft.m

net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 448 ;
net.meta.augmentation.keepAspect = true;
lr = 0.1 * ones(1,100) * (10 ^ -1.5);
```
#### Cars

```matlab
%% MPN_COV_init_resnet_ft.m

net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 448;
net.meta.augmentation.keepAspect = true;
lr = 0.1 * ones(1,150) * (10 ^ -1.5);
```
#### Aircrafts

```matlab
%% MPN_COV_init_resnet_ft.m

net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 512 ;
net.meta.augmentation.keepAspect = false;
lr = 0.1 * ones(1,100) * (10 ^ -1.5);
```


### References

##### [1] T.-Y. Lin, A. RoyChowdhury, and S. Maji. Bilinear CNN models for fine-grained visual recognition. IEEE TPAMI, 2017.

##### [2] T.-Y. Lin and S. Maji.Improved Bilinear Pooling with CNNs. In BMVC, 2017.




