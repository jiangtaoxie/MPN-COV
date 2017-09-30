## Demo code on training MPN-COV ConvNets from scratch on ImageNet
   
This demo contains the code which implements training our MPN-COV ConvNets from scratch on ImageNet 2012.

### Classification results (top-1/top-5 error rates: %) on ImageNet 2012 validation set

 Network            |224x224<br />1-crop|224x224<br />10-crop
 ---|:---:|:---:
 MPN-COV-ResNet-50 |22.27/6.35         |21.16/5.58 
 MPN-COV-ResNet-101 |21.17/5.70        |19.71/5.01
 MPN-COV-AlexNet |38.37/17.14          |34.97/14.60
 MPN-COV-VGG-M   |34.63/14.64          |31.81/12.52
 MPN-COV-VGG-16 |26.55/8.94         |24.68/7.75

### Usage

#### Tutorial

1. Download the ImageNet 2012 [dataset](http://image-net.org/download.php).

2. Put the dataset you downloaded into `matconvnet_root_dir/data/`.

3. Modify the `opts.modelType` in `MPN_COV_main.m` to your needs.

4. run `MPN_COV_main.m`.

#### Function descriptions

1. `MPN_COV_main.m`: The main function.

2. `MPN_COV_init_resnet.m`: Initialize, for training from scratch, the proposed MPN-COV ConvNets under ResNet architecture using DagNN, and set the hyper-parameters involved in training.

3. `MPN_COV_init_simplenn.m`: Initialize, for training from scratch, the proposed MPN-COV ConvNets under the architectures of AlexNet, VGG-M and VGG-VD using SimpleNN, and set the hyper-parameters involved in training.








