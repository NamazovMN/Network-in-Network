# Network-in-Network

_The project is about Network in Network model. The project has been done based on the paper of **Network in Network** which was published by **Min Lin**, **Qiang Chen** and **Shuicheng Yan** in 2014._

## Main Idea

_It is a novel deep network structure which uses mlpconv layers instead of conventional convolutional layers. On the other hand, Global Average Pooling layer was used on the top of the Network instead of the dense layers. This Global Average Pooling layer usage prevents overfitting can be caused because of dense layers._

_Model has been trained on **CIFAR 10**, **CIFAR 100**, **MNIST**, **SVHN** datasets._

## Code files

### data_preparator.py

_This file includes class which is used for downloading and preparing the data is used for training and validation. Additionally, as it was mentioned in the related paper, **ZCA whitening** and **Global Contrast Normalization** were applied to the images in this file, too._

### model.py

_Network in Network model was implemented in this file which consists of stacked 3 mlpconv layers and Global Average Pooling Layer. 

**Note:** 

_**Please not be confused if you see 9 convolutional layers. Each 3 of them correspond to the 1 mlpconv layer. Because of mathematical properties of the mlpconv layers they can be used as convolutional layer which has 1x1 kernel size.**_

### train.py

_This code file includes train phase of the model. Additionally, it includes also validation phase, too. After each epoch validation data is used in order to observe how the model acts. CIFAR 10, CIFAR 100 have been trained 200 epochs, MNIST and SVHN were trained 50 epochs because of the dataset properties._

### main.py

_Here we use all previous classes "remotely" and start also to train the model. According to the gathered data, graphs are generated and saved to the results folder._


_**BEST REGARDS,**_

_**Mahammad Namazov**_
