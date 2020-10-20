# The SE(2) group convolutional neural network library
This repository contains the code for SE(2) group convolutional networks. The theory is described in the paper:

Bekkers, E., Lafarge, M., Veta, M., Eppenhof, K., Pluim, J., Duits, R.: Roto-translation covariant
convolutional networks for medical image analysis. Accepted at MICCAI 2018, arXiv preprint arXiv:1804.03393 (2018). Available at: https://arxiv.org/abs/1804.03393

It was subsequently modified from the version at https://github.com/tueimage/se2cnn by R. James Cotton to this version.

# Contents
This repository contains the following folders:
* **se2cnn** - the main python library for se2 group convolutional networks. 

# Some notes about the proposed SE(2) CNN layers

The library provides code for building group equivariant convolutional networks for the case when the group G is SE(2), the group of planar roto-translations. In this case the lifting layer (se2cnn.layers.z2_se2n) probes the 2D input with rotated and translated versions of the convolution kernels. The data is thus lifted to the space of positions and orientations. In order to make the following layers (se2cnn.layers.se2n_se2n) equivariant with respect to rotations and translations of the input these layers are defined in terms of the left-regular representation of SE(2) on SE(2)-images. The kernels used in the layers are trained to recognize the (joint) activations of positions and orientations relative to each other. Finally, in order to make the entire network invariant to certain transformations one can decide to apply max-pooling (tf.reduce_max) over sub-groups. In our case we might for example do a maximum projection over the sub-group of rotations in order to make the network locally rotation invariant.

