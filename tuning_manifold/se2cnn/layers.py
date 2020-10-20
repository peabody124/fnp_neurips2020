# -*- coding: utf-8 -*-
"""
Adopted from below for Tensorflow2 by R. James Cotton.

se2cnn/layers.py

Implementation of tensorflow layers for operations in SE2N.
Details in MICCAI 2018 paper: "Roto-Translation Covariant Convolutional Networks for Medical Image Analysis".

Released in June 2018
@author: EJ Bekkers, Eindhoven University of Technology, The Netherlands
@author: MW Lafarge, Eindhoven University of Technology, The Netherlands
________________________________________________________________________

Copyright 2018 Erik J Bekkers and Maxime W Lafarge, Eindhoven University
of Technology, the Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
________________________________________________________________________
"""

import tensorflow as tf
from .rotation_matrix import *


def rotate_lifting_kernels(kernel, orientations_nb, periodicity=2 * np.pi, diskMask=True):
    """ Rotates the set of 2D lifting kernels.

        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [Height, Width, ChannelsIN, ChannelsOUT]
            - orientations_nb, an integer specifying the number of rotations

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially

        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, Height, Width, ChannelsIN, ChannelsOUT]
    """

    # Unpack the shape of the input kernel
    kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
    #print("Z2-SE2N BASE KERNEL SHAPE:", kernel.get_shape())  # Debug

    # Flatten the baseline kernel
    # Resulting shape: [kernelSizeH*kernelSizeW, channelsIN*channelsOUT]
    kernel_flat = tf.reshape(kernel, [kernelSizeH * kernelSizeW, channelsIN * channelsOUT])

    # Generate a set of rotated kernels via rotation matrix multiplication
    # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
    # Result: The non-zero indices and weights of the rotation matrix
    idx, vals = MultiRotationOperatorMatrixSparse(
        [kernelSizeH, kernelSizeW],
        orientations_nb,
        periodicity=periodicity,
        diskMask=diskMask)

    # Sparse rotation matrix
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # kernelSizeH*kernelSizeW]
    rotOp_matrix = tf.SparseTensor(
        idx, vals,
        [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])

    # Matrix multiplication
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,
    # channelsIN*channelsOUT]
    set_of_rotated_kernels = tf.sparse.sparse_dense_matmul(tf.cast(rotOp_matrix, kernel_flat.dtype),
                                                           kernel_flat)

    # Reshaping
    # Resulting shape: [nbOrientations, kernelSizeH, kernelSizeW, channelsIN,
    # channelsOUT]
    set_of_rotated_kernels = tf.reshape(
        set_of_rotated_kernels, [orientations_nb, kernelSizeH, kernelSizeW, channelsIN, channelsOUT])

    return set_of_rotated_kernels


def rotate_gconv_kernels(kernel, periodicity=2 * np.pi, diskMask=True):
    """ Rotates the set of SE2 kernels.
        Rotation of SE2 kernels involves planar rotations and a shift in orientation,
        see e.g. the left-regular representation L_g of the roto-translation group on SE(2) images,
        (Eq. 3) of the MICCAI 2018 paper.

        INPUT:
            - kernel, a tensor flow tensor with expected shape:
                [Height, Width, nbOrientations, ChannelsIN, ChannelsOUT]

        INPUT (optional):
            - periodicity, rotate in total over 2*np.pi or np.pi
            - disk_mask, True or False, specifying whether or not to mask the kernels spatially

        OUTPUT:
            - set_of_rotated_kernels, a tensorflow tensor with dimensions:
                [nbOrientations, Height, Width, nbOrientations, ChannelsIN, ChannelsOUT]
              I.e., for each rotation angle a rotated (shift-twisted) version of the input kernel.
    """

    # Rotation of an SE2 kernel consists of two parts:
    # PART 1. Planar rotation
    # PART 2. A shift in theta direction

    # Unpack the shape of the input kernel
    kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(
        int, kernel.shape)

    # PART 1 (planar rotation)
    # Flatten the baseline kernel
    # Resulting shape: [kernelSizeH*kernelSizeW,orientations_nb*channelsIN*channelsOUT]
    #
    kernel_flat = tf.reshape(
        kernel, [kernelSizeH * kernelSizeW, orientations_nb * channelsIN * channelsOUT])

    # Generate a set of rotated kernels via rotation matrix multiplication
    # For efficiency purpose, the rotation matrix is implemented as a sparse matrix object
    # Result: The non-zero indices and weights of the rotation matrix
    idx, vals = MultiRotationOperatorMatrixSparse(
        [kernelSizeH, kernelSizeW],
        orientations_nb,
        periodicity=periodicity,
        diskMask=diskMask)

    # The corresponding sparse rotation matrix
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW,kernelSizeH*kernelSizeW]
    #
    rotOp_matrix = tf.SparseTensor(
        idx, vals,
        [orientations_nb * kernelSizeH * kernelSizeW, kernelSizeH * kernelSizeW])

    # Matrix multiplication (each 2D plane is now rotated)
    # Resulting shape: [nbOrientations*kernelSizeH*kernelSizeW, orientations_nb*channelsIN*channelsOUT]
    #
    kernels_planar_rotated = tf.sparse.sparse_dense_matmul(
        tf.cast(rotOp_matrix, kernel_flat.dtype), kernel_flat)
    kernels_planar_rotated = tf.reshape(
        kernels_planar_rotated, [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT])

    # PART 2 (shift in theta direction)
    set_of_rotated_kernels = [None] * orientations_nb
    for orientation in range(orientations_nb):
        # [kernelSizeH,kernelSizeW,orientations_nb,channelsIN,channelsOUT]
        kernels_temp = kernels_planar_rotated[orientation]
        # [kernelSizeH,kernelSizeW,channelsIN,channelsOUT,orientations_nb]
        kernels_temp = tf.transpose(a=kernels_temp, perm=[0, 1, 3, 4, 2])
        # [kernelSizeH*kernelSizeW*channelsIN*channelsOUT*orientations_nb]
        kernels_temp = tf.reshape(
            kernels_temp, [kernelSizeH * kernelSizeW * channelsIN * channelsOUT, orientations_nb])
        # Roll along the orientation axis
        roll_matrix = tf.constant(
            np.roll(np.identity(orientations_nb), orientation, axis=1), dtype=tf.float32)
        kernels_temp = tf.matmul(kernels_temp, roll_matrix)
        kernels_temp = tf.reshape(
            kernels_temp, [kernelSizeH, kernelSizeW, channelsIN, channelsOUT, orientations_nb])  # [Nx,Ny,Nin,Nout,Ntheta]
        kernels_temp = tf.transpose(a=kernels_temp, perm=[0, 1, 4, 2, 3])
        set_of_rotated_kernels[orientation] = kernels_temp

    return tf.stack(set_of_rotated_kernels)


class SE2Lifting(tf.keras.layers.Conv2D):

    def __init__(self, nb_orientations=4, disk_mask=True, periodicity=2 * np.pi, *args, **kwargs):
        self.nb_orientations = nb_orientations
        self.periodicity = periodicity
        self.disk_mask = disk_mask
        super(SE2Lifting, self).__init__(*args, **kwargs)

    def compute_output_shape(self, input_shapes, *args, **kwargs):
        s = super(SE2Lifting, self).compute_output_shape(input_shapes, *args, **kwargs)
        output_shape = [*s[:-1], self.nb_orientations, s[-1]]
        return output_shape

    def build(self, input_shapes):
        super(SE2Lifting, self).build(input_shapes)

    def call(self, inputs):
        kernel = self.weights[0]
        kernel_stack = rotate_lifting_kernels(kernel,
                                              periodicity=self.periodicity,
                                              orientations_nb=self.nb_orientations,
                                              diskMask=self.disk_mask)

        kernels_as_if_2D = tf.transpose(a=kernel_stack, perm=[1, 2, 3, 0, 4])
        kernelSizeH, kernelSizeW, channelsIN, channelsOUT = map(int, kernel.shape)
        kernels_as_if_2D = tf.reshape(kernels_as_if_2D,
                                      [kernelSizeH, kernelSizeW, channelsIN, self.nb_orientations * channelsOUT])

        # Perform the 2D convolution
        layer_output = tf.nn.conv2d(
            inputs,
            filters=kernels_as_if_2D,
            strides=[1, 1, 1, 1],
            padding='VALID' if self.padding == 'valid' else 'SAME')

        # For now just leave channels and orientations mixed together
        # Reshape to an SE2 image (split the orientation and channelsOUT axis)
        # Note: the batch size is unknown, hence this dimension needs to be
        # obtained using the tensorflow function tf.shape, for the other
        # dimensions we keep using tensor.shape since this allows us to keep track
        # of the actual shapes (otherwise the shapes get convert to
        # "Dimensions(None)").
        layer_output = tf.reshape(layer_output,
                                  [tf.shape(input=layer_output)[0], int(layer_output.shape[1]),
                                  int(layer_output.shape[2]), self.nb_orientations, channelsOUT])

        if self.use_bias:
            layer_output = layer_output + tf.reshape(self.weights[1], [1] * 4 + [self.filters])

        return layer_output


class SE2GroupConv(tf.keras.layers.Conv3D):

    def __init__(self, nb_orientations=4, disk_mask=True, periodicity=2 * np.pi,
                 kernel_size=(5, 5), *args, **kwargs):
        self.nb_orientations = nb_orientations
        self.periodicity = periodicity
        self.disk_mask = disk_mask

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, self.nb_orientations)
        else:
            kernel_size = (*kernel_size, self.nb_orientations)
        super(SE2GroupConv, self).__init__(kernel_size=kernel_size, *args, **kwargs)

    def compute_output_shape(self, input_shapes, *args, **kwargs):
        s = super(SE2GroupConv, self).compute_output_shape(input_shapes, *args, **kwargs)
        return [*s[:-2], self.nb_orientations, s[-1]]

    def call(self, inputs):
        kernel = self.weights[0]

        # Kernel dimensions
        kernelSizeH, kernelSizeW, orientations_nb, channelsIN, channelsOUT = map(
            int, kernel.shape)

        # Preparation for group convolutions
        # Precompute a rotated stack of se2 kernels
        # With shape: [orientations_nb, kernelSizeH, kernelSizeW, orientations_nb,
        # channelsIN, channelsOUT]
        kernel_stack = rotate_gconv_kernels(kernel, self.periodicity, self.disk_mask)

        # Group convolutions are done by integrating over [x,y,theta,input-channels] for each translation and rotation of the kernel
        # We compute this integral by doing standard 2D convolutions (translation part) for each rotated version of the kernel (rotation part)
        # In order to efficiently do this we use 2D convolutions where the theta
        # and input-channel axes are merged (thus treating the SE2 image as a 2D
        # feature map)

        # Prepare the input tensor (merge the orientation and channel axis) for
        # the 2D convolutions:
        input_tensor_as_if_2D = tf.reshape(
            inputs,
            [tf.shape(inputs)[0], int(inputs.shape[1]),
             int(inputs.shape[2]), self.nb_orientations * channelsIN])

        # Reshape the kernels for 2D convolutions (orientation+channelsIN axis are
        # merged, rotation+channelsOUT axis are merged)
        kernels_as_if_2D = tf.transpose(a=kernel_stack, perm=[1, 2, 3, 4, 0, 5])
        kernels_as_if_2D = tf.reshape(
            kernels_as_if_2D, [kernelSizeH, kernelSizeW, orientations_nb * channelsIN, orientations_nb * channelsOUT])

        # Perform the 2D convolutions
        layer_output = tf.nn.conv2d(
            input=input_tensor_as_if_2D,
            filters=kernels_as_if_2D,
            strides=[1, 1, 1, 1],
            padding='VALID' if self.padding == 'valid' else 'SAME')

        # Reshape into an SE2 image (split the orientation and channelsOUT axis)
        layer_output = tf.reshape(
            layer_output,
            [tf.shape(input=layer_output)[0], int(layer_output.shape[1]), int(layer_output.shape[2]), orientations_nb,
             channelsOUT])

        if self.use_bias:
            layer_output = layer_output + tf.reshape(self.weights[1], [1] * 4 + [self.filters])

        return layer_output


class SE2GroupSeparableConv(tf.keras.layers.Layer):

    def __init__(self, nb_orientations=4, disk_mask=True, periodicity=2 * np.pi,
                 filters=5, kernel_size=(5, 5), densenet=True, bottleneck_factor=4, *args, **kwargs):

        kwargs1 = dict((k, v) for k, v in kwargs.items() if k == 'name')
        kwargs = dict((k, v) for k, v in kwargs.items() if k != 'name')

        self.densenet=densenet
        super(SE2GroupSeparableConv, self).__init__(**kwargs1)

        # note these are applied BEFORE the layers, consistent with
        # the Densenet architecture
        self.bottleneck_batchnorm = tf.keras.layers.BatchNormalization()
        self.spatial_batchnorm = tf.keras.layers.BatchNormalization()

        self.bottleneck = SE2GroupConv(filters=bottleneck_factor * filters, nb_orientations=nb_orientations, kernel_size=(1, 1),
                                       use_bias=False, name=self.name + "-B")
        self.spatial = SE2GroupConv(filters=filters, nb_orientations=nb_orientations, kernel_size=kernel_size,
                                    disk_mask=disk_mask, periodicity=periodicity, name=self.name + "-G",
                                    use_bias=False,
                                    *args, **kwargs)

    def compute_output_shape(self, input_shapes, *args, **kwargs):

        s = self.spatial.compute_output_shape(input_shapes, *args, **kwargs)
        if self.densenet:
            return tf.TensorShape([*s[:-1], s[-1] + input_shapes[-1]])
        return s

    def call(self, inputs):

        bn_inputs = self.bottleneck_batchnorm(inputs)
        bn_inputs = tf.nn.elu(bn_inputs)
        b_inputs = self.bottleneck(bn_inputs)

        b_inputs = self.spatial_batchnorm(b_inputs)
        b_inputs = tf.nn.elu(b_inputs)
        outputs = self.spatial(b_inputs)

        if self.densenet:
            outputs = tf.concat([inputs, outputs], axis=-1)

        return outputs


class SE2ProjCat(tf.keras.layers.Layer):

    def init(self, *args, **kwargs):
        super(SE2ProjCat, self).__init__(*args, **kwargs)
        self._input_shape = None

    def build(self, input_shape):
        super(SE2ProjCat, self).build(input_shape)
        self._input_shape = input_shape

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], input_shape[3]*input_shape[4]])

    def call(self, x):
        return tf.reshape(x, [-1, self._input_shape[1], self._input_shape[2], self._input_shape[3] * self._input_shape[4]])


SE2ProjMax = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-2))