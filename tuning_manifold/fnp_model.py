"""
Deterministic implementation of the K-shot neural response prediction.

The goal is to learn two core mappings. The first is a convolutional
network on images to the relevant features of stimuli:
  g(x)

The second is an embedding of a set of stimulus and responses from a
a neuron
  ax, ay, w, b = f( {g(x_i), r_i) )

This is used to predict the responses to new stimuli

  ELU(dot(outer(softmax(ax), softmax(ay), w), g(x)) + b) + 1

We have to pay attention to the axis details here, too, as the
goal is to predict over sets with multiple neurons too (with
the same stimulus). For stimulus the dimension is

  set / batch X image_x X image_y X channels (1)

For responses the dimension is

  set X neuron_i

All the vector parameters describing each cell (ax, ay, w, b) are
shaped

  set X neuron_i X feature_length

So when expanding the dot product in the above equation the last
feature_length dimension will actually be expanded into three
dimensions so we will essentially be working in a space of

  set X neuron_i X image_x X image_y X conv_features

Where the g(x) will have the neuron_i dimension inserted and
broadcast along the neuron_i axis.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from .util import interpolate_bilinear
from .se2cnn.layers import SE2Lifting, SE2GroupConv, SE2ProjCat, SE2ProjMax, SE2GroupSeparableConv

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class LaplaceL2Regularizer(tf.keras.regularizers.L1L2):

    def __init__(self, sigma_L=5.0, bias_L=1.0, **kwargs):
        super(LaplaceL2Regularizer, self).__init__(**kwargs)
        self.laplace_filter = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tf.float32)[..., None, None]
        self.sigma_L = tf.constant(sigma_L, dtype=tf.float32)
        self.bias_L = tf.constant(bias_L, dtype=tf.float32)

    def __call__(self, x):
        nx, ny, ch_in, ch_out = x.shape
        x_unwrapped = tf.reshape(x, [1, nx, ny, -1])
        x_unwrapped = tf.transpose(x_unwrapped, perm=[3, 1, 2, 0])

        X, Y = tf.meshgrid(tf.linspace(-(nx - 1) / 2, (nx - 1) / 2, nx), tf.linspace(-(ny - 1) / 2, (ny - 1) / 2, ny))
        weights = self.bias_L - tf.exp(-(tf.square(X) + tf.square(Y)) / 2 / tf.square(self.sigma_L))

        filtered_weights = tf.nn.conv2d(x_unwrapped, self.laplace_filter,
                                        padding='SAME', strides=1)

        return super(LaplaceL2Regularizer, self).__call__(filtered_weights * weights[None, ..., None])


class DeepSetSimple(tfkl.Layer):
    """DeepSet layer that computes the set collapse of all other elements"""

    def __init__(self, cummulative, axis=1, **kwargs):
        super(DeepSetSimple, self).__init__(**kwargs)
        self.cummulative = cummulative
        self.axis = axis

    def call(self, input):

        x = tf.cast(input, tf.float64)

        N = input.shape[self.axis]

        if not self.cummulative:
            # Note that we remove each element from the set dimension from the
            # sum so the output of this is actually INDEPENDENT of the input
            # element and only dependent on the other elements
            x = (tf.math.reduce_sum(x, axis=self.axis, keepdims=True) - x)
            N = N - 1
        else:
            x = tf.math.cumsum(x, axis=self.axis, exclusive=True)
            Ncount = tf.linspace(1e-5, tf.cast(N-1, tf.float32), N)
            count_shape = [1] * len(input.shape)
            count_shape[self.axis] = N
            N = tf.reshape(Ncount, count_shape)

        # normalize by the number of elements summed
        x = tf.cast(x, input.dtype) / tf.cast(N, input.dtype)
        #x = tf.cast(x, input.dtype)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class ConvResponseConcatenate(tfkl.Layer):
    def __init__(self, sta_feature=True, **kwargs):
        self.sta_feature = sta_feature
        self.response_shape = None
        self.conv_shape = None
        super(ConvResponseConcatenate, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.response_shape, self.conv_shape = input_shapes

    def get_config(self):
        base_config = super(ConvResponseConcatenate, self).get_config()
        base_config.update({'sta_feature': self.sta_feature})
        return base_config

    def compute_output_shape(self, input_shapes):
        response_shape, conv_shape = [x for x in input_shapes]
        bs, stimuli, N = response_shape
        bs, stimuli, ny, nx, nf = conv_shape
        if self.sta_feature:
            return tf.TensorShape([bs, stimuli, N, ny, nx, nf * 2 + 1])
        else:
            return tf.TensorShape([bs, stimuli, N, ny, nx, nf + 1])

    def call(self, inputs):
        responses, g = inputs
        bs, stimuli, N = responses.shape

        # at this point g is purely based on the stimulus and has dimensions of
        #  batch_size X stimuli X nx X ny X features
        g = tf.expand_dims(g, axis=2)
        g = tf.tile(g, [1, 1, N, 1, 1, 1])
        bs, stimuli, N, nx, ny, nf = g.shape

        # at this point responses is batch_size X stimuli x N
        r = tf.expand_dims(tf.expand_dims(tf.expand_dims(responses, axis=3), axis=4), axis=5)
        r = tf.tile(r, [1, 1, 1, nx, ny, 1])

        # Finally concatenate out response
        if self.sta_feature:
            gr = tf.concat([g, g*r, r], axis=-1)
        else:
            gr = tf.concat([g, r], axis=-1)

        return gr


def image_to_gaussian(im, dtype=tf.float32):
    im = tf.cast(im, dtype=dtype)
    im.shape.assert_is_compatible_with([None, None, None, 1])

    tf.debugging.assert_all_finite(im, 'Received bad image to parameterize')

    _, Ny, Nx, Nc = im.shape

    im_flatten = tf.reshape(im, [-1, Nx * Ny])
    #im_flatten = (im_flatten - tf.reduce_min(im_flatten, axis=-1, keepdims=True)) + 1e-3
    #im_flatten = im_flatten / tf.reduce_sum(im_flatten, axis=-1, keepdims=True)
    im_flatten = tf.nn.softmax(im_flatten, axis=-1)

    x, y = tf.meshgrid(tf.linspace(0.0, Nx, Nx), tf.linspace(0.0, Ny, Ny))
    x = tf.cast(tf.reshape(x, [1, -1]), dtype=dtype)
    y = tf.cast(tf.reshape(y, [1, -1]), dtype=dtype)

    min_var = 1e-3

    mx = tf.reduce_sum(im_flatten * x, axis=-1, keepdims=True)
    my = tf.reduce_sum(im_flatten * y, axis=-1, keepdims=True)
    vx = tf.maximum(tf.reduce_sum(im_flatten * (x - mx) ** 2, axis=-1, keepdims=True), min_var)
    vy = tf.maximum(tf.reduce_sum(im_flatten * (y - my) ** 2, axis=-1, keepdims=True), min_var)

    vx = tf.sqrt(vx)
    vy = tf.sqrt(vy)

    params = tf.concat([my, mx, vy, vx], axis=-1)

    #tf.print(params[0])
    #tf.print(params[-1])

    #tf.print(im[-1])
    #tf.print(im_flatten[-1])
    tf.debugging.assert_all_finite(params, f'Guassin parameters broke. {im_flatten[-1]}')

    return params


def image_to_distribution(x):
    params = image_to_gaussian(x)
    loc, scale = tf.split(params, 2, axis=-1)
    _, Ny, Nx, Nc = x.shape

    # note we use (y,x) indexing to match indexing into images
    dist = tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=[Ny, Nx])

    return dist


class HeatmapToLocParams(tfkl.Layer):
    def __init__(self, **kwargs):
        super(HeatmapToLocParams, self).__init__(**kwargs)

        self.param_regressor = tfk.Sequential([tfkl.Flatten(),
                                               tfkl.Dense(4, bias_initializer=tf.initializers.zeros,
                                                          activity_regularizer=tfk.regularizers.l2(1e-3))])

    def call(self, inputs):

        tf.debugging.assert_all_finite(inputs, f'HeatmapToLocParams nan')

        params = self.param_regressor(inputs)

        loc, scale = tf.split(params, 2, axis=-1)
        _, Ny, Nx, Nc = inputs.shape

        dim_scale = tf.constant([[Ny, Nx]], dtype=inputs.dtype)
        loc = tf.nn.sigmoid(loc) * dim_scale
        scale = tf.nn.softplus(scale)  + 1e-3

        tf.debugging.assert_all_finite(loc, f'Guassian location broke')
        tf.debugging.assert_all_finite(scale, f'Guassian location broke')
        dist = tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=[Ny, Nx])

        return dist.sample()


class HigherRankOperator(tfkl.Layer):
    """ Layer that applys an operation across multiple ranks like the batch size

        This allows running something like a convolution over stimuli X neurons
        by unrolling those two dimensions into a single batch axis. The Conv2D
        operator expects the data to have this format. The size is also determined
        in the call function, which means it can vary the dimension from call to
        call.
    """

    def __init__(self, operator, op_rank=3, **kwargs):
        # op rank is how many dimensions the operator consumes,
        # not counting the batch
        self.op_rank = op_rank
        self.operator = operator
        kwargs['name'] = 'HRO-'+ operator.name
        super(HigherRankOperator, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # discard the second dimension (will be rolled into first)
        bs = tf.math.reduce_prod(input_shape[:-self.op_rank]) if input_shape[0] else None
        unrolled_shape = tf.TensorShape([bs, *input_shape[-self.op_rank:]])
        operator_shape = self.operator.compute_output_shape(unrolled_shape)
        return tf.TensorShape([*input_shape[:-self.op_rank], *operator_shape[1:]])

    def call(self, inputs):
        """ Perform the operator along stimulus x neurons. """

        # cache data shape to unfold back out
        dims = inputs.shape

        # wrap stimuli and neurons into batch dimension
        x = tf.reshape(inputs, [-1, *dims[-self.op_rank:]])

        # apply layer
        y = self.operator(x)

        # get dimensions of output from operator
        _, *new_dims = y.shape

        # and unfold back out
        y = tf.keras.backend.reshape(y, [-1, *dims[1:-self.op_rank], *new_dims])

        return y


_activation = tfkl.ELU


class ConvStack(tfk.Sequential):

    def __init__(self, architecture, **kwargs):
        super(ConvStack, self).__init__(**kwargs)

        for layer in architecture:

            if len(layer) == 5:
                depth, width, stride, norm, kwargs = layer
            elif len(layer) == 4:
                depth, width, stride, kwargs = layer
                norm = None
            else:
                depth, width, stride = layer
                norm = None
                kwargs = {}

            use_bias = True if norm is None else norm.lower() == 'batch'
            self.add(tfkl.Conv2D(depth, width, use_bias=use_bias, strides=stride, **kwargs))
            if norm is None:
                pass
            elif norm.lower() == 'batch':
                self.add(tfkl.BatchNormalization())
            elif norm.lower() == 'layer':
                self.add(tfkl.LayerNormalization(scale=True))
            self.add(_activation())


class SE2CNN(tf.keras.Sequential):

    def __init__(self, architecture, l2_weight=0.01, nb_orientations=8, padding='same', norm=None, dense=True, **kwargs):
        super(SE2CNN, self).__init__(**kwargs)

        regularizer = tfk.regularizers.l2(l2_weight)

        for i, (k, c) in enumerate(list(architecture[:-1])):
            if i == 0:
                self.add(SE2Lifting(filters=c, kernel_size=k, nb_orientations=nb_orientations,
                                    kernel_regularizer=regularizer, use_bias=False, name='GCNN-Lift'))
            else:
                self.add(SE2GroupSeparableConv(filters=c, kernel_size=k, padding=padding,
                                               nb_orientations=nb_orientations,
                                               kernel_regularizer=regularizer,
                                               densenet=dense, name=f'GCNN{i-1}'))

        if norm is None:
            pass
        elif norm.lower() == 'batch':
            # TODO: make sure BN on channel dimension is rotation equivariant
            self.add(tfkl.BatchNormalization())
        elif norm.lower() == 'layer':
            self.add(tfkl.LayerNormalization(scale=True))
        self.add(tfkl.Activation(_activation()))

        # use learnable projection from the equivariant form
        self.add(SE2ProjCat())

        c = architecture[-1]
        use_bias = True if norm is None else norm.lower() == 'batch'
        self.add(tfkl.Conv2D(c, 1, strides=1, padding='same', kernel_regularizer=regularizer, use_bias=use_bias))
        if norm is None:
            pass
        elif norm.lower() == 'batch':
            # TODO: make sure BN on channel dimension is rotation equivariant
            self.add(tfkl.BatchNormalization())
        elif norm.lower() == 'layer':
            self.add(tfkl.LayerNormalization(scale=True))

        self.add(tfkl.Activation(_activation()))


class CNN(ConvStack):

    def __init__(self, architecture=None, padding='same', norm=None, l2_weight=1e-4, **kwargs):
        # store for save and reconstruction
        if architecture is None:
            architecture = [[11, 32], [5, 32], [3, 32]]
        self.l2_weight = l2_weight
        self.architecture = architecture
        self.padding = padding
        self.norm = norm

        regularizer = tfk.regularizers.l2(l2_weight)

        arch_list = []
        for i, (k, c) in enumerate(list(architecture)):
            if i == 0:
                arch_list.append([c, k, 1, self.norm, {'kernel_regularizer': regularizer, 'padding': 'valid'}])
            else:
                arch_list.append([c, k, 1, self.norm, {'kernel_regularizer': regularizer, 'padding': self.padding}])

        super(CNN, self).__init__(arch_list, **kwargs)

    def get_fn(self):
        return 'convstack_' + '_'.join(str(x) for x in self.architecture + [self.norm])


class Predictor(tfk.Model):

    def __init__(self,
                 sta_feature=True,
                 cummulative=False,
                 laplace_weight=0.0,
                 cell_latent_dim=10,
                 l2_weight=0.01,
                 activity_weight=0.0,
                 architecture=[[11, 32], [5, 32], [3, 32], 'same', "none"],
                 contrastive_weight=0.05,
                 contrastive_dimension=5,
                 stochastic=True,
                 samples=1,
                 name='predictor', **kwargs):
        super(Predictor, self).__init__(name=name, **kwargs)

        print(f'Architecture: {architecture}')
        self.sta_feature = sta_feature
        self.regularizer = tfk.regularizers.l2(l2_weight)
        self.activity_regularizer = tfk.regularizers.l2(activity_weight)
        self.cummulative = cummulative
        self.samples = samples

        if type(architecture) == list:
            # this maintains backward compatibility with prior way of passing the
            # architecture
            channel_depth = architecture[-3]
            self.channel_depth = channel_depth

            im_conv = SE2CNN(architecture=architecture[:-2], padding=architecture[-2], norm=architecture[-1], l2_weight=l2_weight, name='SE2CNN')
        else:

            if 'nb_orientations' in architecture.keys() and architecture['nb_orientations'] == 1:
                architecture.pop('nb_orientations')
                print('Constructing regular CNN (not G-CNN)')
                channel_depth = architecture['architecture'][-1][-1]
                im_conv = CNN(**architecture)
            else:
                print('Constructin G-CNN')
                channel_depth = architecture['architecture'][-1]
                im_conv = SE2CNN(**architecture, name='SE2CNN')

        print(f'Channel depth: {channel_depth}')
        self.channel_depth = channel_depth
        self.im_conv_wrapper = HigherRankOperator(im_conv)

        # import os
        # f = os.path.join(os.path.dirname(__file__), 'weights', self.im_conv.get_fn())
        # if os.path.exists(f + '.index'):
        #     print(f"Loading initializing weights from {f}")
        #     self.im_conv.load_weights(f)
        # else:
        #     print(f'Weights not found at {f}')

        # Layer that appends responses to convolution output
        self.crc = ConvResponseConcatenate(self.sta_feature)

        # convolution layers that mixes responses and image features into a map of receptive
        # fields.
        # use some 1x1 convolutions on the
        loc_norm = None
        location_heatmap = ConvStack([[self.channel_depth, 1, 1, loc_norm, {'kernel_regularizer': self.regularizer}],
                                      [self.channel_depth, 1, 1, loc_norm, {'kernel_regularizer': self.regularizer}],
                                      [1, 1, 1]], name='loc_features_conv')
        location_heatmap.pop()

        #heatmap_to_dist = HeatmapToLocParams()

        # draw samples from the distribution and move them from the batch dimension
        heatmap_to_dist = tf.keras.layers.Lambda(lambda x: tf.transpose(image_to_distribution(x).sample(samples), [1, 0, 2]))

        self.location_predictor = tfk.Sequential([
            # Perform convolution on each g-response image and output flattend version
            HigherRankOperator(location_heatmap),
            # Exclusive set collapse
            DeepSetSimple(cummulative=self.cummulative),
            # Take the collapsed image and convert to distribution
            HigherRankOperator(heatmap_to_dist)
        ], name='location_predictor')

        if stochastic:
            bottleneck_layer = [
                tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(cell_latent_dim), activation=None),
                tfpl.MultivariateNormalTriL(cell_latent_dim, convert_to_tensor_fn=lambda x: x.sample())
            ]
        else:
            bottleneck_layer = [
                tfkl.Dense(cell_latent_dim, activation=_activation())
            ]

        self.feature_mlp = tfk.Sequential([tfkl.Dense(self.channel_depth + cell_latent_dim, activation=_activation()),
                                           tfkl.Dense(self.channel_depth + cell_latent_dim, activation=_activation()),
                                           DeepSetSimple(cummulative=self.cummulative),
                                           tfkl.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
                                           *bottleneck_layer,
                                           tfkl.Dense(self.channel_depth + 1, activation=_activation()),
                                           tfkl.Dense(self.channel_depth + 1, activation=None)])

        self.contrastive_weight = contrastive_weight
        if contrastive_weight > 0:
            self.contrastive_projection = tfk.Sequential([tfkl.Dense(self.channel_depth, activation=_activation()),
                                                          tfkl.Dense(contrastive_dimension, activation=None),
                                                          tfkl.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))])

    def compute_output_shape(self, input_shape):
        return input_shape[0] + [self.samples]

    def get_config(self):
        return {'name': self.name, 'channel_depth': self.channel_depth, 'sta_feature': self.sta_feature}

    def compute_summary(self, inputs, return_im_feat=False):
        responses, stimuli = inputs

        # convolve input stimuli
        g = self.im_conv_wrapper(stimuli)
        gr = self.crc([responses, g])

        sample_locations = self.location_predictor(gr)

        # extract the image feature for each trial x neuron estimate of the location
        bs, stimuli, Ny, Nx, Nc = g.shape
        bs, stimuli, neurons, samples, coordinates = sample_locations.shape
        tf.assert_equal(coordinates, 2)
        im_feat = interpolate_bilinear(tf.reshape(g, [-1, Ny, Nx, Nc]),
                                       tf.reshape(sample_locations, [-1, neurons * samples, 2]))
        im_feat = tf.reshape(im_feat, [-1, stimuli, neurons, samples, Nc])

        # construct vector for each trial that includes information about the responses
        # and the feature, including a STA type response
        response_samples = tf.tile(responses[:, :, :, None, None], [1, 1, 1, samples, 1])
        x2 = tf.concat([im_feat, im_feat * response_samples, response_samples], axis=-1)

        # then let those interact through an MLP and then compute an average feature.
        # again for trial N this is computed only using information from the other
        # trials. This should compute a summary statistics describing a neuron (other
        # than the spatial location) based on those other trials.
        cell_summary = self.feature_mlp(x2)

        if not return_im_feat:
            return sample_locations, cell_summary
        else:
            return sample_locations, cell_summary, im_feat

    def compute_grad_input(self, inputs):
        responses, stimuli = inputs

        neurons = responses.shape[-1]
        image_shape = stimuli.shape[2:]

        sample_locations, cell_summary = self.compute_summary(inputs)

        # take neuron summary statistics from first batch and last trial
        w, b = cell_summary[0, -1, ..., 0, :-1], cell_summary[0, -1, ..., 0, -1]
        sample_locations = tf.expand_dims(sample_locations[0, -1, :, 0], axis=1)

        blank_im = tf.constant(0.0, dtype=tf.float32, shape=[neurons, *image_shape], name='blank_im')

        with tf.GradientTape() as grad:
            grad.watch(blank_im)
            g = self.im_conv_wrapper.operator(blank_im, training=False)

            # extract the image feature for each trial x neuron estimate of the location
            # note we drop the middle dimension as we are using the batch dimension for
            # neurons here and that is singleton
            im_feat = interpolate_bilinear(g, sample_locations)[:,0,:]

            t = tf.reduce_sum(tf.multiply(im_feat, w), axis=-1) + b

        dy_dx = grad.gradient(t, blank_im)[..., 0]

        return dy_dx

    def contrastive_loss(self, inputs):
        responses, stimuli = inputs

        responses1, responses2 = tf.split(responses, 2, axis=1)
        stimuli1, stimuli2 = tf.split(stimuli, 2, axis=1)

        _, cell_summary1 = self.compute_summary((responses1, stimuli1))
        _, cell_summary2 = self.compute_summary((responses2, stimuli2))

        p1 = self.contrastive_projection(cell_summary1[:, -1, ..., :-1])
        p2 = self.contrastive_projection(cell_summary2[:, -1, ..., :-1])

        p1 = tf.reshape(p1, [-1, tf.shape(p1)[-1]])
        p2 = tf.reshape(p2, [-1, tf.shape(p1)[-1]])

        neurons = tf.shape(p1)[0]
        temperature = 0.5
        LARGE_NUM = 1e9
        labels = tf.one_hot(tf.range(neurons), neurons * 2)
        masks = tf.one_hot(tf.range(neurons), neurons)

        logits_aa = tf.matmul(p1, p1, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(p2, p2, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(p1, p2, transpose_b=True) / temperature
        logits_ba = tf.matmul(p2, p1, transpose_b=True) / temperature

        loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b

        return tf.reduce_mean(loss)

    def call(self, inputs):

        if self.contrastive_weight > 0:
            cl = self.contrastive_loss(inputs)
            self.add_loss(cl * self.contrastive_weight)
            self.add_metric(cl, aggregation='mean', name='contrastive_loss')

        locations, cell_summary, im_feat = self.compute_summary(inputs, return_im_feat=True)

        w, b = cell_summary[..., :-1], cell_summary[..., -1]
        t = tf.reduce_sum(tf.multiply(im_feat, w), axis=-1) + b
        t = tf.nn.elu(t) + 1

        def multiple_samples_mixture_distribution(lam):
            dist = tfd.Poisson(tf.clip_by_value(lam, 1e-2, 1e3))
            dist = tfd.MixtureSameFamily(tfd.Categorical(probs=[1.0/self.samples] * self.samples),
                                         components_distribution=dist)
            return dist

        return tfpl.DistributionLambda(make_distribution_fn=lambda s: multiple_samples_mixture_distribution(s),
                                       convert_to_tensor_fn=lambda s: s.mean())(t)


def test_compile_save():
    neurons = 32
    image_shape = [64, 34, 1]

    inputs = [tfk.Input([neurons], name='responses'), tfk.Input(image_shape, name='stimuli')]
    predictor = Predictor()
    pred = predictor(inputs)

    model = tfk.Model(inputs, pred)
    optimizer = tf.optimizers.Adam(learning_rate=1e-2, clipnorm=10)

    model.compile(loss='mse', optimizer=optimizer)

    model.summary()
    predictor.summary()

    def print_layers(layers, depth=0):
        for layer in layers:
            print((depth, layer.name, layer))
            if hasattr(layer, 'layers'):
                print_layers(layer.layers, depth+1)

    print_layers(model.layers)

    # Test model works when changing to different sizes
    neurons = 16
    stimuli = 100
    x = [tf.ones([stimuli, neurons]), tf.ones([stimuli, *image_shape])]
    model(x)

    neurons = 32
    stimuli = 200
    x = [tf.ones([stimuli, neurons]), tf.ones([stimuli, *image_shape])]
    model(x)

    model.save('test_save.h5')
    model.save_weights('test_save_weights.h5')

    #model.save_weights('gs://emg-imu-mlengine/tf-determinsitic-training/models/deterministic_sythenticFalse_neurons32_stimuli512_channels32_staTrue_lr_0.01_20190512_131026.h5')

if __name__ == '__main__':
    test_compile_save()
