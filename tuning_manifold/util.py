import tensorflow as tf


def negloglik(x, rv_x):
    return -rv_x.log_prob(x)


def pearson(y_true, y_pred, axis=1):
    from tensorflow.python.framework import ops
    from tensorflow.python.keras import backend as K
    from tensorflow.python.ops import math_ops

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    detrended_true = y_true - K.mean(y_true, axis=axis, keepdims=True)
    detrended_pred = y_pred - K.mean(y_pred, axis=axis, keepdims=True)

    std_true = math_ops.sqrt(K.mean(math_ops.square(detrended_true), axis=axis, keepdims=True))
    std_pred = math_ops.sqrt(K.mean(math_ops.square(detrended_pred), axis=axis, keepdims=True))

    return K.mean(detrended_true * detrended_pred / std_true / std_pred, axis=axis)


class EarlyStoppingNan(tf.keras.callbacks.EarlyStopping):
    def init__(self, **kwargs):
        super(EarlyStoppingNan, self).__init__(self, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super(EarlyStoppingNan, self).on_epoch_end(epoch, logs)

        current = self.get_monitor_value(logs)
        if current is None:
            return

        print('Current: {}'.format(current))
        if tf.math.is_nan(current).numpy():
            print("Stopping on NAN")
            self.stopped_epoch = epoch
            self.model.stop_training = True


def make_image_grid(images, Nx=4, Ny=4, scale=True, border=True):
    import numpy as np
    image_shape = images[0].shape

    if scale:
        def rescale(x):
            x = x - np.percentile(x, 0.1)
            x = x / np.percentile(x, 99.9)
            x[x < 0] = 0
            x[x > 1] = 1
            return x

        images = [rescale(x) for x in images]

    if border:
        for i in range(len(images)):
            images[i][:, -1] = 0
            images[i][-1, :] = 0

    # do some voodoo to shuffle images around
    dat = np.concatenate(images, axis=0)
    tiled = np.concatenate([np.concatenate(list(x), axis=1) for x in
                            list(dat.reshape(Ny, Nx, image_shape[0], image_shape[1]))], axis=0)

    return tiled


@tf.function
def interpolate_bilinear(grid, query_points, name=None):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      name: a name for the operation (optional).
    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """

    with tf.name_scope(name or "interpolate_bilinear"):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)

        # grid shape checks
        grid_static_shape = grid.shape
        grid_shape = tf.shape(grid)
        if grid_static_shape.dims is not None:
            if len(grid_static_shape) != 4:
                raise ValueError("Grid must be 4D Tensor")
            if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
                raise ValueError("Grid height must be at least 2.")
            if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
                raise ValueError("Grid width must be at least 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        grid_shape[1], 2, message="Grid height must be at least 2."
                    ),
                    tf.debugging.assert_greater_equal(
                        grid_shape[2], 2, message="Grid width must be at least 2."
                    ),
                    tf.debugging.assert_less_equal(
                        tf.cast(
                            grid_shape[0] * grid_shape[1] * grid_shape[2],
                            dtype=tf.dtypes.float32,
                        ),
                        np.iinfo(np.int32).max / 8.0,
                        message="The image size or batch size is sufficiently "
                        "large that the linearized addresses used by "
                        "tf.gather may exceed the int32 limit.",
                    ),
                ]
            ):
                pass

        # query_points shape checks
        query_static_shape = query_points.shape
        query_shape = tf.shape(query_points)
        if query_static_shape.dims is not None:
            if len(query_static_shape) != 3:
                raise ValueError("Query points must be 3 dimensional.")
            query_hw = query_static_shape[2]
            if query_hw is not None and query_hw != 2:
                raise ValueError("Query points last dimension must be 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        query_shape[2],
                        2,
                        message="Query points last dimension must be 2.",
                    )
                ]
            ):
                pass

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_static_shape[1],
            grid_static_shape[2],
            grid_static_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] # if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


def load_datajoint_model(run_id, neurons=None, stimuli=None, samples=None):
    data_path = '/home/jcotton/projects/tuning_manifold/data/'

    from tuning_manifold.dj_train import Np4Np, Np4NpParam, schema
    from tuning_manifold.fnp_model import Predictor
    from tuning_manifold.util import negloglik, pearson
    from tuning_manifold.synthetic_sampler import dataset_sampler as synthetic_sampler
    from tuning_manifold.dataset_sampler import dataset_sampler

    import tempfile
    import base64

    tfk = tf.keras

    res = (Np4Np() * Np4NpParam() & f'run_id={run_id}').fetch1()

    # make a temporary file with model from the database
    binary = base64.b64decode(res['final_model'])
    model_file = tempfile.NamedTemporaryFile('wb', suffix='.h5')
    model_file.write(binary)

    neurons = neurons or res['neurons']
    stimuli = stimuli or res['stimuli']
    sta_feature = res['sta_feature']
    cummulative = res['cummulative']
    cell_latent = res['cell_latent']
    architecture = eval(res['architecture'])
    learning_rate = res['learning_rate']
    l2_weight = res['l2_weight']
    activity_weight = res['activity_weight']
    laplace_weight = res['laplace_weight']
    contrastive_loss = res['contrastive_loss']
    samples = samples or res['samples']

    image_shape = eval(res['image_shape'])
    synthetic_rf_dim = res['synthetic_rf_dim']

    if neurons < 0:
        validation_data = None # if we don't want to reload data
    elif res['synthetic']:
        training_data, validation_data, image_shape = synthetic_sampler(neurons=neurons,
                                                                        n_stimuli=stimuli,
                                                                        dim_rf=synthetic_rf_dim,
                                                                        image_shape=image_shape)
        validation_data = validation_data.batch(1)
    else:
        training_data, validation_data, image_shape = dataset_sampler(data_path, neurons=neurons,
                                                                      n_stimuli=stimuli)
        validation_data = validation_data.batch(1)

    # this code matches most of our training and should be refactored somewhere
    inputs = [tfk.Input([stimuli, neurons], name='responses'), tfk.Input([stimuli, *image_shape], name='stimuli')]
    predictor = Predictor(architecture=architecture,
                          l2_weight=l2_weight, cell_latent_dim=cell_latent,
                          sta_feature=sta_feature, cummulative=cummulative,
                          activity_weight=activity_weight, laplace_weight=laplace_weight,
                          contrastive_weight=contrastive_loss,
                          samples=samples)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=10)
    pred = predictor(inputs)

    model = tfk.Model(inputs, pred)
    model.compile(loss=negloglik,
                  metrics=[pearson, 'mse'],
                  optimizer=optimizer)

    model.load_weights(model_file.name)
    model_file.close()

    return model, validation_data


def compute_excess(model, data, iterations=1000):
    import tensorflow_probability as tfp
    import numpy as np

    tfpl = tfp.layers
    tfd = tfp.distributions

    ll = []
    true_ll = []

    import itertools
    for data in itertools.islice(data, iterations):
        ll.append(-model(data[0]).log_prob(data[1])[0])

        true_dist = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.Poisson(t))(data[0][-1] + 1e-7)
        true_ll.append(negloglik(data[1], true_dist)[0])

    ll = np.concatenate(ll, axis=-1)
    true_ll = np.concatenate(true_ll, axis=-1)

    final_ll = np.mean(np.mean(ll, axis=-1)[-10:])
    true_ll = np.mean(true_ll)
    model_ll = np.mean(ll, -1)  # across neurons

    data = dict({'final_ll': final_ll,
                 'true_ll': true_ll,
                 'excess_ll': final_ll - true_ll,
                 'll_function': model_ll})

    return data


def compute_fev(model, data, samples=100, iterations=1000):
    import numpy as np
    import itertools
    import scipy

    fev = []
    r = []

    for sample in itertools.islice(data, iterations):
        assert len(sample[0]) > 2, "Cannot compute FEV for ground truth"

        mr = sample[0][-1][0]  # ground truth
        rhat = model(sample[0], training=False).mean()[0] # prediction

        v = np.var(mr, axis=0, keepdims=True)
        fev.extend(1 - np.mean((mr[-samples:] - rhat[-samples:]) ** 2 / v, axis=0))
        r.extend([scipy.stats.pearsonr(mr[-samples:, i].numpy(), rhat[-samples:, i].numpy())[0]
                  for i in range(rhat.shape[1])])

    return dict({'mean_r': np.mean(r),
                 'mean_fev': np.mean(fev),
                 'r': r,
                 'fev': fev})