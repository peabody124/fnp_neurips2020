import tensorflow as tf
import numpy as np


def kernel_generator(x_gen, size=(36, 64, 1), phase_shifted=False):
    """Generator for sample images (e.g. proxy for MEIs)

    This will infer the dimensionality of the latent space to create images, using
    a default if there is less

    dim1 - orientation
    dim2 - xoffset
    dim3 - yoffset
    dim4 - spatial frequency
    dim5 - width
    dim6 - offset
    """

    from skimage.filters import gabor_kernel
    from scipy.ndimage import shift

    for x in x_gen:
        dim = x.shape[0]
        theta = x[0]  # always at least one latent variable

        # defaults that are overridden by parameters
        xoffset = 0
        yoffset = 0
        freq = 0.16
        width = 2
        offset = 0
        gain = 1

        if dim > 1: xoffset = x[1]
        if dim > 2: yoffset = x[2]
        if dim > 3: freq = 0.08 + 0.16 / (1 + np.exp(x[3]))
        if dim > 4: width = 2 + 1 / (1 + np.exp(x[4]))
        if dim > 5: offset = np.array(x[5])
        if dim > 6: pass # this is used outside this to determine complex cell
        if dim > 7: gain = 1.0 + 0.5 * np.tanh(x[7])

        if phase_shifted:
            offset = offset + np.pi / 2

        std = 64 / width

        arr = np.real(gabor_kernel(frequency=freq, theta=theta, n_stds=std, sigma_x=width, sigma_y=width, offset=offset))
        arr = gain * arr / np.max(np.abs(arr))

        # shift image
        arr = shift(arr, [yoffset, xoffset])

        # clip image at specified size
        dim1, dim2 = arr.shape
        offset1 = (dim1 - size[0]) >> 1
        offset2 = (dim2 - size[1]) >> 1

        # if size is even and the kernel is odd, throw additional row/column out
        if size[0] % 2 == 0: arr = arr[1:]
        if size[1] % 2 == 0: arr = arr[:, 1:]

        yield arr[offset1:-offset1, offset2:-offset2]


def dataset_sampler(n_stimuli=512, neurons=64, training_neurons=5000, total_stimuli=10000, dim_rf=5,
                    image_shape=(16, 16, 1)):
    """ Generate simulated response from neurons

        For each neuron picks a preferred stimulus and a selection of stimuli.
        Then performs a convolution with each stimulus against the RF to generate
        a simulated response. The batch dimension is a set of (stimuli, response)
        pairs.

        Params:
            neurons (int)         : number of neurons to simulate for each block of stimuli
            stimuli (list of int) : list of how many stimuli to generate in each mini-batch
            dim_stim (int)        : dimensionality of the stimulus space
            dim_rf (int)          : dimensionality of the receptive field space
    """

    test_im = next(kernel_generator(np.array([[0]]), size=image_shape))
    image_shape = [*test_im.shape, 1]

    import tensorflow_datasets as tfds

    if image_shape[0] == 16:
        (tfds_train_image_ds, tfds_test_image_ds), _ = tfds.load('imagenet_resized/16x16', split=['train', 'validation'],
                                                                   shuffle_files=True, with_info=True,
                                                                   data_dir='/home/jcotton/tensorflow_datasets')
    elif image_shape[0] == 32:
        (tfds_train_image_ds, tfds_test_image_ds), _ = tfds.load('imagenet_resized/32x32', split=['train', 'validation'],
                                                                   shuffle_files=True, with_info=True,
                                                                   data_dir='/home/jcotton/tensorflow_datasets')

    def get_and_shape_stim(dataset):
        images = next(iter(dataset.batch(total_stimuli)))['image']
        images = tf.cast(images, tf.float32).numpy()[..., 0]
        images = images / 256.0 - 0.5
        return images

    training_stimuli = get_and_shape_stim(tfds_train_image_ds)
    testing_stimuli = get_and_shape_stim(tfds_test_image_ds)

    bounds = np.array([2 * np.pi, image_shape[1] / 3, image_shape[0] / 3])

    def generate_neuron_params(n):
        params = [np.random.uniform(-bounds * np.ones((n, 1)), bounds * np.ones((n, 1))),  # theta, x and y      [0:3]
                  np.random.randn(n, 2),                                                   # frequency and width [3:5]
                  np.random.uniform(0, np.ones((n, 1)) * np.pi),                           # phase offset        [5]
                  np.random.choice([0, 1], (n, 1)),                                        # complex or not      [6]
                  np.random.randn(n, 1)                                                    # gain                [7]
                 ]
        return np.concatenate(params, axis=1)[:, :dim_rf]

    if training_neurons is not None:
        all_neuron_params = generate_neuron_params(training_neurons)

    def gen(training):

        while True:

            stimuli_idx = np.random.choice(total_stimuli, size=n_stimuli, replace=False)
            if training:
                stimuli = training_stimuli[stimuli_idx]
            else:
                stimuli = testing_stimuli[stimuli_idx]

            # create latent variables for images
            if not training or training_neurons is None:
                neuron_params = generate_neuron_params(neurons)
            else:
                neuron_idx = np.random.choice(all_neuron_params.shape[0], size=neurons, replace=False)
                neuron_params = all_neuron_params[neuron_idx]

            neuron_kernels = np.stack(list(kernel_generator(neuron_params, size=image_shape)))
            responses = np.einsum('ijk,ljk', neuron_kernels, stimuli).transpose()

            neuron_kernels_ps = np.stack(list(kernel_generator(neuron_params, size=image_shape, phase_shifted=True)))
            responses_ps = np.einsum('ijk,ljk', neuron_kernels_ps, stimuli).transpose()

            if dim_rf >= 7:
                complex = neuron_params[:, 6]
            else:
                complex = np.zeros((neurons,))

            complex_response = tf.sqrt(responses**2 + responses_ps**2 * complex)
            mean_responses = tf.nn.relu(responses) * (1-complex) + complex_response * complex
            mean_responses = mean_responses * 5
            responses = np.random.poisson(lam=mean_responses)

            # go through the individual neurons
            yield (responses, stimuli[:, :, :, np.newaxis], neuron_params, mean_responses)

    def _preprocess(r, s, p, mr):
        return (r, s, p, mr), r

    types = (tf.float32, tf.float32, tf.float32, tf.float32)
    shapes = (tf.TensorShape([None, neurons]), tf.TensorShape([None, *image_shape]),
              tf.TensorShape([neurons, dim_rf]), tf.TensorShape([None, neurons]))

    training_ds = tf.data.Dataset.from_generator(gen, types, shapes, [True])
    training_ds = training_ds.map(_preprocess, num_parallel_calls=8).prefetch(buffer_size=128)

    validation_ds = tf.data.Dataset.from_generator(gen, types, shapes, [True])
    validation_ds = validation_ds.map(_preprocess, num_parallel_calls=8).prefetch(buffer_size=128)

    return training_ds, validation_ds, image_shape
