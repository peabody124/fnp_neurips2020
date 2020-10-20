#!/usr/env python

import os
import datetime
from absl import app, flags
import tensorflow as tf
import numpy as np

from tuning_manifold.synthetic_sampler import dataset_sampler as synthetic_sampler, kernel_generator
from tuning_manifold.fnp_model import Predictor
from tuning_manifold.util import pearson, negloglik, make_image_grid

tfk = tf.keras

log_path = '/logs/'
data_path = '/data'
model_path = '/models/'

# Catch when we are not running in docker
if not os.path.isdir(log_path):
    log_path = '~/projects/tuning_manifold/logs'
if not os.path.isdir(data_path):
    data_path = '~/projects/tuning_manifold/data/'
if not os.path.isdir(model_path):
    model_path = '~/projects/tuning_manifold/models/'


FLAGS = flags.FLAGS
flags.DEFINE_integer('neurons', 8, 'Number of neurons to use')
flags.DEFINE_integer('stimuli', 512, 'Number of stimuli per minibatch')
flags.DEFINE_integer('samples', 8, 'Number of samples from latents to use')
flags.DEFINE_integer('cell_latent_dim', 32, 'Bottleneck for the cell latent dimensionality')
flags.DEFINE_boolean('synthetic', True, 'Test on synthetic stimuli')
flags.DEFINE_integer('synthetic_rf_dim', 3, 'Number of dimensions for artificial RFs')
flags.DEFINE_boolean('checkpoints', True, 'Save model checkpoint at each step')
flags.DEFINE_boolean('early_stopping', True, 'Enable early stopping')
flags.DEFINE_boolean('sta_feature', True, 'Include spike triggered average feature')
flags.DEFINE_boolean('cummulative', True, 'Use cummulative embedding (versus leave one out)')
flags.DEFINE_string('image_shape', "(16,16,1)", 'CNN architecture')
flags.DEFINE_string('architecture', "[[11, 32], [5, 8], [3, 8], 32, 'same','batch']", 'CNN architecture')
flags.DEFINE_float("learning_rate", 1e-4, 'learning rate for gradient descent, default=.0001')
flags.DEFINE_float("l2_weight", 1e-4, 'L2 weight applied throughout model, default=.0001')
flags.DEFINE_float("activity_weight", 0, 'L2 weight applied throughout model, default=0')
flags.DEFINE_float("laplace_weight", 0, 'L2 weight applied throughout model, default=0')
flags.DEFINE_float("contrastive_loss", 0.0, 'Weight applied to contrastive loss, default=0')


def main(argv):

    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    # tf.compat.v1.disable_eager_execution()

    random_seed = 1234
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    mean_true_likelihood = 0

    if FLAGS.synthetic:
        training_data, validation_data, image_shape = synthetic_sampler(neurons=FLAGS.neurons, n_stimuli=FLAGS.stimuli,
                                                                        dim_rf=FLAGS.synthetic_rf_dim,
                                                                        image_shape=eval(FLAGS.image_shape))

        likelihoods = []
        import itertools
        for data in itertools.islice(training_data, 10):
            import tensorflow_probability as tfp
            tfd = tfp.distributions
            tfpl = tfp.layers

            inputs, outputs = data
            responses, stimuli, params, mean_response = inputs

            true_dist = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.Poisson(t))(mean_response + 1e-7)
            true_ll = negloglik(outputs, true_dist)

            likelihoods.append(true_ll)
        mean_true_likelihood = np.mean(tf.concat(likelihoods, axis=0))
        print(f'Mean True Likelihood: {mean_true_likelihood}')
    else:
        # Note: implement sample for your data
        training_data, validation_data, image_shape = dataset_sampler(data_path, neurons=FLAGS.neurons,
                                                                      n_stimuli=FLAGS.stimuli)

    print(tf.config.experimental.list_logical_devices('GPU'))
    logical_gpus = len(tf.config.experimental.list_logical_devices('GPU'))
    training_data = training_data.batch(logical_gpus)
    validation_data = validation_data.batch(logical_gpus)

    sampled_per_minibatch = FLAGS.neurons * FLAGS.stimuli * 1
    scaled_learning_rate = FLAGS.learning_rate * sampled_per_minibatch / (1024.0 * 16.0)
    print(f'Scaled learning rate is {scaled_learning_rate}')

    if logical_gpus == 1:
        class Stub():
            def scope(self):
                import contextlib
                return contextlib.suppress()
        mirrored_strategy = Stub()
    else:
        print(f'Using {logical_gpus} GPUs')
        mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():

        inputs = [tfk.Input([FLAGS.stimuli, FLAGS.neurons], name='responses'), tfk.Input([FLAGS.stimuli, *image_shape], name='stimuli')]
        predictor = Predictor(l2_weight=FLAGS.l2_weight, architecture=eval(FLAGS.architecture),
                              sta_feature=FLAGS.sta_feature, cummulative=FLAGS.cummulative,
                              laplace_weight=FLAGS.laplace_weight, activity_weight=FLAGS.activity_weight,
                              contrastive_weight=FLAGS.contrastive_loss, cell_latent_dim=FLAGS.cell_latent_dim,
                              samples=FLAGS.samples)
        pred = predictor(inputs)

        model = tfk.Model(inputs, pred)

        optimizer = tf.optimizers.Adam(learning_rate=scaled_learning_rate) #, clipnorm=10)

        model.compile(optimizer, metrics=[pearson, 'mse'], loss=lambda x, y: negloglik(x, y) - mean_true_likelihood)

    predictor.im_conv_wrapper.operator.summary()
    predictor.location_predictor.summary()
    predictor.summary()
    model.summary()

    time_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    architecture = eval(FLAGS.architecture)
    if type(architecture) == dict:
        architecture = architecture['architecture']
    name = "deterministic_latent{}_architecture{}_sythentic{}_rfDim{}_neurons{}_stimuli{}_samples{}_sta{}_cummulative{}_lr{}_l2{}_cl{}_{}"\
        .format(FLAGS.cell_latent_dim, architecture, FLAGS.synthetic, FLAGS.synthetic_rf_dim,
                FLAGS.neurons, FLAGS.stimuli, FLAGS.samples, FLAGS.sta_feature,
                FLAGS.cummulative, FLAGS.learning_rate, FLAGS.l2_weight, FLAGS.contrastive_loss,
                time_name)

    model_file_base = os.path.join(model_path, name + '_{epoch:02d}_{loss:.2f}_{pearson:.3f}.hdf5')

    cbs = []

    # Setup Learning Rate decay.
    # lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
    #   lambda epoch: FLAGS.learning_rate + (1e-2 - FLAGS.learning_rate) * (0.5 ** epoch) )
    # cbs.append(lr_decay_cb)

    # seems to trigger some warnings about using the profile
    log_file_dir = os.path.join(log_path, 'deterministic/{}'.format(name))
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_file_dir)
    cbs.append(tb)

    if FLAGS.synthetic:
        file_writer_rfs = tf.summary.create_file_writer(log_file_dir + '/rfs')
        plotting_neurons = 16
        plotting_data, _, _ = synthetic_sampler(neurons=plotting_neurons, n_stimuli=FLAGS.stimuli,
                                                dim_rf=FLAGS.synthetic_rf_dim, image_shape=eval(FLAGS.image_shape))
        plotting_data = plotting_data.batch(1)
        plotting_iter = iter(plotting_data)
        validation_rfs = next(plotting_iter)

    def make_summary_fig(iterations=10):
        # Same as above but using the mean response from ground truth
        import io
        import itertools
        import matplotlib.pyplot as plt
        import tensorflow_probability as tfp

        tfpl = tfp.layers
        tfd = tfp.distributions

        deltas = []
        for data in itertools.islice(plotting_iter, iterations):
            responses, stimuli, params, mean_response = data[0]

            # this also seems to do what we expect
            predicted_distribution = predictor((responses, stimuli))

            model_ll = negloglik(responses, predicted_distribution)

            true_dist = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.Poisson(t))(mean_response + 1e-7)
            true_ll = negloglik(data[1], true_dist)

            delta_ll = model_ll - true_ll
            deltas.append(delta_ll)

        delta = tf.concat(deltas, axis=-1)[0] # batch x trials x neurons * iterations
        plt.figure(figsize=(10, 7))
        plt.plot(tf.reduce_mean(delta, axis=-1), 'k.')
        plt.xlabel('Trials')
        plt.ylabel('$\Delta$ LL (model - true)')
        plt.ylim(0, 5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        im = plt.imread(buf)

        return im

    def log_rfs(epoch, logs):

        dy_dx = list(predictor.compute_grad_input(validation_rfs[0][:-2]).numpy())

        grad_rf = make_image_grid(dy_dx, Ny=int(plotting_neurons/4))
        grad_rf = tf.cast(tf.expand_dims(tf.expand_dims(grad_rf, 0), -1), tf.float16)
        true_rf = make_image_grid(list(kernel_generator(validation_rfs[0][2][0], size=image_shape)), Ny=int(plotting_neurons / 4))
        true_rf = tf.cast(tf.expand_dims(tf.expand_dims(true_rf, 0), -1), tf.float16)
        im = tf.concat([grad_rf, true_rf], axis=1)
        with file_writer_rfs.as_default():
            tf.summary.image("RF Grad", im, step=epoch)

        im = make_summary_fig()
        with file_writer_rfs.as_default():
            tf.summary.image("Excessive Uncertainty", im[None, ...], step=epoch)

    # if FLAGS.synthetic:
    #     rf_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_rfs)
    #     cbs.append(rf_callback)

    if FLAGS.checkpoints:
        ms = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_base, save_best_only=True, monitor='loss')
        cbs.append(ms)

    # es = EarlyStoppingNan(patience=50, monitor='pearson')
    # if FLAGS.early_stopping:
    #    es = tf.keras.callbacks.EarlyStopping(patience=75, monitor='loss')
    #    cbs.append(es)

    model.fit(training_data,
              epochs=1500, steps_per_epoch=250,
              validation_data=validation_data, validation_steps=40,
              callbacks=cbs,
              verbose=1)

    print(f'Saving to {name}.h5')
    fn = os.path.join(model_path, name + '.h5')
    model.save(fn)

    print('Evaluation')
    model.load_weights(fn)
    model.evaluate(training_data, steps=100)


if __name__ == "__main__":
    # execute only if run as a script
    app.run(main)
