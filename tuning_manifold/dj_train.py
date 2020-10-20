'''

DataJoint code to organize training the Neural Processes for Neural Processes

For now keep it fairly simple, although this might need to generalize. Lookup table with the
run_id as the primary key and the parameters as the subsequent keys. Will also need to include
the githash. At the point where we break the calling convention will just trash it all and start
over for now. Later can get more clever like Fabian suggested w.r.t. a configuration string and
githash in the lookup table and a derived table that is used for restrictions.

As a result of the run we want to store the final loss and validation loss, pearson and validation
pearson, and finally the training history as a blob and the final model as a blob.
'''

import datajoint as dj
dj.config["enable_python_native_blobs"] = True

schema = dj.schema('neural_processes2', locals())

@schema
class Np4NpParam(dj.Lookup):
    definition = """
    run_id                          : smallint  # unique run ID
    ---
    neurons                         : smallint
    stimuli                         : smallint  
    cell_latent                     : smallint
    samples                         : smallint
    synthetic                       : boolean
    synthetic_rf_dim                : smallint
    sta_feature                     : boolean
    cummulative                     : boolean
    architecture                    : varchar(200)
    image_shape                     : varchar(200)
    learning_rate                   : float
    l2_weight                       : float
    activity_weight                 : float
    laplace_weight                  : float
    contrastive_loss                : float
    random_seed                     : int
    early_stopping                  : boolean
    gpus                            : int
    date = CURRENT_TIMESTAMP        : timestamp
    """

@schema
class Np4Np(dj.Computed):
    definition = """
    -> Np4NpParam
    ---
    train_pearson  : float
    val_pearson     : float
    train_loss     : float
    val_loss       : float
    name           : varchar(200) 
    history        : longblob
    final_model    : longblob
    """

    def _make_tuples(self, key, ):
        import tensorflow as tf
        import numpy as np
        import os
        import datetime

        if not tf.executing_eagerly():
            tf.enable_eager_execution()

        from tuning_manifold.dataset_sampler import dataset_sampler
        from tuning_manifold.synthetic_sampler import dataset_sampler as synthetic_sampler, kernel_generator
        from tuning_manifold.fnp_model import Predictor
        from tuning_manifold.util import pearson, negloglik, EarlyStoppingNan, make_image_grid

        tfk = tf.keras

        neurons, stimuli, synthetic, synthetic_rf_dim, sta_feature, learning_rate, l2_weight, early_stopping, random_seed = (Np4NpParam() & key)\
            .fetch1('neurons', 'stimuli', 'synthetic', 'synthetic_rf_dim', 'sta_feature', 'learning_rate', 'l2_weight', 'early_stopping', 'random_seed')
        params = (Np4NpParam() & key).fetch(as_dict=True)[0]

        print(f'Neurons: {neurons}')
        print(f"Cell Latent: {params['cell_latent']}")
        print(f"Architecture: {params['architecture']}")
        print(f"Image Shape: {params['image_shape']}")
        print(f'Stimuli: {stimuli}')
        print(f'Synthetic: {synthetic}')
        print(f'Synthetic RF dim: {synthetic_rf_dim}')
        print(f'STA Feature: {sta_feature}')
        print(f'Samples: {params["samples"]}')
        print(f'Learning rate: {learning_rate}')
        print(f'L2 weight: {l2_weight}')
        print(f"Activity weight: {params['activity_weight']}")
        print(f"Laplace weight: {params['laplace_weight']}")
        print(f'Early stopping: {early_stopping}')
        print(f"Cummulative: {params['cummulative']}")
        print(f"Contrastive Loss: {params['contrastive_loss']}")

        _tuple = key.copy()

        logical_gpus = len(tf.config.experimental.list_logical_devices('GPU'))
        if logical_gpus > 1:
            neurons = int(params['neurons'] / logical_gpus) or 1
            print(f"{logical_gpus}-GPU training. split {params['neurons']} neurons into mini-batches of {neurons}")
        else:
            neurons = params['neurons']

        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        # set default paths by figuring out if we are running in a docker

        log_path = '/logs/'
        data_path = '/data'
        model_path = '/models/'

        # Catch when we are not running in docker
        if not os.path.isdir(log_path):
            log_path = '/home/jcotton/projects/tuning_manifold/logs/'
        if not os.path.isdir(data_path):
            data_path = '/home/jcotton/projects/tuning_manifold/data/'
        if not os.path.isdir(model_path):
            model_path = '/home/jcotton/projects/tuning_manifold/models/'

        mean_true_likelihood = 0

        if synthetic:
            training_data, validation_data, image_shape = synthetic_sampler(neurons=neurons,
                                                                            n_stimuli=stimuli,
                                                                            dim_rf=synthetic_rf_dim,
                                                                            image_shape=eval(params['image_shape']))

            likelihoods = []
            import itertools
            for data in itertools.islice(training_data, 10):
                import tensorflow_probability as tfp
                tfd = tfp.distributions
                tfpl = tfp.layers

                inputs, outputs = data
                responses, _, _, mean_response = inputs

                true_dist = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.Poisson(t))(mean_response + 1e-7)
                true_ll = negloglik(outputs, true_dist)

                likelihoods.append(true_ll)
            mean_true_likelihood = np.mean(tf.concat(likelihoods, axis=0))
            print(f'Mean True Likelihood: {mean_true_likelihood}')
        else:
            training_data, validation_data, image_shape = dataset_sampler(data_path, neurons=neurons,
                                                                          n_stimuli=stimuli)

        training_data = training_data.batch(logical_gpus)
        validation_data = validation_data.batch(logical_gpus)

        sampled_per_minibatch = neurons * params['stimuli'] * logical_gpus
        print(f'Neurons: {neurons}. Stimuli {stimuli}. Logical GPU: {logical_gpus}. Samples per minibatch: {sampled_per_minibatch}')
        scaled_learning_rate = learning_rate * sampled_per_minibatch / (1024.0 * 16.0)
        print(f'Scaled learning rate is {scaled_learning_rate}')

        if logical_gpus == 1:
            class Stub():
                def scope(self):
                    import contextlib
                    return contextlib.suppress()
            mirrored_strategy = Stub()

        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            inputs = [tfk.Input([stimuli, neurons], name='responses'), tfk.Input([stimuli, *image_shape], name='stimuli')]
            predictor = Predictor(architecture=eval(params['architecture']),
                                  l2_weight=l2_weight, cell_latent_dim=params['cell_latent'],
                                  sta_feature=sta_feature, cummulative=params['cummulative'],
                                  activity_weight=params['activity_weight'], laplace_weight=params['laplace_weight'],
                                  contrastive_weight=params['contrastive_loss'],
                                  samples=params["samples"])
            pred = predictor(inputs)

            model = tfk.Model(inputs, pred)
            optimizer = tf.optimizers.Adam(learning_rate=scaled_learning_rate) #, clipnorm=10)

            model.compile(loss=lambda x, y: negloglik(x, y) - mean_true_likelihood,
                          metrics=[pearson, 'mse'], optimizer=optimizer)

        predictor.im_conv_wrapper.operator.summary()
        predictor.location_predictor.summary()
        predictor.summary()
        model.summary()

        time_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        architecture = eval(params['architecture'])
        if type(architecture) == dict:
            architecture = architecture['architecture']
        name = "latent{}_rfDim{}_neurons{}_stimuli{}_architecture{}_sta{}_cummulative{}_lr{}_l2{}_cl{}_is{}_{}".format(params['cell_latent'],
                                                                                                    synthetic_rf_dim,
                                                                                                    params['neurons'],
                                                                                                    stimuli,
                                                                                                    architecture,
                                                                                                    sta_feature,
                                                                                                    params['cummulative'],
                                                                                                    learning_rate,
                                                                                                    l2_weight,
                                                                                                    params['contrastive_loss'],
                                                                                                    params['image_shape'],
                                                                                                    time_name)
        _tuple['name'] = name

        model_file_base = os.path.join(model_path, name + '_{epoch:02d}_{loss:.2f}_{pearson:.3f}.hdf5')
        log_file_dir = os.path.join(log_path, 'deterministic/{}'.format(name))

        cbs = []

        # Setup Learning Rate decay.
        # r_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        #    lambda epoch: learning_rate + (0.01 - learning_rate) * (0.5 ** epoch),
        #    verbose=True)
        # cbs.append(lr_decay_cb)

        tb = tf.keras.callbacks.TensorBoard(log_dir=log_file_dir)
        cbs.append(tb)

        if synthetic:
            file_writer_rfs = tf.summary.create_file_writer(log_file_dir + '/validation')
            plotting_neurons = 16
            plotting_data, _, _ = synthetic_sampler(neurons=plotting_neurons, n_stimuli=stimuli, dim_rf=synthetic_rf_dim, image_shape=eval(params['image_shape']))
            plotting_data = plotting_data.batch(1)
            validation_rfs = next(iter(plotting_data))

            def make_summary_fig(plotting_data, iterations=10):
                # Same as above but using the mean response from ground truth
                import io
                import matplotlib.pyplot as plt
                import tensorflow_probability as tfp

                tfpl = tfp.layers
                tfd = tfp.distributions

                deltas = []
                for i in range(iterations):
                    data = next(iter(plotting_data))
                    responses, stimuli, params, mean_response = data[0]

                    # this also seems to do what we expect
                    predicted_distribution = predictor((responses, stimuli))

                    model_ll = negloglik(responses, predicted_distribution)

                    true_dist = tfpl.DistributionLambda(make_distribution_fn=lambda t: tfd.Poisson(t))(mean_response + 1e-7)
                    true_ll = negloglik(data[1], true_dist)

                    delta_ll = model_ll - true_ll
                    deltas.append(delta_ll)

                delta = tf.concat(deltas, axis=-1)[0]
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

                grad_rf = make_image_grid(dy_dx, Ny=int(plotting_neurons / 4))
                grad_rf = tf.cast(tf.expand_dims(tf.expand_dims(grad_rf, 0), -1), tf.float16)
                true_rf = make_image_grid(list(kernel_generator(validation_rfs[0][2][0], size=image_shape)), Ny=int(plotting_neurons / 4))
                true_rf = tf.cast(tf.expand_dims(tf.expand_dims(true_rf, 0), -1), tf.float16)
                im = tf.concat([grad_rf, true_rf], axis=1)
                with file_writer_rfs.as_default():
                    tf.summary.image("RF Grad", im, step=epoch)

                im = make_summary_fig(plotting_data)
                with file_writer_rfs.as_default():
                    tf.summary.image("Excessive Uncertainty", im[None, ...], step=epoch)

            rf_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_rfs)
            # cbs.append(rf_callback)

        # send a keep-alive signal to the database connection
        def safe_ping(a, b):
            try:
                dj.conn.connection.ping()
            except:
                print("Ping failed!")
        ping_callback = tf.keras.callbacks.LambdaCallback(safe_ping)
        cbs.append(ping_callback)

        if True: # storing checkpoints
            ms = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_base, save_best_only=True, monitor='loss')
            cbs.append(ms)

        if early_stopping:
            # es = EarlyStoppingNan(patience=50, monitor='pearson')
            es = tf.keras.callbacks.EarlyStopping(patience=150, monitor='loss')
            cbs.append(es)

        history = model.fit(training_data,
                  epochs=3000, steps_per_epoch=300,
                  validation_data=validation_data, validation_steps=40,
                  callbacks=cbs,
                  verbose=1)
        print("Finished fitting")

        output_file = os.path.join(model_path, name + '.h5')
        model.save(output_file)

        print(history.history)

        _tuple['history'] = history.history
        with open(output_file, 'rb') as model_file:
            import base64
            _tuple['final_model'] = base64.b64encode(model_file.read())
        _tuple['train_pearson'] = history.history['pearson'][-1]
        _tuple['val_pearson'] = history.history['val_pearson'][-1]
        _tuple['train_loss'] = history.history['loss'][-1]
        _tuple['val_loss'] = history.history['val_loss'][-1]

        self.insert1(_tuple)


@schema
class Np4NpExcessiveUncertainty(dj.Computed):
    definition = """
    -> Np4Np
    ---
    final_ll       : float
    true_ll        : float
    excess_ll      : float
    ll_function    : longblob
    """

    def _make_tuples(self, key):
        from tuning_manifold.util import compute_excess, load_datajoint_model

        run_id = key['run_id']

        print(f'Computing excess uncertainty for {run_id}')
        model, data = load_datajoint_model(run_id, neurons=1, samples=1)

        _tuple = key.copy()
        _tuple.update(compute_excess(model, data, iterations=1000))
        self.insert1(_tuple)


@schema
class Np4NpFEV(dj.Computed):
    definition = """
    -> Np4Np
    ---
    mean_r         : float
    mean_fev       : float
    r              : longblob
    fev            : longblob
    """

    def _make_tuples(self, key):
        from tuning_manifold.util import compute_fev, load_datajoint_model

        run_id = key['run_id']
        print(f'Computing FEV for {run_id}')
        model, data = load_datajoint_model(run_id, neurons=1, samples=1)

        _tuple = key.copy()
        _tuple.update(compute_fev(model, data, iterations=1000))

        self.insert1(_tuple)


@schema
class Np4NpEpochs(dj.Computed):
    definition = """
    -> Np4Np
    ---
    epochs       : int
    """

    def _make_tuples(self, key):

        _tuple = key.copy()
        _tuple['epochs'] = len((Np4Np & key).fetch1('history')['loss'])
        self.insert1(_tuple)


def main():
    import os
    os.environ['NCCL_DEBUG'] = 'INFO'

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    terminated_jobs = (schema.jobs & 'status="error"' & 'error_message LIKE "System%%"')
    print(f'Terminated jobs: {terminated_jobs}')
    terminated_jobs.delete()

    n_gpus = len(gpus)

    Np4Np().populate(Np4NpParam() & f'gpus={n_gpus}', reserve_jobs=True)
    Np4NpExcessiveUncertainty().populate(reserve_jobs=True)
    Np4NpEpochs.populate(reserve_jobs=True)
    Np4NpFEV.populate(reserve_jobs=True)


if __name__ == "__main__":
    main()
