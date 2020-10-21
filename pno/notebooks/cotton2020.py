from collections import OrderedDict
from itertools import zip_longest

from neuralpredictors.data.datasets import FileTreeDataset
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import Subsample, ToTensor, NeuroNormalizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from neuralpredictors.data.samplers import RepeatsBatchSampler

from neuralpredictors.training import device_state
import types
import contextlib

from functools import partial

from tqdm import tqdm

from neuralpredictors.measures import *
from neuralpredictors import measures as mlmeasures
from neuralpredictors.training import early_stopping, MultipleObjectiveTracker, eval_state, LongCycler
from nnfabrik.utility.nn_helpers import set_random_seed

from measures import get_correlations, get_poisson_loss
import measures


def nnvision_trainer(model, dataloaders, seed, avg_loss=False, scale_loss=True,  # trainer args
                                loss_function='PoissonLoss', stop_function='get_correlations',
                                loss_accum_batch_n=None, device='cuda', verbose=True,
                                interval=1, patience=5, epoch=0, lr_init=0.005,  # early stopping args
                                max_iter=100, maximize=True, tolerance=1e-6,
                                restore_best=True, lr_decay_steps=3,
                                lr_decay_factor=0.3, min_lr=0.0001,  # lr scheduler args
                                cb=None, track_training=False, return_test_score=False, **kwargs):
    """

    Args:
        model:
        dataloaders:
        seed:
        avg_loss:
        scale_loss:
        loss_function:
        stop_function:
        loss_accum_batch_n:
        device:
        verbose:
        interval:
        patience:
        epoch:
        lr_init:
        max_iter:
        maximize:
        tolerance:
        restore_best:
        lr_decay_steps:
        lr_decay_factor:
        min_lr:
        cb:
        track_training:
        **kwargs:

    Returns:

    """

    def full_objective(model, dataloader, data_key, *args):
        """

        Args:
            model:
            dataloader:
            data_key:
            *args:

        Returns:

        """
        loss_scale = np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0]) if scale_loss else 1.0
        return loss_scale * criterion(model(args[0].to(device), data_key), args[1].to(device)) \
               + model.regularizer(data_key)

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(getattr(measures, stop_function), dataloaders=dataloaders["validation"], device=device, per_neuron=False, avg=True)

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min',
                                                           factor=lr_decay_factor, patience=patience,
                                                           threshold=tolerance,
                                                           min_lr=min_lr, verbose=verbose, threshold_mode='abs')

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = len(dataloaders["train"].keys()) if loss_accum_batch_n is None else loss_accum_batch_n

    if track_training:
        tracker_dict = dict(correlation=partial(get_correlations(), model, dataloaders["validation"], device=device, per_neuron=False),
                            poisson_loss=partial(get_poisson_loss(), model, dataloaders["validation"], device=device, per_neuron=False, avg=False))
        if hasattr(model, 'tracked_values'):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    for epoch, val_obj in early_stopping(model, stop_closure, interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best, tracker=tracker,
                                         scheduler=scheduler, lr_decay_steps=lr_decay_steps):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(enumerate(LongCycler(dataloaders["train"])), total=n_iterations,
                                               desc="Epoch {}".format(epoch)):

            loss = full_objective(model, dataloaders["train"], data_key, *data)
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False)
    test_correlation = get_correlations(model, dataloaders["test"], device=device, as_dict=False, per_neuron=False)

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(test_correlation) if return_test_score else np.mean(validation_correlation)
    return score, output, model.state_dict()



def neurips_loader(path,
                        batch_size,
                        seed=None,
                        areas=None,
                        layers=None,
                        tier=None,
                        neuron_ids=None,
                        get_key=False,
                        cuda=True,
                        normalize=True,
                        exclude=None,
                        trial_ids=None,
                        **kwargs):
    dat = FileTreeDataset(path,  'images',  'responses')
    # The permutation MUST be added first and the conditions below MUST NOT be based on the original order

    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    conds = np.ones(len(dat.neurons.area), dtype=bool)
    if areas is not None:
        conds &= (np.isin(dat.neurons.area, areas))
    if layers is not None:
        conds &= (np.isin(dat.neurons.layer, layers))
    if neuron_ids is not None:
        conds &= (np.isin(dat.neurons.unit_ids, neuron_ids))

    idx = np.where(conds)[0]
    more_transforms = [Subsample(idx), ToTensor(cuda)]
    if normalize:
        more_transforms.insert(0, NeuroNormalizer(dat, exclude=exclude))

    dat.transforms.extend(more_transforms)

    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ['train', 'validation', 'test']
    for tier in keys:

        if seed is not None:
            set_random_seed(seed)
            # torch.manual_seed(img_seed)

        # sample images
        conds = np.ones_like(dat.trial_info.tiers).astype(bool)
        conds &= (dat.trial_info.tiers == tier)
        
        if trial_ids is not None and tier in trial_ids:
            print(f'Subsampling {tier} set to {len(trial_ids[tier])} trials', flush=True)
            conds &= (np.isin(dat.trial_info.trial_idx, trial_ids[tier]))
            
        subset_idx = np.where(conds)[0] 
        if tier == 'train':
            sampler = SubsetRandomSampler(subset_idx) 
            dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)
        elif tier == 'validation':
            sampler = SubsetSequentialSampler(subset_idx)
            dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)
        else:
            sampler = RepeatsBatchSampler(dataloaders['train'].dataset.trial_info.frame_image_id, subset_index=subset_idx)
            dataloaders[tier] = DataLoader(dat, batch_sampler=sampler, sampler=None)

    # create the data_key for a specific data path
    data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')
    return (data_key, dataloaders) if get_key else dataloaders


def neurips_loaders(paths,
                         batch_size,
                         seed=None,
                         areas=None,
                         layers=None,
                         tier=None,
                         neuron_ids=None,
                         cuda=True,
                         normalize=False,
                         exclude=None,
                         trial_ids=None,
                         **kwargs):

    neuron_ids = neuron_ids if neuron_ids is not None else []
    areas = areas if areas is not None else ('V1',)
    dls = OrderedDict({})
    keys = [tier] if tier else ['train', 'validation', 'test']
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip_longest(paths, neuron_ids, fillvalue=None):
        data_key, loaders = neurips_loader(path, batch_size, seed=seed,
                                                areas=areas, layers=layers, cuda=cuda,
                                                tier=tier, get_key=True, neuron_ids=neuron_id,
                                                normalize=normalize, exclude=exclude, trial_ids=trial_ids)
        for k in dls:
            dls[k][data_key] = loaders[k]

    return dls


def model_predictions_repeats(dataloader, model, data_key, device='cpu', broadcast_to_target=False):
    """
    Computes model predictions for dataloader that yields batches with identical inputs along the first dimension
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated 
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """
    
    output = torch.empty(0)
    target = []
    
    # Get unique images and concatenate targets:
    unique_images = torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        
        assert torch.all(torch.eq(images[-1,], images[0,],)), "All images in the batch should be equal"
        unique_images = torch.cat((unique_images, images[0:1, ]), dim=0)
        target.append(responses.detach().cpu().numpy())
    
    # Forward unique images once:
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        with device_state(model, device):
            output = model(unique_images.to(device), data_key=data_key).detach().cpu()
    
    output = output.numpy()   
        
    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]
    
    
    return target, output
    

def model_predictions(dataloader, model, data_key, device='cpu'):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat((output, (model(images.to(device), data_key=data_key).detach().cpu())), dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()



def get_avg_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and average responses over repeated trials
    
    """
    if 'test' in dataloaders:
        dataloaders = dataloaders['test']
    
    correlations = {}
    for k, loader in dataloaders.items():

        # Compute correlation with average targets
        target, output = model_predictions_repeats(dataloader=loader, model=model, data_key=k, device=device, broadcast_to_target=False)
        target_mean = np.array([t.mean(axis=0) for t in target])
        correlations[k] = corr(target_mean, output, axis=0)
        
        # Check for nans
        if np.any(np.isnan(correlations[k])):
            warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    correlations = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            correlations[k] = corr(target, output, axis=0)

            if np.any(np.isnan(correlations[k])):
                warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
            correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_poisson_loss(model, dataloaders, device='cpu', as_dict=False, avg=False, per_neuron=False, eps=1e-12):
    poisson_loss = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            loss = output - target * np.log(output + eps)
            poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return np.mean(np.hstack([v for v in poisson_loss.values()])) if avg else np.sum(np.hstack([v for v in poisson_loss.values()]))


def get_repeats(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for inputs, outputs in dataloader:
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if r < min_repeats:  # minimum number of frame repeats to be considered for oracle, free choice
            continue
        assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_oracles(dataloaders, as_dict=False):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr(np.array(outputs))
    return oracles if as_dict else np.hstack([v for v in oracles.values()])


def compute_oracle_corr(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r) * r / (r - 1)
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)
            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def get_explainable_var(dataloaders, as_dict=False):
    explainable_var = {}
    for k ,v in dataloaders.items():
        _, outputs = get_repeats(v)
        explainable_var[k] = compute_explainable_var(outputs)
    return explainable_var if as_dict else np.hstack([v for v in explainable_var.values()])


def compute_explainable_var(outputs):
    ImgVariance = []
    TotalVar = np.var(np.vstack(outputs), axis=0, ddof=1)
    for out in outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / TotalVar
    return explainable_var


def get_FEV(dataloaders, model, device='cpu', as_dict=False):
    FEV = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            repeated_inputs, repeated_outputs = get_repeats(v)
            FEV[k] = compute_FEV(repeated_inputs=repeated_inputs,
                                 repeated_outputs=repeated_outputs,
                                 model=model,
                                 device=device,
                                 data_key=k,
                                 )
    return FEV if as_dict else np.hstack([v for v in FEV.values()])


def compute_FEV(repeated_inputs, repeated_outputs, model, data_key=None, device='cpu', return_exp_var=False):

    ImgVariance = []
    PredVariance = []
    for i, outputs in enumerate(repeated_outputs):
        inputs = repeated_inputs[i]
        predictions = model(torch.tensor(inputs).to(device), data_key=data_key).detach().cpu().numpy()
        PredVariance.append((outputs - predictions) ** 2)
        ImgVariance.append(np.var(outputs, axis=0, ddof=1))

    PredVariance = np.vstack(PredVariance)
    ImgVariance = np.vstack(ImgVariance)

    TotalVar = np.var(np.vstack(repeated_outputs), axis=0, ddof=1)
    NoiseVar = np.mean(ImgVariance, axis=0)
    FEV = (TotalVar - NoiseVar) / TotalVar

    PredVar = np.mean(PredVariance, axis=0)
    FEVe = 1 - (PredVar - NoiseVar) / (TotalVar - NoiseVar)
    return [FEV, FEVe] if return_exp_var else FEVe


def get_cross_oracles(data, reference_data):
    _, outputs = get_repeats(data)
    _, outputs_reference = get_repeats(reference_data)
    cross_oracles = compute_cross_oracles(outputs, outputs_reference)
    return cross_oracles


def compute_cross_oracles(repeats, reference_data):
    pass


def normalize_RGB_channelwise(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min(axis=(1, 2), keepdims=True)
    mei_copy = mei_copy / mei_copy.max(axis=(1, 2), keepdims=True)
    return mei_copy


def normalize_RGB(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min()
    mei_copy = mei_copy / mei_copy.max()
    return mei_copy
