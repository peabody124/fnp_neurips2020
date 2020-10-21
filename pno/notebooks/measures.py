import warnings
import numpy as np
from neuralpredictors.measures import corr
from neuralpredictors.training import eval_state, device_state
import types
import contextlib
import torch

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

