from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from torch import nn
from torch.nn import functional as F
from neuralpredictors.layers.cores import DepthSeparableConv2d, Core2d
from neuralpredictors.layers.readouts import SpatialXFeatureLinear
import torch
from collections import OrderedDict, Iterable
from neuralpredictors import regularizers


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


# Squeeze and Excitation Block
class SQ_EX_Block(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SQ_EX_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class MultiReadout:

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_noise, bias, normalize, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, SpatialXFeatureLinear(
                in_shape=in_shape,
                outdims=n_neurons,
                init_noise=init_noise,
                bias=bias,
                normalize=normalize
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


class SE2dCore(Core2d, nn.Module):
    def __init__(
            self,
            input_channels,
            hidden_channels,
            input_kern,
            hidden_kern,
            layers=3,
            gamma_input=0.0,
            skip=0,
            final_nonlinearity=True,
            bias=False,
            momentum=0.1,
            pad_input=True,
            batch_norm=True,
            hidden_dilation=1,
            laplace_padding=None,
            input_regularizer="LaplaceL2norm",
            stack=None,
            se_reduction=32,
            n_se_blocks=1,
            depth_separable=False,
            linear=False,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            bias:           Adds a bias layer. Note: bias and batch_norm can not both be true
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                neuralpredictors.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            stack = -1 will only select the last layer as the readout layer
                            stack = 0  will only readout from the first layer
            se_reduction:   Int. Reduction of Channels for Global Pooling of the Squeeze and Excitation Block.
        """

        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.features = nn.Sequential()
        self.n_se_blocks = n_se_blocks
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [*range(self.layers)[stack:]] if isinstance(stack, int) else stack

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels, hidden_channels, input_kern, padding=input_kern // 2 if pad_input else 0, bias=bias
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if (layers > 1 or final_nonlinearity) and not linear:
            layer["nonlin"] = nn.ELU(inplace=True)
        self.features.add_module("layer0", nn.Sequential(layer))

        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            hidden_padding = ((hidden_kern[l - 1] - 1) * hidden_dilation + 1) // 2
            if depth_separable:
                layer["ds_conv"] = DepthSeparableConv2d(hidden_channels, hidden_channels,
                                                        kernel_size=hidden_kern[l - 1],
                                                        dilation=hidden_dilation, padding=hidden_padding, bias=False,
                                                        stride=1)
            else:
                layer["conv"] = nn.Conv2d(
                    hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                    hidden_channels,
                    hidden_kern[l - 1],
                    padding=hidden_padding,
                    bias=bias,
                    dilation=hidden_dilation,
                )
            if batch_norm:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)

            if (final_nonlinearity or l < self.layers - 1) and not linear:
                layer["nonlin"] = nn.ELU(inplace=True)

            if (self.layers - l) <= self.n_se_blocks:
                layer["seg_ex_block"] = SQ_EX_Block(in_ch=hidden_channels, reduction=se_reduction)

            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            if l in self.stack:
                ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def regularizer(self):
        return self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels

def se_core_spatialXfeature_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                                    hidden_kern=3, layers=3, gamma_input=15.5,
                                    skip=0, final_nonlinearity=True, momentum=0.9,
                                    pad_input=False, batch_norm=True, hidden_dilation=1,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    init_noise=1e-3, readout_bias=True,  # readout args,
                                    gamma_readout=4, normalize=False, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                                    depth_separable=False, linear=False):
    """
    Model class of a stacked2dCore (from neuralpredictors) and a spatialXfeature (factorized) readout

    Args:

    Returns: An initialized model which consists of model.core and model.readout
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            x = self.readout(x, data_key=data_key,)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    # get a stacked2D core from neuralpredictors
    core = SE2dCore(input_channels=input_channels[0],
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear)

    readout = MultipleSpatialXFeatureLinear(core, in_shape_dict=in_shapes_dict,
                                            n_neurons_dict=n_neurons_dict,
                                            init_noise=init_noise,
                                            bias=readout_bias,
                                            gamma_readout=gamma_readout,
                                            normalize=normalize
                                            )

    # initializing readout bias to mean response
    if readout_bias:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model
