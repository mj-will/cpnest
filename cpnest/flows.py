"""
FlowSequential, CouplingLayer and BatchNorm from: https://github.com/ikostrikov/pytorch-flows
Modified to use multiple hidden layers

Other classes
"""

import math
import types

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def plot_flows(model, n_inputs, N=1000, inputs=None, cond_inputs=None, mode='inverse', output='./'):
    """
    Plot each stage of a series of flows
    """
    import matplotlib.pyplot as plt

    if n_inputs > 2:
        raise NotImplementedError('Plotting for higher dimensions not implemented yet!')

    outputs = []

    if mode == 'direct':
        if inputs is None:
            raise ValueError('Can not sample from parameter space!')
        else:
            inputs = torch.from_numpy(inputs).to(model.device)

        for module in model._modules.values():
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())
    else:
        if inputs is None:
            inputs  = torch.randn(N, n_inputs, device=model.device)
            orig_inputs = inputs.detach().cpu().numpy()
        for module in reversed(model._modules.values()):
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())

    print(len(outputs))

    n = int(len(outputs) / 2) + 1
    m = 1

    if n > 5:
        m = int(np.ceil(n / 5))
        n = 5

    z = orig_inputs
    pospos = np.where(np.all(z>=0, axis=1))
    negneg = np.where(np.all(z<0, axis=1))
    posneg = np.where((z[:, 0] >= 0) & (z[:, 1] < 0))
    negpos = np.where((z[:, 0] < 0) & (z[:, 1] >= 0))

    points = [pospos, negneg, posneg, negpos]
    colours = ['r', 'c', 'g', 'tab:purple']
    colours = plt.cm.Set2(np.linspace(0, 1, 8))


    fig, ax = plt.subplots(m, n, figsize=(n * 3, m * 3))
    ax = ax.ravel()
    for j, c in zip(points, colours):
        ax[0].plot(z[j, 0], z[j, 1], ',', c=c)
        ax[0].set_title('Latent space')
    for i, o in enumerate(outputs[::2]):
        i += 1
        for j, c in zip(points, colours):
            ax[i].plot(o[j, 0], o[j, 1], ',', c=c)
        ax[i].set_title(f'Flow {i}')
        #ax[i].plot(o[:, 0], o[:, 1], ',')
    plt.tight_layout()
    fig.savefig(output + 'flows.png')



def setup_model(n_inputs=None,  n_conditional_inputs=None, n_neurons=128, n_layers=2, n_blocks=4, ftype='RealNVP', device='cpu'):
    """"
    Setup the model
    """
    if device is None:
        raise ValueError("Must provided a device or a string for a device")
    if type(device) == str:
        device = torch.device(device)

    layers = []
    ftype = ftype.lower()
    if ftype == 'realnvp':
        mask = torch.remainder(torch.arange(0, n_inputs, dtype=torch.float, device=device), 2)
        for _ in range(n_blocks):
            layers += [CouplingLayer(n_inputs, n_neurons, mask,
                                     num_cond_inputs=n_conditional_inputs,
                                     num_layers=n_layers),
                       BatchNormFlow(n_inputs)]
            #layers += [MADE(n_inputs, n_neurons)]
            mask = 1 - mask
    elif ftype == 'maf':
        for _ in range(n_blocks):
            layers += [
                MADE(n_inputs, n_neurons, n_conditional_inputs),
                BatchNormFlow(n_inputs),
                Reverse(n_inputs)
            ]
    else:
        raise ValueError('Unknown flow type, choose from RealNPV or MAF')

    model = FlowSequential(*layers)
    model.to(device)
    model.device = device
    return model

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 num_layers=2,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        scale_layers = [nn.Linear(total_inputs, num_hidden), s_act_func()]
        for i in range(0, num_layers):
            scale_layers += [nn.Linear(num_hidden, num_hidden), s_act_func()]
        scale_layers += [nn.Linear(num_hidden, num_inputs)]
        self.scale_net = nn.Sequential(*scale_layers)
        translate_layers = [nn.Linear(total_inputs, num_hidden), t_act_func()]
        for i in range(0, num_layers):
            translate_layers += [nn.Linear(num_hidden, num_hidden), t_act_func()]
        translate_layers += [nn.Linear(num_hidden, num_inputs)]
        self.translate_net = nn.Sequential(*translate_layers)

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples


def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))


class Transform(nn.Module):
    """
    Basic neural net to perform encoding or decoding
    """
    def __init__(self, in_dim, out_dim, n_neurons):
        super(Transform, self).__init__()

        self.linear1 = nn.Linear(in_dim, n_neurons)
        self.linear2 = nn.Linear(n_neurons, n_neurons)
        self.linear3 = nn.Linear(n_neurons, 2 * out_dim)

        with torch.no_grad():
            self.linear3.weight.data.fill_(0.)
            self.linear3.bias.data.fill_(0.)

    def forward(self, x):
        """
        Forward pass
        """
        x = self.linear1(x)
        x = swish(x)
        x = self.linear2(x)
        x = swish(x)
        x = self.linear3(x)
        x_s, x_m = torch.split(x, x.shape[-1] // 2, dim=1)
        x_s = torch.nn.functional.logsigmoid(x_s)
        x_s = torch.clamp(x_s, min=-2.5, max=2.5)
        return x_s, x_m


class AugmentedBlock(nn.Module):
    """
    An implementation on augmented flow blocks

    See: https://arxiv.org/abs/2002.07101
    """
    def __init__(self, x_dim, e_dim, n_layers=2, n_neurons=64):
        """
        Intialise the block
        """
        super(AugmentedBlock, self).__init__()
        self.encoder = Transform(x_dim, e_dim, n_neurons)
        self.decoder = Transform(e_dim, x_dim, n_neurons)

    def forward(self, feature, augment, mode='forward'):
        """
        Forward or backwards pass
        """
        log_J = 0.
        if mode == 'forward':
            # encode e -> z | x
            log_s, m = self.encoder(feature)
            s = torch.exp(log_s)
            z = s * augment + m
            log_J += log_s.sum(-1, keepdim=True)
            # decode x -> y | z
            log_s, m = self.decoder(z)
            s = torch.exp(log_s)
            y = s * feature + m
            log_J += log_s.sum(-1, keepdim=True)
            return y, z, log_J
        else:
            # decode y -> x | z
            log_s, m = self.decoder(augment)
            s = torch.exp(-log_s)
            x = s * (feature - m)
            log_J -= log_s.sum(-1, keepdim=True)
            # encode z -> e | z
            log_s, m = self.encoder(x)
            s = torch.exp(-log_s)
            e = s * (augment - m)
            log_J -= log_s.sum(-1, keepdim=True)
            return x, e, log_J


class AugmentedSequential(nn.Sequential):
    """
    A sequential container for augmented flows
    """
    def forward(self, feature, augment, mode='forward'):
        """
        Forward or backward pass through the flows
        """
        log_dets = torch.zeros(feature.size(0), 1, device=feature.device)
        if mode == 'forward':
            for module in self._modules.values():
                feature, augment, log_J = module(feature, augment, mode)
                log_dets += log_J
        else:
            for module in reversed(self._modules.values()):
                feature, augment, log_J = module(feature, augment, mode)
                log_dets += log_J

        return feature, augment, log_dets

    def log_N(self, x):
        """
        Calculate of the log probability of an N-dimensional gaussian
        """
        return (-0.5 * x.pow(2) - 0.5 * np.log(2 * np.pi)).sum(
            -1, keepdim=True)

    def log_p_xe(self, feature, augment):
        """
        Calculate the joint log probability p(x, e)
        """
        # get transformed features
        y, z, log_J = self(feature, augment)
        # y & z should be gaussian
        y_prob = self.log_N(y)
        z_prob = self.log_N(z)
        return (y_prob + z_prob + log_J).sum(-1, keepdim=True)

    def log_p_x(self, feature, e_dim, K=1000):
        """
        Calculate the lower bound of the marginalised log probability p(x)
        """
        log_p_x = torch.zeros(feature.size(0), 1, device=feature.device)
        # get log p(x, e)
        for i,f in enumerate(feature):
            e = torch.Tensor(K, e_dim).normal_().to(feature.device)
            log_q = self.log_N(e)
            # need to pass the same feature K times (for each e)
            f_repeated = f * torch.ones(K, f.size(0), device=feature.device)
            log_p_xe = self.log_p_xe(f_repeated, e)
            # compute sum of ratio
            lpx = -np.log(K) + torch.logsumexp(log_p_xe - log_q, (0))
            log_p_x[i] = lpx
        return log_p_x


def setup_augmented_model(n_inputs=None,  augment_dim=None, n_conditional_inputs=None, n_neurons=32, n_layers=2, n_blocks=4, ftype='RealNVP', device='cpu', **kwargs):
    """"
    Setup the model with augmented flows
    """
    if device is None:
        raise ValueError("Must provided a device or a string for a device")
    if type(device) == str:
        device = torch.device(device)

    if n_conditional_inputs is not None:
        raise NotImplementedError('Augmented flows are not implemented for conditional inputs')

    layers = []
    ftype = ftype.lower()
    if ftype == 'realnvp':
        blocks = []
        for n in range(n_blocks):
            blocks += [AugmentedBlock(n_inputs, augment_dim, n_layers, n_neurons)]
    else:
        raise ValueError('Unknown flow type, choose from RealNVP')

    model = AugmentedSequential(*blocks).to('cuda')
    model.to(device)
    model.augment_dim = augment_dim
    model.device = device
    return model
