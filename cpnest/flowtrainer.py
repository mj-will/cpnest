from __future__ import division, print_function

import six
import os
import time
import copy
import logging
import json
import numpy as np
import corner
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy import stats as stats

import matplotlib.pyplot as plt

from .trainer import Trainer
from .flows import CouplingLayer, BatchNormFlow, FlowSequential
from .plot import plot_corner_contour


def weight_reset(m):
    """Reset parameters of a given model"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


class FlowModel(nn.Module):
    """
    Builds the sequential flow model with the provided inputs

    Based on SingleSpeed in: https://github.com/adammoss/nnest/blob/master/nnest/networks.py
    """

    def __init__(self, n_inputs=None, n_neurons=128, n_layers=2, n_blocks=4, device=None):
        super(FlowModel, self).__init__()

        if device is None:
            raise ValueError("Must provided a device or a string for a device")

        if type(device) == str:
            self.device = torch.device(device)
        else:
            self.device = device

        self.n_inputs = n_inputs
        mask = torch.remainder(torch.arange(0, n_inputs, dtype=torch.float, device=self.device), 2)

        layers = []
        for _ in range(n_blocks):
            layers += [CouplingLayer(n_inputs, n_neurons, mask, num_layers=n_layers), BatchNormFlow(n_inputs)]
            mask = 1 - mask

        self.net = FlowSequential(*layers)
        self.net.to(self.device)

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        return self.net(inputs, cond_inputs=cond_inputs, mode=mode, logdets=logdets)

    def log_probs(self, inputs, cond_inputs=None):
        return self.net.log_probs(inputs, cond_inputs=None)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        return self.net.sample(num_samples=num_samples, noise=noise, cond_inputs=cond_inputs)


class FlowTrainer(Trainer):

    def __init__(self, trainer_dict, manager=None, output='./'):
        self.manager=None
        super(FlowTrainer, self).__init__(manager=manager, output=output)
        self.outdir = output
        self.logger = logging.getLogger("CPNest")
        self.priors = None
        self.intialised = False
        self.normalise = False
        self.device_tag = 'cpu'
        # default training params
        self.lr = 0.0001
        self.val_size = 0.1
        self.batch_size = 100
        self.max_epochs = 1000
        self.patience = 100

        self._setup_from_input_dict(trainer_dict)

    def save_input(self, attr_dict):
        """Save the dictionary used as an inputs as a JSON file"""
        d = attr_dict.copy()
        output_file = self.outdir + "trainer_dict.json"
        for k, v in list(d.items()):
            if type(v) == np.ndarray:
                d[k] = np.array_str(d[k])
        with open(output_file, "w") as f:
            json.dump(d, f, indent=4)

    def _setup_from_input_dict(self, attr_dict):
        for key, value in six.iteritems(attr_dict):
            setattr(self, key, value)
        self.n_inputs = self.model_dict["n_inputs"]
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.save_input(attr_dict)

    def initialise(self):
        """
        Intialise the model and optimiser
        """
        self.device = torch.device(self.device_tag)
        self.model = FlowModel(device=self.device, **self.model_dict)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        if self.priors is not None:
            self.logger.info("Setting up normalisation")
            self.setup_normalisation()
        self.intialised = True
        self.training_count = 0
        if self.manager is not None:
            self.logger.info("Sending init confirmation")
            self.producer_pipe.send(1)

    def _reset_model(self):
        """Reset the weights and optimiser"""
        self.model.apply(weight_reset)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

    def setup_normalisation(self):
        """
        Setup normalisation using the priors
        """
        self._prior_max = np.max(self.priors, axis=0)
        self._prior_min = np.min(self.priors, axis=0)
        self.normalise = True

    def normalise_samples(self, x):
        """
        Normalise a set of samples
        """
        return 2. * ((x - self._prior_min) / (self._prior_max - self._prior_min)) - 1

    def rescale_sample(self, x):
        """
        Apply the inverse of the normalisation
        """
        return (self._prior_max - self._prior_min) * ((x + 1) / 2.) + self._prior_min


    def train(self, payload):
        """
        Training the flow given a payload of CPnest LivePoints
        """
        samples = np.array([p.values for p in payload])
        self.logger.info("Starting training setup")

        D, p_value = self._train_on_data(samples)

        self.logger.info("Training complete")

        # send weights and check whether to enable the flow
        if self.manager is not None:
            if D >= 0.01:
                self.manager.use_flow.value = 1
            self.manager.trained.value = 1
            self.producer_pipe.send(self.weights_file)
            self.logger.info("Weights sent")


    def _train_on_data(self, samples, plot=True):
        """
        Train the flow on samples
        """
        if not self.intialised:
            self.logger.info("Initialising")
            self.initialise()
        else:
            self._reset_model()
        if self.normalise:
            self.logger.info("Using normalisation")
            samples = self.normalise_samples(samples)

        block_outdir = "{}block{}/".format(self.outdir, self.training_count)

        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)
        if plot:
            #print("Flow trainer: plotting input")
            #plot_corner_contour(samples, filename=block_outdir + "input_samples.png")
            pass

        # setup data loading
        x_train, x_val = train_test_split(samples, test_size=self.val_size)
        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)

        # train
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        self.logger.info("Starting training")
        self.logger.info("Training parameters:")
        self.logger.info(f"Max. epochs: {self.max_epochs}")
        self.logger.info(f"Patience: {self.patience}")
        history = dict(loss=[], val_loss=[])
        for epoch in range(1, self.max_epochs + 1):

            loss = self._train(train_loader)
            val_loss = self._validate(val_loader)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 50:
                self.logger.info(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > self.patience:
                self.logger.info(f"Epoch {epoch}: Reached patience")
                break

        self.training_count += 1
        self.model.load_state_dict(best_model.state_dict())
        self.weights_file = block_outdir + 'model.pt'
        torch.save(self.model.state_dict(), self.weights_file)
        # sample for plots
        output = self.sample(N=len(samples))
        if plot:
            self.logger.info("Plotting output")
            fig = plt.figure()
            epochs = np.arange(1, epoch + 1, 1)
            plt.plot(epochs, history['loss'], label='Loss')
            plt.plot(epochs, history['val_loss'], label='Val. loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            fig.savefig(block_outdir + 'loss.png')
            plot_corner_contour([samples, output], labels=["Input data", "Generated data"], filename=block_outdir+"comparison.png")

        # compute mean KS
        D, p_value = self.compute_mean_ks(samples, output)
        self.logger.info(f"Computed KS - D: {D}, p: {p_value}")
        return D, p_value


    def _train(self, loader, noise_scale=0.):
        """Loop over the data and update the weights"""
        model = self.model
        model.train()
        train_loss = 0

        for idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                data = data[0]
            data = (data + noise_scale * torch.randn_like(data)).to(self.device)
            self.optimiser.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            loss.backward()
            self.optimiser.step()

            for module in model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 0

            with torch.no_grad():
                model(loader.dataset.tensors[0].to(data.device))

            for module in model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 1

            return train_loss / len(loader.dataset)

    def _validate(self, loader):
        """Loop over the data and get validation loss"""
        model = self.model
        model.eval()
        val_loss = 0

        for idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                val_loss += -model.log_probs(data).mean().item()

            return val_loss / len(loader.dataset)

    def sample(self, N=1000):
        """Produce N samples drawn from the latent space"""
        z = torch.randn(N, self.n_inputs, device=self.device)
        output, _ = self.model(z, mode='inverse')

        z = z.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        return output

    def load_weights(self, weights_file):
        """Load weights for the model"""
        if not self.initialised:
            self.initialise()
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def compute_mean_ks(self, samples, output):
        """Compute the KS over each dimension and take the mean"""
        samples = np.array(samples)
        output = np.array(output)
        n_dims = samples.shape[-1]
        D_values = np.empty(n_dims,)
        p_values = np.empty(n_dims,)
        for i, t, p in zip(range(n_dims), samples.T, output.T):
            D_values[i], p_values[i] = stats.ks_2samp(*[t, p])
        return D_values.mean(), p_values.mean()
