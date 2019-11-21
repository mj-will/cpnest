from __future__ import division, print_function

import six
import os
import time
import copy
import numpy as np
import corner
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .trainer import Trainer
from .flows import CouplingLayer, BatchNormFlow, FlowSequential
from .plot import plot_corner_contour


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
        self.priors = None
        self.intialised = False
        self.normalise = False
        self.device_tag = 'cpu'
        self._setup_from_input_dict(trainer_dict)

        # default training params
        self.lr = 0.0001
        self.val_size = 0.1
        self.batch_size = 100
        self.max_epochs = 1000
        self.patience = 100

    def _setup_from_input_dict(self, attr_dict):
        for key, value in six.iteritems(attr_dict):
            setattr(self, key, value)
        self.device = torch.device(self.device_tag)
        self.n_inputs = self.model_dict["n_inputs"]
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        # setup device

    def initialise(self):
        """
        Intialise the model and optimiser
        """
        self.model = FlowModel(device=self.device, **self.model_dict)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        if self.priors is not None:
            print("Flow trainer: setting up normalisation")
            self.setup_normalisation()
        self.intialised = True
        self.training_count = 0

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
        return (x - self._prior_min) / (self._prior_max - self._prior_min)


    def train(self, payload):
        """
        Training the flow given a payload of CPnest LivePoints
        """
        samples = []
        for p in payload:
            samples.append(p.values)
        samples = np.array(samples)

        self._train_on_data(samples)

    def _train_on_data(self, samples):
        """
        Train the flow on samples
        """
        if not self.intialised:
            self.initialise()

        if self.normalise:
            print("Using normalisation")
            samples = self.normalise_samples(samples)

        block_outdir = "{}block{}/".format(self.outdir, self.training_count)

        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)
        plot_corner_contour(samples, filename=block_outdir + "input_samples.png")
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

        for epoch in range(1, self.max_epochs + 1):

            loss = self._train(train_loader)
            val_loss = self._validate(val_loader)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 50:
                print(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > self.patience:
                print(f"Epoch {epoch}: Reached patience")
                break
        self.training_count += 1
        self.model.load_state_dict(best_model.state_dict())
        self.weights_file = block_outdir + 'model.pt'
        torch.save(self.model.state_dict(), self.weights_file)
        # sample for plots
        output = self.sample(N=5000)
        if self.manager is not None:
            self.manager.trained.value = 1

        plot_corner_contour([samples, output], labels=["Input data", "Generated data"], filename=block_outdir+"comparison.png")

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
