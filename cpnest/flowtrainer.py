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
import hdbscan

import matplotlib.pyplot as plt

from .trainer import Trainer
from .flows import CouplingLayer, BatchNormFlow, FlowSequential, MADE, setup_model, setup_augmented_model
from .plot import plot_corner_contour


def logistic(x):
   return 1. / (1. + np.exp(-x))

def logit(x):
    return - np.log((1. / x) - 1.)

def weight_reset(m):
    """
    Reset parameters of a given model
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

def plot_loss(epoch, history, output='./'):
    """
    Plot a loss function
    """
    fig = plt.figure()
    epochs = np.arange(1, epoch + 1, 1)
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val. loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative log-likelihood')
    plt.legend()
    plt.tight_layout()
    fig.savefig(output + 'loss.png')
    plt.yscale('log')
    fig.savefig(output + 'loss_log.png')
    plt.close('all')

def plot_samples(z, samples, output='./', filename='output_samples.png'):
    """
    Plot the samples in the latent space and parameter space
    """
    N = samples.shape[0]
    d = samples.shape[-1]

    samples = samples[np.isfinite(samples).all(axis=1)]

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d >= 2:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].plot(samples[:,j], samples[:,i], marker=',', linestyle='')
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                else:
                    ax[i, j].plot(z[:,j], z[:,i], marker=',', linestyle='')
    else:
        ax.hist(samples, int(np.sqrt(N)), histtype='step')
    plt.tight_layout()
    fig.savefig(output + filename)
    plt.close('all')

def plot_inputs(samples, output='./', filename='input_samples.png'):
    """
    Plot n-dimensional input samples
    """
    N = samples.shape[0]
    d = samples.shape[-1]

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d > 1:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].plot(samples[:,j], samples[:,i], marker=',', linestyle='')
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                else:
                    ax[i, j].set_axis_off()
    else:
        ax.hist(samples, int(np.sqrt(N)), histtype='step')
    plt.tight_layout()
    fig.savefig(output + filename ,bbox_inches='tight')
    plt.close('all')

def plot_comparison(truth, samples, output='./', filename='sample_comparison.png'):
    """
    Plot the samples in the latent space and parameter space
    """
    d = samples.shape[-1]

    samples = samples[np.isfinite(samples).all(axis=1)]

    xs = [truth, samples]
    labels = ['reference', 'flows']

    fig, ax = plt.subplots(d, d, figsize=(12, 12))
    if d > 1:
        for i in range(d):
            for j in range(d):
                for x, l in zip(xs, labels):
                    if j < i:
                        ax[i, j].plot(x[:,j], x[:,i], marker=',', linestyle='')
                    elif j == i:
                        ax[i, j].hist(x[:, j], int(np.sqrt(samples.shape[0])), histtype='step', lw=2.0)
                    else:
                        ax[i, j].axis('off')
    else:
        for x, l in zip(xs, labels):
            ax.hist(x, int(np.sqrt(samples.shape[0])), histtype='step')
    plt.tight_layout()
    fig.savefig(output + filename)
    plt.close('all')


class FlowModel(nn.Module):
    """
    Builds the sequential flow model with the provided inputs

    Based on SingleSpeed in: https://github.com/adammoss/nnest/blob/master/nnest/networks.py
    """

    def __init__(self, n_inputs=None, n_conditional_inputs=None, n_neurons=128, n_layers=2, n_blocks=4, device=None):
        super(FlowModel, self).__init__()

        if device is None:
            raise ValueError("Must provided a device or a string for a device")

        if type(device) == str:
            self.device = torch.device(device)
        else:
            self.device = device

        self.n_inputs = n_inputs
        if n_conditional_inputs is not None:
            self.conditional = True
            self.n_clusters = n_conditional_inputs
        else:
            self.conditional = False
            self.n_clusters = 1

        mask = torch.remainder(torch.arange(0, n_inputs, dtype=torch.float, device=self.device), 2)

        layers = []
        for _ in range(n_blocks):
            layers += [CouplingLayer(n_inputs, n_neurons, mask,
                                     num_cond_inputs=n_conditional_inputs,
                                     num_layers=n_layers),
                       BatchNormFlow(n_inputs)]
            #layers += [MADE(n_inputs, n_neurons)]
            mask = 1 - mask

        self.net = FlowSequential(*layers)
        self.net.to(self.device)

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """
        Forward pass
        """
        return self.net(inputs, cond_inputs=cond_inputs, mode=mode, logdets=logdets)

    def log_probs(self, inputs, cond_inputs=None):
        """
        Log Likelihood
        """
        return self.net.log_probs(inputs, cond_inputs=cond_inputs)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        """
        Produce samples
        """
        return self.net.sample(num_samples=num_samples, noise=noise, cond_inputs=cond_inputs)


class FlowTrainer(Trainer):


    def __init__(self, trainer_dict=None, manager=None, output='./'):
        self.manager=None
        self.setup_model = setup_model
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
        self.logit = False
        # define setup function
        self._setup_from_input_dict(trainer_dict)


    def save_input(self, attr_dict):
        """
        Save the dictionary used as an inputs as a JSON file
        """
        d = attr_dict.copy()
        output_file = self.outdir + "trainer_dict.json"
        for k, v in list(d.items()):
            if type(v) == np.ndarray:
                d[k] = np.array_str(d[k])
        with open(output_file, "w") as f:
            json.dump(d, f, indent=4)

    def _setup_from_input_dict(self, attr_dict):
        """
        Setup the trainer from a dictionary
        """
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
        self.model = self.setup_model(**self.model_dict, device=self.device)
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
        """
        Reset the weights and optimiser
        """
        self.model.apply(weight_reset)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

    def setup_normalisation(self):
        """
        Setup normalisation using the priors
        """
        self._prior_max = np.max(self.priors, axis=0)
        self._prior_min = np.min(self.priors, axis=0)
        self.normalise = True

        if self.logit:
            self.logger.debug('Using logit')
            self.normalise_samples = self.normalise_logit
            self.rescale_samples = self.inverse_normalise_logit

    def normalise_logit(self, x):
        """
        Normalise samples to 0, 1 then apply the logit function
        """
        x_bar = (x - self._prior_min) / (self._prior_max - self._prior_min)
        return logit(x_bar)

    def inverse_normalise_logit(self, x):
        """
        Apply the logistic function (inverse logit) and rescle to original inputs
        """
        x = logistic(x)
        return (self._prior_max - self._prior_min) * x + self._prior_min

    def normalise_samples(self, x):
        """
        Normalise a set of samples
        """
        return 2. * ((x - self._prior_min) / (self._prior_max - self._prior_min)) - 1

    def rescale_samples(self, x):
        """
        Apply the inverse of the normalisation
        """
        return (self._prior_max - self._prior_min) * ((x + 1) / 2.) + self._prior_min

    def train(self, payload):
        """
        Training the flow given a payload of CPnest LivePoints
        """
        iteration = payload[0]
        samples = np.array([p.values for p in payload[1]])
        self.logger.info("Starting training setup")

        D, p_value = self._train_on_data(samples, plot=True, iteration=iteration)

        self.logger.info("Training complete")

        # send weights and check whether to enable the flow
        if self.manager is not None:
            if D >= 0.:      # 0 since flow is always used
                self.manager.enable_flow.value = 1
            self.manager.trained.value = 1
            self.producer_pipe.send(self.weights_file)
            self.logger.info("Weights sent")


    def _prep_data(self, samples, plot=False, output='./'):
        """
        Prep data and return dataloaders for training
        """
        if plot:
            self.logger.debug('Plotting inputs before normalisation')
            plot_inputs(samples, output=output, filename='input_pre_norm.png')

        if self.normalise:
            self.logger.info("Using normalisation")
            samples = self.normalise_samples(samples)
            #samples = self.normalise_logit(samples)

        if plot:
            self.logger.debug("Flow trainer: plotting input")
            plot_inputs(samples, output=output)

        self.logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        x_train, x_val = train_test_split(samples, test_size=self.val_size, shuffle=False)
        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)

        return train_loader, val_loader

    def _train_on_data(self, samples, plot=False, max_epochs=None, patience=None, iteration=None):
        """
        Train the flow on samples
        """
        if not self.intialised:
            self.logger.info("Initialising")
            self.initialise()
            # Option to force first training to be longer
            #max_epochs = 5000
            #patience = 500

        elif self.training_count:
            self.logger.info("Reseting weights")
            self._reset_model()

        block_outdir = "{}block{}/".format(self.outdir, self.training_count)

        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)

        train_loader, val_loader = self._prep_data(samples, plot=True, output=block_outdir)

        # train
        if max_epochs is None:
            max_epochs = self.max_epochs
        if patience is None:
            patience = self.patience
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        self.logger.info("Starting training")
        self.logger.info("Training parameters:")
        self.logger.info(f"Max. epochs: {max_epochs}")
        self.logger.info(f"Patience: {patience}")
        history = dict(loss=[], val_loss=[])

        self.weights_file = block_outdir + 'model.pt'

        if iteration is not None:
            pass
            # TODO: make this general?
            #P_theta = 1. / 400.
            #N_live = 5000
            #prior_kl = (iteration / N_live) + np.log(P_theta)
            #self.logger.debug(f"Prior portion of KL: {prior_kl}")

        for epoch in range(1, max_epochs + 1):

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

            if epoch - best_epoch > patience:
                self.logger.info(f"Epoch {epoch}: Reached patience")
                break

            if self.manager is not None:
                if self.manager.stop_training.value:
                    if epoch >= patience:
                        break

        self.training_count += 1
        self.model.load_state_dict(best_model.state_dict())
        self.weights_file = block_outdir + 'model.pt'
        torch.save(self.model.state_dict(), self.weights_file)

        self.model.eval()
        # sample for plots
        z, output = self.sample(N=10000)
        np.save(block_outdir + 'samples.npy', [z, output])

        if plot:
            self.logger.info("Plotting output")
            plot_loss(epoch, history, output=block_outdir)
            plot_samples(z, output, output=block_outdir)
            if self.normalise:
                output = self.rescale_samples(output)
            #rescaled_output = self.inverse_normalise_logit(output)
            plot_samples(z, output, output=block_outdir, filename='rescaled_output_samples.png')
            # TODO: fix plotting for N dimensions
            #plot_corner_contour([samples, output], labels=["Input data", "Generated data"], filename=block_outdir+"comparison.png")

        # compute mean KS
        # TDOD: remove this?
        D, p_value = self.compute_mean_ks(samples, output)
        self.logger.info(f"Computed KS - D: {D}, p: {p_value}")
        return D, p_value

    def _train(self, loader, noise_scale=0.01):
        """
        Loop over the data and update the weights
        """
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

            if cond_data is not None:
                with torch.no_grad():
                    model(loader.dataset.tensors[0].to(data.device),
                          loader.dataset.tensors[1].to(data.device).float())
            else:
                with torch.no_grad():
                    model(loader.dataset.tensors[0].to(data.device))

            for module in model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 1

        return train_loss / len(loader)

    def _validate(self, loader):
        """
        Loop over the data and get validation loss
        """
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
                val_loss += -model.log_probs(data, cond_data).mean().item()

        return val_loss / len(loader)

    def sample(self, N=1000):
        """
        Produce N samples drawn from the latent space
        """
        z = torch.randn(N, self.n_inputs, device=self.device)
        output, _ = self.model(z, cond_inputs=None, mode='inverse')
        output = output.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        return z, output

    def load_weights(self, weights_file):
        """
        Load weights for the model

        Model is loaded in evaluation mode (model.eval())
        """
        if not self.initialised:
            self.initialise()
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def compute_mean_ks(self, samples, output):
        """
        Compute the KS over each dimension and take the mean
        """
        samples = np.array(samples)
        output = np.array(output)
        n_dims = samples.shape[-1]
        D_values = np.empty(n_dims,)
        p_values = np.empty(n_dims,)
        for i, t, p in zip(range(n_dims), samples.T, output.T):
            D_values[i], p_values[i] = stats.ks_2samp(*[t, p])
        return D_values.mean(), p_values.mean()


class AugmentedFlowTrainer(FlowTrainer):
    """
    A class for training flows with an augemented data space
    """
    def __init__(self, **kwargs):
        """
        Intialise
        """
        super(AugmentedFlowTrainer, self).__init__(**kwargs)
        self.augment_dim = self.model_dict['augment_dim']
        # overwrite default setup function
        self.setup_model = setup_augmented_model

    def _prep_data(self, samples, plot=False, output='./'):
        """
        Prep data and return dataloaders for training
        """
        if plot:
            plot_inputs(samples, output=output, filename='input_pre_norm.png')

        if self.normalise:
            self.logger.info("Using normalisation")
            samples = self.normalise_samples(samples)
            if plot:
                self.logger.debug("Flow trainer: plotting input")
                plot_inputs(samples, output=output)

        self.logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        x_train, x_val = train_test_split(samples, test_size=self.val_size, shuffle=False)

        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        e_train = torch.randn(x_train.shape[0], self.augment_dim)
        train_dataset = torch.utils.data.TensorDataset(train_tensor, e_train)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        e_val = torch.randn(x_val.shape[0], self.augment_dim)
        val_dataset = torch.utils.data.TensorDataset(val_tensor, e_val)
        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=x_val.shape[0], shuffle=False)

        return train_loader, val_loader

    def _train(self, loader, noise_scale=0.):
        """
        Loop over the data and update the weights
        """
        model = self.model
        model.train()
        train_loss = 0

        for idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 2:
                    raise NotImplementedError('Augmented flows are not implemented with conditional inputs')
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                x, e = data[0], data[1]
            x = (x + noise_scale * torch.randn_like(x)).to(self.device)
            e = e.to(self.device)
            self.optimiser.zero_grad()
            loss = -model.log_p_xe(x, e).mean()
            train_loss += loss.item()
            loss.backward()
            self.optimiser.step()

            #TODO: conditional?

        return train_loss / len(loader)

    def _validate(self, loader):
        """
        Loop over the data and get validation loss
        """
        model = self.model
        model.eval()
        val_loss = 0

        for idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 2:
                    raise NotImplementedError('Augmented flows are not implemented with conditional inputs')
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                x, e = data[0], data[1]
            x = x.to(self.device)
            e = e.to(self.device)
            with torch.no_grad():
                val_loss += -model.log_p_xe(x, e).mean().item()

        return val_loss / len(loader)

    def sample(self, N=1000, return_augment=False):
        """
        Produce N samples drawn from the latent space
        """
        # latent for x
        y = torch.randn(N, self.n_inputs, device=self.device)
        # latent for e
        z = torch.randn(N, self.augment_dim, device=self.device)
        # generate samples
        x, e, _ = self.model(y, z, mode='generate')
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if return_augment:
            z = z.detach().cpu().numpy()
            e = e.detach().cpu().numpy()
            return y, z, x, e
        else:
            return y, x


class ClusterFlowTrainer(FlowTrainer):

    clusterer_args = dict(min_cluster_size=100,
                        min_samples=100,
                        cluster_selection_epsilon=0.5)

    def __init__(self, max_clusters=10, clusterer_args=None, **kwargs):
        super(ClusterFlowTrainer, self).__init__(**kwargs)

        self.max_clusters = max_clusters

        if clusterer_args is not None:
            self.update_clusterer_args(clusterer_args)
        # init clusterer
        self.update_clusterer()

    def update_clusterer_args(self, args):
        """
        Update the arguments used for the clusterer
        """
        self.clusterer_args.update(args)

    def update_clusterer(self, args=None):
        """
        Update the clusterer
        """
        if args is not None:
            self.clusterer_update_args(args)
        self.clusterer = hdbscan.HDBSCAN(**self.clusterer_args)

    def plot_clusters(self, samples, labels, unique_labels, output=None, filename=None):
        """
        Plot samples by cluster with labels

        0 indicates noise class
        """
        dims = samples.shape[1]
        N = samples.shape[0]
        fig = plt.figure()
        fig, ax = plt.subplots(dims, dims, figsize=(12, 12))
        for c in range(self.n_clusters):
            idx = labels[:, c].astype(bool)
            d = samples[idx]
            for i in range(dims):
                for j in range(dims):
                    if j < i:
                        ax[i, j].plot(d[:,j], d[:,i], marker=',', linestyle='')
                        ax[i, j].text(d[0, 0], d[0, 1], f'{c}')
                    elif j == i:
                        ax[i, j].hist(d[:, j], int(np.sqrt(N)), histtype='step')
                    else:
                        ax[i, j].set_axis_off()
        plt.tight_layout()

        if output is None:
            output = self.output
        if filename is None:
            filename = 'clusters_corner.png'
        fig.savefig(output + filename)
        plt.close('all')

    def cluster(self, samples, plot=False, output='./'):
        """
        Cluster a set of samples
        """
        self.logger.debug('Clustering samples')
        self.clusterer.fit(samples)
        labels = self.clusterer.labels_
        unique_labels = np.unique(labels)
        # noise is labelled as -1 and is problem when converting to one-hot
        if unique_labels[0] == -1:
            labels += 1
            unique_labels += 1
        self.n_clusters = unique_labels.size
        self.logger.debug(f'Found {self.n_clusters} clusters: {unique_labels}')
        labels = np.eye(self.max_clusters)[labels]

        if plot:
            self.plot_clusters(samples, labels, unique_labels, output=output)

        return samples, labels

    def _prep_data(self, samples, plot=False, output='./'):
        """
        Prep data and return dataloaders for training
        """

        samples, labels = self.cluster(samples, plot=plot, output=output)

        if plot:
            plot_inputs(samples, output=output, filename='input_pre_norm.png')

        if self.normalise:
            self.logger.info("Using normalisation")
            samples = self.normalise_samples(samples)

        if plot:
            self.logger.debug("Flow trainer: plotting input")
            plot_inputs(samples, output=output)

        self.logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        x_train, x_val = train_test_split(samples, test_size=self.val_size, shuffle=False)
        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        x_labels, x_val_labels = train_test_split(labels, test_size=self.val_size, shuffle=False)
        train_labels = torch.from_numpy(x_labels.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        val_labels = torch.from_numpy(x_val_labels.astype(np.float32))
        val_dataset = torch.utils.data.TensorDataset(val_tensor, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)

        return train_loader, val_loader

    def sample(self, N=1000, plot=False, output='./'):
        """
        Produce N samples for each cluster drawn from the latent space
        """
        active_clusters = list(range(self.n_clusters))
        z = torch.randn(N * self.n_clusters, self.n_inputs, device=self.device)
        all_outputs = np.empty([0, self.n_inputs])
        all_labels = np.empty([0, self.max_clusters])

        for c in range(self.max_clusters):
            # skip unused clusters
            if c not in active_clusters:
                continue
            labels = np.zeros([N, self.max_clusters])
            labels[:, c] = 1
            all_labels = np.concatenate([all_labels, labels], axis=0)
            idx = c * N
            labels_tensor = torch.from_numpy(labels.astype(np.float32)).to(self.device)
            outputs, _ = self.model(z[idx:idx+N], cond_inputs=labels_tensor, mode='inverse')
            outputs = outputs.detach().cpu().numpy()
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

        z = z.detach().cpu().numpy()

        if plot:
            self.plot_clusters(all_outputs, all_labels, active_clusters, output=output, filename='cluster_samples.png')

        return z, all_outputs, all_labels
