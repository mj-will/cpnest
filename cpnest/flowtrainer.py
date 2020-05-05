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
from .flows import BatchNormFlow, setup_model
from .plot import plot_corner_contour


def update_trainer_dict(d):
    """
    Update the default dictionary for a trainer
    """
    default_model = dict(n_inputs=None, n_neurons=32, n_blocks=4, n_layers=2,
            augment_dim=None, ftype='RealNVP')

    default = dict(outdir='./',
                   lr=0.0001,                  # learning rate
                   training_frequency=1000,    # training frequency in # nested samples
                   batch_size=100,             # batch size
                   val_size=0.1,               # validation per cent (0.1 = 10%)
                   max_epochs=500,             # maximum number of training epochs
                   patience=20,                # stop after n epochs with no improvement
                   device_tag="cuda",          # device for training
                   proposal_device="cpu",      # device for proposals
                   normalise=False,            # normalise using priors
                   logit=False,                # use logit
                   truncate_proposal=False,    # truncate proposal with logL
                   proposal='gaussian',        # proposal distribution
                   proposal_size=10000,        # number of points to propose
                   fuzz=1.0,                   # fuzz factor for radius of contours
                   memory=False,               # memory in number of epochs
                   plot=True,                  # produce diagonostic plots
                   model_dict=default_model)

    if not isinstance(d, dict):
        raise TypeError('Must pass a dictionary to update the default trainer settings')
    else:
        default.update(d)
    # check arguments
    if not default['normalise'] and default['logit']:
        raise RuntimeError('Must enable normalisation to use logit')
    if default['fuzz'] < 1.0:
        raise ValueError('Fuzz factor must be greater or equal to 1.0')

    return default

def logistic(x):
    """Logistic function"""
    return 1. / (1. + np.exp(-x))

def logit(x):
    """Inverse logistic function"""
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

def plot_samples(z, samples, output='./', filename='output_samples.png', names=None, c=None):
    """
    Plot the samples in the latent space and parameter space
    """
    N = samples.shape[0]
    d = samples.shape[-1]

    if names is None:
        names = list(range(d))
    latent_names =  [f'z_{n}' for n in range(d)]

    if c is None:
        c = 'tab:blue'

    samples = samples[np.isfinite(samples).all(axis=1)]

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d >= 2:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].scatter(samples[:,j], samples[:,i], c=c, s=1.)
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                    ax[i, j].set_xlabel(names[j])
                else:
                    ax[i, j].scatter(z[:,j], z[:,i], c=c, s=1.)
                    ax[i, j].set_xlabel(latent_names[j])
                    ax[i, j].set_ylabel(latent_names[i])
    else:
        ax.hist(samples, int(np.sqrt(N)), histtype='step')

    plt.tight_layout()
    fig.savefig(output + filename)
    plt.close('all')

def plot_inputs(samples, output='./', filename='input_samples.png', names=None):
    """
    Plot n-dimensional input samples
    """
    N = samples.shape[0]
    d = samples.shape[-1]
    if names is None:
        names = list(range(d))

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d > 1:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].plot(samples[:,j], samples[:,i], marker=',', linestyle='')
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                    ax[i, j].set_xlabel(names[j])
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

def generate_contour(r, dims, N=1000):
    """Generate a contour"""
    x = np.array([np.random.randn(N) for _ in range(dims)])
    R = np.sqrt(np.sum(x ** 2., axis=0))
    z = x / R
    return r * z.T

def plot_contours(contours, output='./', filename='contours.png', names=None):
    """Plot contours in the latent space and physical space"""
    d = contours.shape[-1]
    if names is None:
        names = list(range(d))
    latent_names =  [f'z_{n}' for n in range(d)]
    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d >= 2:
        for c in contours:
            for i in range(d):
                for j in range(d):
                    if j < i:
                        ax[i, j].scatter(c[1, :,j], c[1, :,i], s=1.)
                    elif j == i:
                        pass
                    else:
                        ax[i, j].scatter(c[0, :,j], c[0, :,i], s=1.)
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].axis('off')
                else:
                    ax[i, j].set_xlabel(latent_names[j])
                    ax[i, j].set_ylabel(latent_names[i])
        plt.tight_layout()
        fig.savefig(output + filename)
        plt.close('all')
    else:
        pass



class FlowTrainer(Trainer):

    def __init__(self, trainer_dict=None, manager=None, output=None, cpnest_model=None, **kwargs):
        self.manager=None
        super(FlowTrainer, self).__init__(manager=manager, output=output, **kwargs)
        self.outdir = output
        self.parameters = None
        self.re_parameters = None
        self.logger = logging.getLogger("CPNest")
        self.intialised = False
        # define setup function
        trainer_dict = update_trainer_dict(trainer_dict)
        self._setup_from_input_dict(trainer_dict, cpnest_model=cpnest_model)

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

    def _setup_from_input_dict(self, attr_dict, cpnest_model=None):
        """
        Setup the trainer from a dictionary
        """
        if self.outdir is not None:
            attr_dict.pop('outdir')

        if attr_dict['normalise']:
            if 'prior_bounds' not in attr_dict and cpnest_model is None:
                raise RuntimeError('Must provided CPNest model or prior_bounds to use normalisation')
            else:
             if cpnest_model is not None:
                attr_dict["prior_bounds"] = np.array(cpnest_model.bounds)
        else:
            self.prior_bounds = None
        for key, value in six.iteritems(attr_dict):
            setattr(self, key, value)
        self.n_inputs = self.model_dict["n_inputs"]
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.save_input(attr_dict)

        if 'mask' in self.model_dict.keys():
            self.mask = self.model_dict.pop('mask')
        else:
            self.mask = None

    def get_mask(self, mask):
        """
        Get a the mask
        """
        return None

    def initialise(self):
        """
        Intialise the model and optimiser
        """
        if self.prior_bounds is not None and self.normalise:
            self.logger.info("Setting up normalisation")
            self.setup_normalisation()

        self.device = torch.device(self.device_tag)
        self.model_dict['mask'] = self.get_mask(self.mask)
        self.model = setup_model(**self.model_dict, device=self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

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
        self._prior_min = self.prior_bounds[:,0]
        self._prior_max = self.prior_bounds[:,1]

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

        self._train_on_data(samples, plot=self.plot, iteration=iteration)

        self.logger.info("Training complete")

        # send weights and check whether to enable the flow
        if self.manager is not None:
            self.manager.enable_flow.value = 1
            self.manager.trained.value = 1
            self.producer_pipe.send(self.weights_file)
            self.logger.info("Weights sent")


    def _prep_data(self, samples, plot=False, output='./'):
        """
        Prep data and return dataloaders for training
        """
        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]
        if plot:
            self.logger.debug('Plotting inputs before normalisation')
            plot_inputs(samples, output=output, filename='input_pre_norm.png',
                    names=self.parameters)

        if self.normalise:
            self.logger.info("Using normalisation")
            samples = self.normalise_samples(samples)

        if plot:
            self.logger.debug("Flow trainer: plotting input")
            plot_inputs(samples, output=output, names=self.re_parameters)

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

        elif self.training_count:
            self.logger.info("Reseting weights")
            self._reset_model()

        block_outdir = "{}block{}/".format(self.outdir, self.training_count)

        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)

        train_loader, val_loader = self._prep_data(samples, plot=plot, output=block_outdir)

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
        self.logger.debug(f'Training with {samples.shape[0]} samples')
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

        data_latent = np.empty([0, self.n_inputs])
        for idx, data in enumerate(val_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                else:
                    cond_data = None
                data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                val = self.model(data, cond_data)
                val = val[0]
                val = val.detach().cpu().numpy()
                data_latent = np.concatenate([data_latent, val], axis=0)

        if plot:
            plot_inputs(data_latent, output=block_outdir, filename='transformed_input_samples.png')
            self.logger.info("Plotting output")
            plot_loss(epoch, history, output=block_outdir)
            plot_samples(z, output, output=block_outdir, names=self.re_parameters)
            if self.normalise:
                output = self.rescale_samples(output)
            plot_samples(z, output, output=block_outdir,
                    filename='rescaled_output_samples.png', names=self.parameters)
            # plot_contours
            radii = np.linspace(0.1, 2., 4)
            contours = np.empty([radii.shape[0], 2, 1000, self.n_inputs])
            for i, r in enumerate(radii):
                c = generate_contour(r, self.n_inputs, N=1000)
                contours[i, 0] = c
                contour_tensor = torch.from_numpy(c.astype('float32')).to(self.device)
                with torch.no_grad():
                    contour_phys, _ = self.model(contour_tensor)
                contours[i, 1] =contour_phys.detach().cpu().numpy()

            plot_contours(contours, output=block_outdir, filename='contours.png', names=self.re_parameters)


    def _train(self, loader, noise_scale=0.1):
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

    def _prep_data(self, samples, plot=False, output='./'):
        """
        Prep data and return dataloaders for training
        """
        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]
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
