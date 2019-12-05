from __future__ import division, print_function

import time
import six
import os
import json
import copy
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .trainer import Trainer

import torch
import torch.nn as nn

def ELU(x, alpha=0.01):
    """Exponetial Linear Unit"""
    y = x.copy()
    neg_indices = np.where(x <= 0.)
    y[neg_indices] = alpha * (np.exp(y[neg_indices]) - 1.)
    return y

def IELU(x, alpha=0.01):
    """Inverse of the Exponential Linear Unit"""
    y = x.copy()
    neg_indices = np.where(x <= 0.)
    y[neg_indices] = np.log(y[neg_indices] / alpha + 1.)
    return y

def weight_reset(m):
    """Reset parameters of a given model"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


class SplitNetwork(nn.Module):

    def __init__(self, n_inputs, n_outputs=1, n_layers=4, n_neurons=128, activation='relu', batchnorm=False, dropout=None):
        super(SplitNetwork, self).__init__()

        if isinstance(n_inputs, list):
            split = True
            if not isinstance(n_layers, list):
                n_layers = [n_layers for _ in range(len(n_inputs))]
            elif len(n_layers) == 2:
                n_layers = [n_layers[0] for _ in range(len(n_inputs))] + [n_layers[1]]
            print("Number of layers:", n_layers)
            if isinstance(n_neurons, int):
                n_neurons = [[n_neurons] * n  for n in n_layers]
        else:
            split = False
            if isinstance(n_neurons, int):
                n_neurons = [n_neurons] * n_layers
        print("Neurons per layer: ", n_neurons)

        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'elu': nn.ELU}

        if isinstance(activation, dict):
            self.add_activation(activation)
            activation = list(activation.keys())[0]
        activation_fn = activations[activation]

        if split:
            self._build_split_model(n_inputs, n_outputs, n_layers, n_neurons, activation_fn, batchnorm, dropout)
        else:
            self._build_simple_model(n_inputs, n_outputs, n_layers, n_neurons, activation_fn, batchnorm, dropout)

    def forward(self, x):
        """Placeholder forward pass"""
        return x

    def _build_simple_model(self, n_inputs, n_outputs, n_layers, n_neurons, activation_fn, batchnorm, dropout):
        modules = list()

        modules += [nn.Linear(n_inputs, n_neurons[0]), activation_fn()]
        if dropout is not None:
            modules += [nn.Dropout(p=dropout)]
        if batchnorm:
            modules += [nn.BatchNorm1d(n_neurons[0])]
        for n in range(n_layers - 1):
            modules += [nn.Linear(n_neurons[n], n_neurons[n+1]), activation_fn()]
            if dropout is not None:
                modules += [nn.Dropout(p=dropout)]
            if batchnorm:
                if n + 1 < n_layers -1:
                    modules += [nn.BatchNorm1d(n_neurons[n+1])]
        modules += [nn.Linear(n_neurons[-1], n_outputs)]

        self.modules_list = nn.ModuleList(modules)
        self.forward = self._simple_forward

    def _simple_forward(self, x):
        """Forward pass for a network with no split"""
        for m in self.modules_list:
            x = m(x)
        return x

    def _build_split_model(self, n_inputs, n_outputs, n_layers, n_neurons, activation_fn, batchnorm, dropout):

        modules = [[] for _ in range(len(n_inputs) + 1)]
        count = 0
        # list of modules in the split inputs
        for i in range(len(n_inputs)):
            modules[i] += [nn.Linear(n_inputs[i], n_neurons[i][0]), activation_fn()]
            if dropout is not None:
                modules[i] += [nn.Dropout(p=dropout)]
            if batchnorm:
                modules[i] += [nn.BatchNorm1d(n_neurons[i][0])]
            for j in range(n_layers[i] - 1):
                modules[i] += [nn.Linear(n_neurons[i][j], n_neurons[i][j+1]), activation_fn()]
                if dropout is not None:
                    modules[i] += [nn.Dropout(p=dropout)]
                if batchnorm:
                    modules[i] += [nn.BatchNorm1d(n_neurons[i][j+1])]
        # layers to go after concat
        modules[-1] += [nn.Linear(np.sum(np.fromiter((n_neurons[i][-1] for i in range(len(n_inputs))), int)), n_neurons[-1][0]), activation_fn()]
        for j in range(n_layers[-1] - 1):
            modules[-1] += [nn.Linear(n_neurons[-1][j], n_neurons[-1][j+1]), activation_fn()]
            if dropout is not None:
                modules[-1] += [nn.Dropout(p=dropout)]
            if batchnorm:
                if j + 1 < len(n_layers[-1]) - 1:
                    modules[-1] += [nn.BatchNorm1d(n_neurons[-1][j+1])]

        modules[-1] += [nn.Linear(n_neurons[-1][-1], n_outputs)]

        self.modules_list = [nn.ModuleList(m) for m in modules]
        self.all_models = nn.ModuleList([y for x in self.modules_list for y in x])
        self.forward = self._split_forward


    def _split_forward(self, x):
        """Forward pass for a network with split inputs"""
        # pass through two splits
        for i, l in enumerate(self.modules_list[:-1]):
            for m in l:
                x[i] = m(x[i])
        # joint outputs of splits
        x = torch.cat(x, axis=1)
        # pass through final layers
        for m in self.modules_list[-1]:
            x = m(x)
        return x

    def add_activation(self, new_activation):
        """Add an activation to the dictionary of possible activation functions"""
        self.activations = dict(self.activations, **new_activation)


class FunctionApproximator(object):

    def __init__(self, trainer_dict=None, attr_dict=None, trainable=True, verbose=1, dev=False, device=None):

        self.logger = logging.getLogger(__name__ + ".FunctionApproximator")

        self.verbose = verbose
        # input independent inits
        self.model = None
        self.priors = None
        self.normalise = False
        self.normalise_output = False
        self.split = False
        self.max_epochs = 100
        self.bacth_size = 100
        # optimiser
        self.optimiser = None
        self.loss = 'MSE'            # loss function
        self.weight_decay = None     # weight decay for optimiser
        # learning rate parameters
        self.lr = 0.001
        self.lr_patience = None      # patience for decreasing learning rate
        self.factor = 0.5            # factor by whic to reduce lr
        self.cooldown = 0            # minimum period between reductions in lr

        self.trainable = trainable
        self.dev = dev

        self.scheduler = None
        self._count = 0

        # device will override dictionary
        if device is not None:
            trainer_dict["device_tag"] = device


        if trainer_dict is not None and attr_dict is not None:
            raise ValueError("Provided both json file and attribute dict, use one or other")

        if trainer_dict is not None:
            self.setup_from_dict(trainer_dict)
            self.parameter_names = ["parameter_" + str(i) for i in range(self._n_parameters)]
            if trainable:
                self.save_input(trainer_dict)
        elif attr_dict is not None:
            self.n_inputs = False
            self.setup_from_attr_dict(attr_dict, verbose=verbose)
            self.normalise_output = False
        else:
            raise ValueError("No json file or saved FA file for setup")

        if not trainable:
            self.model.eval()

    def __str__(self):
        args = "".join("{}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))
        return "FunctionApproximator instance\n" + "".join("    {}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))

    @property
    def _n_parameters(self):
        """Return the number of input parameters"""
        return np.sum(self.input_shape)

    def set_device(self, device_tag):
        """Set device from a string"""
        if device_tag is None:
            device_tag = 'cpu'
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_tag)
        self.logger.info("Running on " + device_tag)

    def setup_from_dict(self, trainer_dict):
        """Set up the class before training from a dict"""
        if self.model is None:
            self.logger.info("Setting up function approximator")
            for key, value in six.iteritems(trainer_dict):
                setattr(self, key, value)
            if self.trainable:
                self._setup_directories()
            self.set_device(self.device_tag)
            self._setup_model(self.model_dict)
            self._setup_optimiser()
            self.input_shape = self.model_dict["n_inputs"]
            self._setup_split()
            if self.priors is not None:
                self._setup_input_normalisation()
            if self.normalise_output:
                self._setup_output_normalisation()
            self._start_time = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            self.data_all = {}

    def setup_from_attr_dict(self, attr_dict, verbose=0):
        """Set up the approximator from a dictionary of attributes"""
        for key, value in six.iteritems(attr_dict):
            setattr(self, key, value)
        self.verbose = verbose
        self._setup_model(self.model_dict)

    def _setup_directories(self):
        """Setup final output directory and a temporary directory for use during training"""
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

    def _setup_model(self, trainer_dict):
        """Setup up the model"""
        self.model = SplitNetwork(**trainer_dict)
        self.logger.info("Model:", self.model)
        self.model.to(self.device)

    def _setup_optimiser(self):
        """Setup the optimiser the model"""
        loss_fns = {'MSE': nn.MSELoss(reduction='mean'),
                    'MAE': nn.L1Loss(reduction='mean')}
        if self.weight_decay is None:
            self.weight_decay = 0.
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_patience is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimiser, patience=self.lr_patience, factor=self.factor, cooldown=self.cooldown, verbose=True)
        self.loss_fn  = loss_fns[self.loss]

    def _reset_model(self):
        """Reset the weights and optimiser"""
        self.model.apply(weight_reset)
        self._setup_optimiser()

    def _shuffle_data(self, x, y):
        """Shuffle data"""
        p = np.random.permutation(len(y))
        return x[p], y[p]

    def _setup_input_normalisation(self):
        """Set up input normalisation from priors"""
        self._prior_max = np.max(self.priors, axis=0)
        self._prior_min = np.min(self.priors, axis=0)
        self.normalise = True

    def _setup_output_normalisation(self, f=None, inv_f=None, f_kwargs=None):
        """
        Get range of priors for the parameters to be later used for normalisation
        NOTE: expect parameters to be ordered in same order as parameter sets
        """
        if f is None and inv_f is None:
            self._output_norm_f = ELU
            self._output_norm_inv_f = IELU
            if not f_kwargs is None:
                self.logger.info("Setting up default output normalisation with custom values")
                self._output_norm_kwargs = f_kwargs
            else:
                self.logger.info("Setting up defautl output normalisation with default values")
                self._output_norm_kwargs = dict(alpha=0.01)
        elif f is None or inv_f is None:
            raise RuntimeError("Must provide both normalisation function and its inverse.")
        else:
            self.logger.info("Setting up output normalisation with custom function")
            self._output_norm_f = f
            self._output_norm_inv_f = inv_f
            if not f_kwargs is None:
                self._output_norm_kwargs = f_kwargs
            else:
                self._output_norm_kwargs = dict()

    @property
    def _priors(self):
        """Return the min and max of the priors used to normalise values"""
        return self._prior_min, self._prior_max

    def _normalise_input_data(self, x):
        """Normalise the input data given the prior values provided at setup"""
        return (x - self._prior_min) / (self._prior_max - self._prior_min)

    def _normalise_output_data(self, x):
        """Normalise the output (normally loglikelihood) using a function"""
        return self._output_norm_f(x, **self._output_norm_kwargs)

    def _denormalise_output_data(self, x):
        """Denormalise the output (normally loglikelihood) using the inverse of the function"""
        return self._output_norm_inv_f(x, **self._output_norm_kwargs)

    def _setup_split(self):
        """Enable or disable split"""
        if isinstance(self.input_shape, list):
            self.split = True

    def _split_data(self, x):
        """Split data according to number of input parameter sets parameters"""
        if self.split:
            x_split = []
            m = 0
            for n in self.input_shape:
                x_split.append(x[:, m:m + n])
                m = n
        else:
            x_split = x
        return x_split

    def _plot_input_data(self, Y, block_outdir):
        """Plot the input data and possible truncations"""
        y_mean = np.concatenate(Y, axis=0).mean()
        y_std = np.concatenate(Y, axis=0).std()
        sigma = [10, 5, 3]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex='col')
        axes[0,0].hist(Y[0], bins=20, density=True, alpha=0.7)
        axes[1,0].hist(Y[1], bins=20, density=True, alpha=0.7)
        axes[0,0].set_title('Input data', fontsize=14)

        for i, y in enumerate(Y):
            for j, s in enumerate(sigma):
                j += 1
                if s is not None:
                    y_trunc = y[np.where(np.abs(y - y_mean) < s * y_std)]
                else: y_trunc = y
                axes[i, j].hist(y_trunc, bins=20, density=True, alpha=0.7)
                if not i:
                    axes[i,j].set_title(f'Cut at {s}$ sigma$', fontsize=14)

        colours = plt.cm.Dark2(np.linspace(0,1,9))[1:]
        for ax in axes.ravel():
            for i, s in enumerate([1, 3, 5, 10]):
                y_lower = y_mean - s * y_std
                y_upper = y_mean + s * y_std
                xlims = ax.get_xlim()
                if y_lower > xlims[0]:
                    ax.axvline(x=y_lower, linestyle='--', c=colours[i])
                if y_upper < xlims[1]:
                    ax.axvline(x=y_upper, linestyle='--', c=colours[i])

        axes[0,0].set_ylabel('Training data', fontsize=14)
        axes[1,0].set_ylabel('Validation data', fontsize=14)
        plt.figtext(0.5, -0.02, 'Log-likelihood', ha='center', fontsize=16)
        plt.tight_layout()
        labels = ['1', '3', '5', '10']
        handles = [mpl.lines.Line2D([0], [0], color=c, lw=2, ls='--') for c in colours[:len(labels)]]
        leg = plt.legend(handles, labels, loc=(-0.5, -0.35), title='Sigma:', ncol=len(labels), fontsize=12, title_fontsize=12)
        leg._legend_box.align = "left"
        fig.savefig(block_outdir + 'input_data.png', bbox_inches='tight')

    def train(self, x, y, split=0.8, accumulate=False, plot=False, max_training_data=None, truncate=True, reset=False):
        """
        Train on provided data

        Args:
            x : list of array-like samples
            y : list of true values
        """
        if reset:
            self._reset_model()

        if self.optimiser is None:
            raise RuntimeError("Optimiser not defined")

        block_outdir = self.outdir + "block{}/".format(self._count)
        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)

        if self.normalise:
            x = self._normalise_input_data(x)
        # if normalising output all ouput data will be saved normalised
        if self.normalise_output:
            y = self._normalise_output_data(y)

        # shuffle
        x, y = self._shuffle_data(x, y)
        # split into train/val
        n = len(y)
        x_split = np.array_split(x, [int(split * n)], axis=0)
        y_split = np.array_split(y, [int(split * n)], axis=0)
        # accumlate data if flag true and not the first instance of training
        if accumulate and self._count:
            self.logger.info("Using accumulated data")
            x_split = [np.concatenate([acc_x, x_tmp], axis=0) for acc_x, x_tmp in zip(self._accumulated_data[0], x_split)]
            y_split = [np.concatenate([acc_y, y_tmp], axis=0) for acc_y, y_tmp in zip(self._accumulated_data[1], y_split)]
            # remove any duplicates
            for i, x_tmp in enumerate(x_split):
                _, idx = np.unique(x_tmp, axis=0, return_index=True)
                idx = np.argsort(idx)
                x_split[i] = x_split[i][idx]
                y_split[i] = y_split[i][idx]

        if self.dev:
            self._plot_input_data(y_split, block_outdir)

        # remove outliers
        if not self.normalise_output:
            if truncate:
                y_m = np.concatenate(y_split, axis=0).mean()
                y_std = np.concatenate(y_split, axis=0).std()
                for i, y_tmp in enumerate(y_split):
                    y_idx = np.where(np.abs(y_tmp - y_m) < 5. * y_std)
                    x_split[i] = x_split[i][y_idx]
                    y_split[i] = y_split[i][y_idx]

        # save processed data if accumulating
        if accumulate is not False:
            self._accumulated_data = (x_split, y_split)

        # get train/val and split if inputs are split
        x_train, x_val = x_split
        y_train, y_val = y_split
        n_train = x_train.shape[0]
        n_val = x_val.shape[0]
        # if maximum number of training samples is set, use a subset
        if max_training_data is not None and n_train > max_training_data:
            self.logger.info("Using random subset of data")
            idx = np.random.permutation(range(n_train))[:max_training_data]
            x_train = x_train[idx]
            y_train = y_train[idx]
            if n_val > max_training_data:
                idx_val = np.random.permutation(range(n_val))[:max_training_data]
                x_val = x_val[idx_val]
                y_val = y_val[idx_val]
        self.logger.info("Training data shapes: x {}, y: {}".format(x_train.shape, y_train.shape))
        self.logger.info("Validation data shapes: x {}, y: {}".format(x_val.shape, y_val.shape))

        train_tensor = [torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))]
        train_dataset = torch.utils.data.TensorDataset(*train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_tensor = [torch.from_numpy(x_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))]
        val_dataset = torch.utils.data.TensorDataset(*val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)

        history = {"loss": [], "val_loss": [], "lr": []}

        for epoch in range(1, self.max_epochs + 1):

            loss = self._train(train_loader)
            val_loss = self._validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                if self.dev:
                    for pg in self.optimiser.param_groups:
                        history["lr"].append(pg["lr"])

            history["loss"].append(loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 50:
                self.logger.info(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > self.patience:
                self.logger.info(f"Epoch {epoch}: Reached patience")
                break

        # load best model
        self.model.load_state_dict(best_model.state_dict())
        self.weights_file = block_outdir + "model.pt"
        torch.save(self.model.state_dict(), self.weights_file)

        # predict
        self.model.eval()
        if self.split:
            x_train_test = self._split_data(train_tensor[0])
            x_test = self._split_data(val_tensor[0])
            x_train = [x.to(self.device) for x in x_train_test]
            x_test = [x.to(self.device) for x in x_test]
        else:
            x_train_test = train_tensor[0].to(self.device)
            x_test = val_tensor[0].to(self.device)

        y_pred = self.model(x_test).detach().cpu().numpy().flatten()
        y_train_pred = self.model(x_train_test).detach().cpu().numpy()

        true_stats = [np.mean(y_val), np.std(y_val)]
        pred_stats = [np.mean(y_pred), np.std(y_pred)]
        # save the x arrays before they're split parameter sets
        results_dict = {"x_train": x_train,
                        "x_val": x_val,
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_train_pred": y_train_pred,
                        "y_pred": y_pred,
                        "parameters": self.parameter_names,
                        "history": history}
        if plot:
            plots = FAPlots(outdir=block_outdir, **results_dict)
            plots.plot_comparison()
            plots.plot_history()
        # add data to data dictionary
        self.data_all["block{}".format(self._count)] = results_dict
        self._count += 1
        return true_stats, pred_stats

    def _train(self, loader):
        """One training step"""
        self.model.train()
        train_loss = 0

        for idx, data in enumerate(loader):
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            if self.split:
                x = self._split_data(x)
            self.optimiser.zero_grad()
            y_pred = self.model(x).flatten()
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            self.optimiser.step()

            return train_loss / len(loader.dataset)

    def _validate(self, loader):
        """ Validate on data"""
        model = self.model.eval()
        val_loss = 0

        for idx, data in enumerate(loader):
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            if self.split:
                x = self._split_data(x)
            y_pred = self.model(x).flatten()
            with torch.no_grad():
                val_loss += self.loss_fn(y_pred, y).item()

            return val_loss / len(loader.dataset)

    def load_weights(self, weights_file):
        """Load weights for the model"""
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def predict(self, x, return_input=False):
        """Get predictions for a given set of points in parameter space that have not been normalised"""
        # normalise if used in training
        x = x.values
        if self.normalise:
            x = self._normalise_input_data(x)
        if self.split:
            x = self._split_data(x)
            x_tensor = [torch.Tensor(a).to(self.device) for a in x]
        else:
            x_tensor = torch.Tensor(x).to(self.device)
        y = np.float64(self.model(x_tensor).detach().cpu().numpy())
        # denormalise if output is normalised
        if self.normalise_output:
            y = self._denormalise_output_data(y)
        # return
        if return_input:
            return x, y
        else:
            return y

    def predict_normed(self, x, return_input=True):
        """Get predictions for data that is already scaled to [0, 1]"""
        if self.split:
            x = self._split_data(x)
            x_tensor = [torch.Tensor(a.astype(np.float32)) for a in x]
        else:
            x_tensor = torch.Tensor(x.astype(np.float32))
        y = np.float64(self.model(x_tensor).detach().cpu().numpy())
        # denormalise if output is normalised
        if self.normalise_output:
            y = self._denormalise_output_data(y)
        # return
        if return_input:
            return x, y
        else:
            return y

    def save_results(self, fname="results.h5"):
        """Save the results from the complete training process and move to final save directory"""
        deepdish.io.save(self.outdir + fname, self.data_all)

    def save_input(self, d):
        """Save the dictionary used as an inputs as a JSON file"""
        output_file = self.outdir + "trainer_dict.json"
        for k, v in list(d.items()):
            if type(v) == np.ndarray:
                d[k] = np.array_str(d[k])
        with open(output_file, "w") as f:
            json.dump(d, f, indent=4)

    def save_approximator(self, fname="fa.pkl"):
        """Save the attributes of the function approximator"""
        self.logger.info("Saving approximator as a dictionary of attributes")
        attr_dict = vars(self)
        attr_dict.pop("model")
        attr_dict.pop("logger")
        output_file = self.outdir + fname
        with open(output_file, "wb") as f:
            six.moves.cPickle.dump(attr_dict, f, protocol=six.moves.cPickle.HIGHEST_PROTOCOL)
        return output_file


class FAPlots(object):

    def __init__(self, outdir='./', x_train=None, y_train=None, x_val=None, y_val=None,
                 y_train_pred=None, y_pred=None, history=None, parameters=None):

        self.outdir = outdir
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.y_pred = y_pred
        self.y_train_pred = y_train_pred
        self.history = history
        self.parameters = parameters

    def plot_comparison(self):
        """Plot comparison between predictions and input data"""
        fig = plt.figure()
        plt.plot(self.y_train, self.y_train_pred, '.')
        plt.plot([self.y_train.min(), self.y_train.max()],[self.y_train.min(), self.y_train.max()])
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Training data")
        fig.savefig(self.outdir + "training_predictions.png")
        fig = plt.figure()
        plt.plot(self.y_val, self.y_pred, '.')
        plt.plot([self.y_val.min(), self.y_val.max()],[self.y_val.min(), self.y_val.max()])
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Test data")
        fig.savefig(self.outdir + "predictions.png")


    def plot_history(self):
        """Plot losses"""
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = np.arange(1, len(loss) + 1, 1)
        if len(self.history['lr']):
            lr = self.history['lr']
        else:
            lr = False
        if lr:
            fig, axes = plt.subplots(2, 1, sharex=True)
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(1, 1, sharex=True)
            axes = [axes]
        axes[0].plot(epochs, loss, label='loss')
        axes[0].plot(epochs, val_loss, label='val. loss')
        axes[0].set_yscale("log")
        if not lr:
            axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        if lr:
            axes[1].plot(epochs, lr)
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Learning rate')
        axes[0].legend()
        fig.savefig(self.outdir + "history.png", bbox_inches='tight')

class FATrainer(Trainer):

    def __init__(self, trainer_dict, manager=None, output='./'):
        self.trainer_dict = trainer_dict
        self.logger = None
        super(FATrainer, self).__init__(manager=manager, output=output)

    def create_logger(self, path="./"):
        """Create logger"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(path + "fa.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)-8s: %(message)s',
            datefmt='%H:%M'))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)-8s: %(message)s',
                datefmt='%H:%M')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _setup_directories(self, outdir):
        """Setup final output directory and a temporary directory for use during training"""
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    def initialise(self):
        """
        Import functions that use tensorflow and initialise the function approximator
        """
        self.trainer_dict["outdir"] = self.output + "fa/"
        self._setup_directories(self.trainer_dict["outdir"])
        self.create_logger(self.trainer_dict["outdir"])
        self.logger.info('Intialising')
        self.fa = FunctionApproximator(trainer_dict=self.trainer_dict,
                                       verbose=1)
        # save so it's easier to init other fa
        self.initialised = True
        if self.manager is not None:
            self.producer_pipe.send(1)

    def train(self, payload):
        """
        Train the function approximator given some input data
        """
        if not self.initialised:
            self.initialise()
        points = []
        logL = []
        for p in payload:
            points.append(p.values)
            logL.append(p.logL)
        x, y = np.array(points), np.array(logL)
        self.logger.info("Training started")
        true_stats, pred_stats = self.fa.train(x, y, accumulate=False, plot=True)
        self.logger.info("Training ended")

        mean_ratio = true_stats[0] / pred_stats[0]
        std_ratio = true_stats[1] / pred_stats[1]

        self.logger.info(f"Mean ratio: {mean_ratio}")
        self.logger.info(f"STD ratio: {std_ratio}")

        if (np.abs(1. - mean_ratio) < 0.05) and (np.abs(1. - std_ratio) < 0.05):
            self.logger.info("Enabling FA")
            self.manager.use_fa.value = 1

        self.manager.trained.value = 1
        self.producer_pipe.send(self.fa.weights_file)
