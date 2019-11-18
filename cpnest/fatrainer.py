from __future__ import division, print_function

import time
import six
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from .trainer import Trainer

import torch
import torch.nn as nn

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
        for n in range(n_layers - 1):
            modules += [nn.Linear(n_neurons[n], n_neurons[n+1]), activation_fn()]
            if dropout is not None:
                modules += [nn.Dropout(p=dropout)]
            if batchnorm:
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
            for j in range(n_layers[i] - 1):
                modules[i] += [nn.Linear(n_neurons[i][j], n_neurons[i][j+1]), activation_fn()]
                if dropout is not None:
                    modules[i] += [nn.Dropout(p=dropout)]
                if batchnorm:
                    modules[i] += [nn.BatchNorm1d(n_neurons[i][j+1])]
        # layers to go after concat
        modules[-1] += [nn.Linear(np.sum(n_neurons[i][-1] for i in range(len(n_inputs))), n_neurons[-1][0]), activation_fn()]
        for j in range(n_layers[-1] - 1):
            modules[-1] += [nn.Linear(n_neurons[-1][j], n_neurons[-1][j+1]), activation_fn()]
            if dropout is not None:
                modules[-1] += [nn.Dropout(p=dropout)]
            if batchnorm:
                modules[-1] += [nn.BatchNorm1d(n_neurons[-1][j+1])]

        modules[-1] += [nn.Linear(n_neurons[-1][-1], n_outputs)]

        #self.modules_list = nn.ModuleList(modules)
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

    def __init__(self, trainer_dict=None, attr_dict=None, verbose=1, trainable=True):

        self.verbose = verbose
        # input independent inits
        self.model = None
        self.optimiser = None
        self.trainable = trainable
        self._count = 0
        self.priors = None
        self.normalise = False
        self.normalise_output = False
        self.split = False
        self.max_epochs = 100
        self.bacth_size = 100
        self.loss = 'MSE'
        self.device_tag = None
        self.device = torch.device('cpu')

        if trainer_dict is not None and attr_dict is not None:
            raise ValueError("Provided both json file and attribute dict, use one or other")

        if trainer_dict is not None:
            self.setup_from_dict(trainer_dict)
            self.parameter_names = ["parameter_" + str(i) for i in range(self._n_parameters)]
        elif attr_dict is not None:
            self.n_inputs = False
            self.setup_from_attr_dict(attr_dict, verbose=verbose)
            self.normalise_output = False
        else:
            raise ValueError("No json file or saved FA file for setup")

        if self.device_tag is not None:
            self.deice = torch.device('cpu')
        if not trainable:
            self.model.eval()

    def __str__(self):
        args = "".join("{}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))
        return "FunctionApproximator instance\n" + "".join("    {}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))

    @property
    def _n_parameters(self):
        """Return the number of input parameters"""
        return np.sum(self.input_shape)

    def setup_from_dict(self, trainer_dict):
        """Set up the class before training from a dict"""
        if self.model is None:
            print("Setting up function approximator")
            for key, value in six.iteritems(trainer_dict):
                setattr(self, key, value)
            if self.trainable:
                self._setup_directories()
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

    def _setup_optimiser(self):
        """Setup the optimiser the model"""
        loss_fns = {'MSE': nn.MSELoss(reduction='sum'),
                    'MAE': nn.L1Loss(reduction='sum')
                }
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn  = loss_fns[self.loss]

    def _shuffle_data(self, x, y):
        """Shuffle data"""
        p = np.random.permutation(len(y))
        return x[p], y[p]

    def _setup_input_normalisatiom(self):
        """Set up input normalisation from priors"""
        self._prior_max = np.max(priors, axis=0)
        self._prior_min = np.min(priors, axis=0)
        self.normalise = True

    def _setup_outpt_normalisation(self, f=None, inv_f=None, f_kwargs=None):
        """
        Get range of priors for the parameters to be later used for normalisation
        NOTE: expect parameters to be ordered in same order as parameter sets
        """
        if f is None and inv_f is None:
            self._output_norm_f = utils.ELU
            self._output_norm_inv_f = utils.IELU
            if not f_kwargs is None:
                print("Setting up default output normalisation with custom values")
                self._output_norm_kwargs = f_kwargs
            else:
                print("Setting up defautl output normalisation with default values")
                self._output_norm_kwargs = dict(alpha=0.01)
        elif f is None or inv_f is None:
            raise RuntimeError("Must provide both normalisation function and its inverse.")
        else:
            print("Setting up output normalisation with custom function")
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

    @property
    def _training_parameters(self):
        """Return a dictionary of the parameters to be passed to model.fit"""
        return {"epochs": self.parameters["epochs"], "batch_size": self.parameters["batch_size"]}

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

    def train(self, x, y, split=0.8, accumulate=False, plot=False, max_training_data=None):
        """
        Train on provided data

        Args:
            x : list of array-like samples
            y : list of true values
        """
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

        # remove outliers
        x = np.unique(x, axis=0)
        y = np.unique(y)
        #idx = np.where(np.abs(y - y.mean()) < 5. * np.std(y))
        #x, y = x[idx], y[idx]
        # shuffle
        x, y = self._shuffle_data(x, y)
        # split into train/val
        n = len(y)
        x_split = np.array_split(x, [int(split * n)], axis=0)
        y_split = np.array_split(y, [int(split * n)], axis=0)
        # accumlate data if flag true and not the first instance of training
        if accumulate and self._count:
            x_split = [np.concatenate([acc_x, x_tmp], axis=0) for acc_x, x_tmp in zip(self._accumulated_data[0], x_split)]
            y_split = [np.concatenate([acc_y, y_tmp], axis=0) for acc_y, y_tmp in zip(self._accumulated_data[1], y_split)]
        # remove any duplicate points from training and validation sets
        y_m = np.concatenate(y_split, axis=0).mean()
        y_std = np.concatenate(y_split, axis=0).std()
        print(y_m, y_std)
        #y_idx = [np.where(np.abs(tmp_y - y_m) < 5. * y_std) for tmp_y in y_split]
        for i, y_tmp in enumerate(y_split):
            y_idx = np.where(np.abs(y_tmp - y_m) < 5. * y_std)
            x_split[i] = x_split[i][y_idx]
            y_split[i] = y_split[i][y_idx]
        for i, x_tmp in enumerate(x_split):
            x_split[i], idx = np.unique(x_tmp, axis=0, return_index=True)
            y_split[i] = y_split[i][idx]

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
            print("Using random subset of data")
            idx = np.random.permutation(range(n_train))[:max_training_data]
            x_train = x_train[idx]
            y_train = y_train[idx]
            if n_val > max_training_data:
                idx_val = np.random.permutation(range(n_val))[:max_training_data]
                x_val = x_val[idx_val]
                y_val = y_val[idx_val]
        print("Training data shapes: x {}, y: {}".format(x_train.shape, y_train.shape))
        print("Validation data shapes: x {}, y: {}".format(x_val.shape, y_val.shape))

        train_tensor = [torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))]
        train_dataset = torch.utils.data.TensorDataset(*train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_tensor = [torch.from_numpy(x_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))]
        val_dataset = torch.utils.data.TensorDataset(*val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)

        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)

        history = {"loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):

            loss = self._train(train_loader)
            val_loss = self._validate(val_loader)

            history["loss"].append(loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 50:
                print(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > self.patience:
                print(f"Epoch {epoch}: Reached patience")
                break

        self.model.load_state_dict(best_model.state_dict())
        self.weights_file = block_outdir + "model.pt"
        torch.save(self.model.state_dict(), self.weights_file)

        # predict
        self.model.eval()
        if self.split:
            x_train_test = self._split_data(train_tensor[0])
            x_test = self._split_data(val_tensor[0])
        else:
            x_train_test = train_tensor[0]
            x_test = val_tensor[0]
        y_pred = self.model(x_test).detach().cpu().numpy().flatten()
        y_train_pred = self.model(x_train_test).detach().cpu().numpy()
        # load weights from best epoch
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
            #data = data.to(self.device)
            x, y = data
            if self.split:
                x = self._split_data(x)
            self.optimiser.zero_grad()
            y_pred = self.model(x)
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
            if self.split:
                x = self._split_data(x)
            y_pred = self.model(x)
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
        if self.normalise:
            x = self._normalise_input_data(x)
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

    def save_approximator(self, fname="fa.pkl"):
        """Save the attributes of the function approximator"""
        print("Saving approximator as a dictionary of attributes")
        attr_dict = vars(self)
        attr_dict.pop("model")
        output_file = self.outdir + fname
        with open(output_file, "wb") as f:
            six.moves.cPickle.dump(attr_dict, f, protocol=six.moves.cPickle.HIGHEST_PROTOCOL)
        return output_file


class FAPlots(object):

    def __init__(self, outdir='./', x_train=None, y_train=None, x_val=None, y_val=None, y_train_pred=None, y_pred=None, history=None, parameters=None):

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
        plt.plot(self.y_val, self.y_pred, '.')
        plt.plot([self.y_val.min(), self.y_val.max()],[self.y_val.min(), self.y_val.max()])
        plt.xlabel("True")
        plt.ylabel("predicted")
        fig.savefig(self.outdir + "predictions.png")

    def plot_history(self):
        """Plot losses"""
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = np.arange(1, len(loss), 1)
        fig = plt.figure()
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val. loss')
        plt.legend()
        fig.savefig(self.outdir + "history.png")

class FATrainer(Trainer):

    def __init__(self, trainer_dict, manager=None, output='./'):
        self.trainer_dict = trainer_dict
        super(FATrainer, self).__init__(manager=manager, output=output)

    def initialise(self):
        """
        Import functions that use tensorflow and initialise the function approximator
        """
        self.trainer_dict["outdir"] = self.output + "fa/"
        self.fa = FunctionApproximator(trainer_dict=self.trainer_dict, verbose=1)
        # save so it's easier to init other fa
        self.initialised = True

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
        print("Function approximator: Training started at: {}".format(time.asctime()))
        true_stats, pred_stats = self.fa.train(x, y, accumulate=True, plot=True)
        print("Function approximator: Training ended at: {}".format(time.asctime()))

        mean_ratio = true_stats[0] / pred_stats[0]
        std_ratio = true_stats[1] / pred_stats[1]

        if (np.abs(1. - mean_ratio) < 0.05) and (np.abs(1. - std_ratio) < 0.05):
            self.manager.use_fa.value = 1

        self.manager.trained.value = 1
        self.producer_pipe.send(self.fa.weights_file)
