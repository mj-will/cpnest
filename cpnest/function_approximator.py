import os
import shutil
import six
import numpy as np
import deepdish
import time

from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, TensorBoard

from matplotlib import pyplot as plt

from . import fa_utils as utils

class FunctionApproximator(object):

    def __init__(self, input_shape=None, parameter_names=None, json_file=None, attr_dict=None, input_dict=None, verbose=1):

        self.verbose = verbose
        # input independent inits
        self.compiled = False
        self.model = None
        self._count = 0
        self.normalise = False
        self.normalise_output = False

        if json_file is not None and attr_dict is not None:
            raise ValueError("Provided both json file and attribute dict, use one or other")

        elif json_file is not None or input_dict is not None:
            # make sure input_shape is a list
            # also determine whether to split inputs or not
            if input_shape is not None:
                if type(input_shape) == int:
                    self.split = False
                    self.input_shape = [input_shape]
                else:
                    if len(input_shape) > 1:
                        self.split = True
                    else:
                        self.split = False
                    self.input_shape = input_shape
                if parameter_names is None:
                    self.parameter_names = ["parameter_" + str(i) for i in range(self._n_parameters)]
                else:
                    self.parameter_names = parameter_names
                # json file requieres input shape is given
                if json_file is not None:
                    self.setup_from_json(json_file)
            elif input_dict is not None:
                self.setup_from_input_dict(input_dict)
                self.parameter_names = ["parameter_" + str(i) for i in range(self._n_parameters)]
                self.split = False
            else:
                raise ValueError("Could not initialise function approximator, check inputs")
        elif attr_dict is not None:
            self.n_inputs = False
            self.setup_from_attr_dict(attr_dict, verbose=verbose)
            self.normalise_output = False
        else:
            raise ValueError("No json file or saved FA file for setup")

    def __str__(self):
        args = "".join("{}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))
        return "FunctionApproximator instance\n" + "".join("    {}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))

    @property
    def _n_parameters(self):
        """Return the number of input parameters"""
        return sum(self.input_shape)

    def setup_from_attr_dict(self, attr_dict, verbose=0):
        """Set up the approximator from a dictionary of attributes"""
        with open(attr_dict, "rb") as f:
            d = six.moves.cPickle.load(f, encoding='latin1')
        for key, value in six.iteritems(d):
            setattr(self, key, value)
        if self.n_inputs:
            self.input_shape = self.n_inputs
        self.verbose = verbose
        self.model = utils.network(self.input_shape, self.parameters, self.verbose)
        self._compile_network()

    def setup_from_json(self, json_file):
        """Set up the class before training from a json file"""
        if self.model is None:
            print("Setting up function approximator")
            self.parameters = utils.get_parameters_from_json(json_file)
            self.save = self.parameters["save"]
            self.outdir = self.parameters["outdir"]
            self._setup_directories()
            shutil.copy(json_file, self.tmp_outdir)
            # setup network
            self.model = utils.network(self.input_shape, self.parameters, self.verbose)
            self._compile_network()
            self._start_time = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            self.data_all = {}
        else:
            print("Function approximator already setup")

    def setup_from_input_dict(self, input_dict):
        """Set up the class before training from a dict"""
        if self.model is None:
            print("Setting up function approximator")
            for key, value in six.iteritems(input_dict):
                setattr(self, key, value)
            self.input_shape = [self.input_shape]
            self._setup_directories()
            self.model = utils.network(self.input_shape, self.parameters, self.verbose)
            self._compile_network()
            self._start_time = time.strftime('%b-%d-%Y_%H%M', time.localtime())
            self.data_all = {}

    def _setup_directories(self):
        """Setup final output directory and a temporary directory for use during training"""
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        try:
            self.tmp_outdir = self.parameters["tmpdir"]
        except:
            self.tmp_outdir = './current_run/'
        # directory may still exist from previous runs that did not terminate properly
        # delete to prevent results from getting mixed up
        if os.path.exists(self.tmp_outdir):
            shutil.rmtree(self.tmp_outdir)
        os.mkdir(self.tmp_outdir)

    def _compile_network(self):
        """Setup the loss function and compile the model"""
        if self.parameters["loss"] in ("KL","kl"):
            print("KL divergence")
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=utils.KL, metrics=["mse", utils.KL, utils.JSD])
        elif self.parameters["loss"] in ("JSD", "jsd", "JS", "js"):
            print("Using JSD divergence")
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=utils.JSD, metrics=["mse", utils.KL, utils.JSD])
        else:
            print("Using " + self.parameters["loss"])
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=self.parameters["loss"])
            #self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=self.parameters["loss"], metrics=[utils.network.KL, utils.network.JSD])

        self.compiled = True

    def _shuffle_data(self, x, y):
        """Shuffle data"""
        p = np.random.permutation(len(y))
        return x[p], y[p]

    def setup_normalisation(self, priors, normalise_output=False, f=None, inv_f=None, f_kwargs=None):
        """
        Get range of priors for the parameters to be later used for normalisation
        NOTE: expect parameters to be ordered in same order as parameter sets
        """
        self._prior_max = np.max(priors, axis=0)
        self._prior_min = np.min(priors, axis=0)
        self.normalise = True
        if normalise_output:
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
            self.normalise_output = True

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

    def train_on_data(self, x, y, split=0.8, accumulate=False, plot=False, max_training_data=None, enable_tensorboard=False):
        """
        Train on provided data

        Args:
            x : list of array-like samples
            y : list of true values
        """
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training")
        block_outdir = self.tmp_outdir + "block{}/".format(self._count)
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
        idx = np.where(np.abs(y - y.mean()) < 5. * np.std(y))
        print(idx)
        x = x[idx]
        y = y[idx]
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
        print(len(x_split[0]))
        y_m = np.concatenate(y_split, axis=0).mean()
        y_std = np.concatenate(y_split, axis=0).std()
        y_idx = [np.where(np.abs(tmp_y - y_m) < 4. * y_std) for tmp_y in y_split]
        for i, y_tmp in enumerate(y_split):
            y_idx = np.where(np.abs(y_tmp - y_m) < 5. * y_std)
            x_split[i] = x_split[i][y_idx]
            y_split[i] = y_split[i][y_idx]
        for i, x_tmp in enumerate(x_split):
            x_split[i], idx = np.unique(x_tmp, axis=0, return_index=True)
            y_split[i] = y_split[i][idx]

        print(len(x_split[0]))

        # save processed data if accumulating
        if accumulate is not False:
                self._accumulated_data = (x_split, y_split)

        # get train/val and split if inputs are split
        x_train, x_val = x_split
        self.x_train = self._split_data(x_train)
        self.x_val = self._split_data(x_val)
        self.y_train, self.y_val = y_split
        # if maximum number of training samples is set, use a subset
        n_train = self.x_train.shape[0]
        n_val = self.x_val.shape[0]
        if max_training_data is not None and n_train > max_training_data:
            print("Using random subset of data")
            idx = np.random.permutation(range(n_train))[:max_training_data]
            self.x_train = self.x_train[idx]
            self.y_train = self.y_train[idx]
            if n_val > max_training_data:
                idx_val = np.random.permutation(range(n_val))[:max_training_data]
                self.x_val = self.x_val[idx_val]
                self.y_val = self.y_val[idx_val]
        print("Training data shapes: x {}, y: {}".format(self.x_train.shape, self.y_train.shape))
        print("Validation data shapes: x {}, y: {}".format(self.x_val.shape, self.y_val.shape))
        callbacks = []
        if self.parameters["patience"]:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=self.parameters["patience"]))
        # save best model during traing
        self.weights_file = block_outdir + "model_weights.h5"
        checkpoint = ModelCheckpoint(self.weights_file, verbose=0, monitor="val_loss", save_best_only=True, mode="auto")
        if enable_tensorboard:
            tensorboard = TensorBoard(log_dir="./logs/run_start_{}/block{}".format(self._start_time, self._count), batch_size=self.parameters["batch_size"])
            callbacks.append(tensorboard)
        callbacks.append(checkpoint)
        # more callbacks can be added by appending to callbacks
        history = self.model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_val, self.y_val), verbose=self.verbose, callbacks=callbacks, **self._training_parameters)
        y_train_pred = self.model.predict(self.x_train)
        # load weights from best epoch
        self.model.load_weights(self.weights_file)
        # cast to float64 since results are loglikelihoods
        # if logL ~ 100, exp(logL) will return inf
        y_pred = np.float64(self.model.predict(self.x_val).ravel())
        # check difference between predictions and truth
        true_stats = [np.mean(self.y_val), np.std(self.y_val)]
        pred_stats = [np.mean(y_pred), np.std(y_pred)]
        # save the x arrays before they're split parameter sets
        results_dict = {"x_train": self.x_train,
                        "x_val": self.x_val,
                        "y_train": self.y_train,
                        "y_val": self.y_val,
                        "y_train_pred": y_train_pred,
                        "y_pred": y_pred,
                        "parameters": self.parameter_names}
        results_dict.update(history.history)
        if plot:
            # TODO: plotting
            print("Plotting disabled")
            fig = plt.figure()
            plt.plot(self.y_val, y_pred, '.')
            plt.plot([self.y_val.min(), self.y_val.max()],[self.y_val.min(), self.y_val.max()])
            np.save(block_outdir + "data.npy", [self.y_val, y_pred])
            #plt.plot(self.x_train[:,0], ',')
            fig.savefig(block_outdir + "fig.png")
        self.data_all["block{}".format(self._count)] = results_dict
        self._count += 1
        return true_stats, pred_stats

    def load_weights(self, weights_file):
        """Load weights for the model"""
        self.model.load_weights(weights_file)

    def predict(self, x, return_input=False):
        """Get predictions for a given set of points in parameter space that have not been normalised"""
        # normalise if used in training
        x = [x.values]
        if self.normalise:
            input_data = self._normalise_input_data(x)
        # split data (returns input if N/A)
        split_input_data = self._split_data(input_data)
        # predict
        y = np.float64(self.model.predict(split_input_data).ravel())
        # denormalise if output is normalised
        if self.normalise_output:
            output_data = self._denormalise_output_data(y)
        else:
            output_data = y
        # TODO: fix this crap
        # return
        if return_input:
            return input_data, output_data
        else:
            return output_data

    def predict_normed(self, x, return_input=True):
        """Get predictions for data that is already scaled to [0, 1]"""
        split_data = self._split_data(x)
        output_data = self.model.predict(split_data).ravel()
        if return_input:
            return split_data, output_data
        else:
            return output_data

    def _make_run_dir(self, run_path=None):
        """Check run count and make outdir"""
        if run_path is None:
            run = 0
            while os.path.isdir(self.outdir + 'run{}'.format(run)):
                run += 1

            run_path = self.outdir + 'run{}/'.format(run)
            if not os.path.exists(run_path):
                os.mkdir(run_path)
        self._run_path = run_path
        return run_path

    def save_results(self, fname="results.h5", save=True):
        """Save the results from the complete training process and move to final save directory"""
        if self.save or save:
            self.run_outdir = self._make_run_dir()
            deepdish.io.save(self.run_outdir + fname, self.data_all)
            utils.copytree(self.tmp_outdir, self.run_outdir)
            # empty current dir
            for f in os.listdir(self.tmp_outdir):
                file_path = os.path.join(self.tmp_outdir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            print("JSON file parameters specified not to save data, skipping. To force saving enable it in this function (not recommeneded)")

    def save_approximator(self, fname="fa.pkl"):
        """Save the attributes of the function approximator"""
        print("Saving approximator as a dictionary of attributes")
        attr_dict = vars(self)
        attr_dict.pop("model")
        outdir = self._run_path + fname
        with open(outdir, "wb") as f:
            six.moves.cPickle.dump(attr_dict, f, protocol=six.moves.cPickle.HIGHEST_PROTOCOL)
        return outdir
