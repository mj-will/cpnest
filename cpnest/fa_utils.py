
import os
import shutil
import json
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, concatenate, Input, BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers

def set_keras_device(device="GPU0"):
    # get type of device: CPU or GPU
    device_type = device.rstrip("0123456789")
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow.keras.backend as K
    if device_type == "GPU" or device_type == "gpu":
        # if no gpu is specified will use 0
        if device_type is not device:
            device_number = device[len(device_type)]
        else:
            device_number = "0"
        # set available gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
        print("Setting up Keras to use GPU with miminal memory on {}".format(device_number))
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
    elif device_type == "CPU" or device_type == "cpu":
        print("Setting up Keras to use CPU")
        config = tf.compat.v1.ConfigProto(device_count = {"GPU": 0})
    # set up session
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    sess = tf.compat.v1.Session(config=config)

def copytree(src, dst, symlinks=False, ignore=None):
    """Move the contents of a directory to a specified directory"""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

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

def KL(y_true, y_pred):
    """Return kullback-Leibler divergence"""
    y_true = K.exp(y_true)
    y_pred = K.exp(y_pred)
    P = (y_true / K.sum(y_true)) + K.epsilon()
    Q = y_pred / K.sum(y_pred) + K.epsilon()
    return K.sum(P * tf.math.log(P / Q))

def JSD(y_true, y_pred):
    """Compute the Jenson-Shannon divergence"""
    y_pred = K.exp(y_pred)
    y_true = K.exp(y_true)
    P = (y_true / K.sum(y_true)) + K.epsilon()
    Q = y_pred / K.sum(y_pred) + K.epsilon()
    const = K.constant(0.5)
    M = const * (P + Q)
    return const * K.sum(P * tf.math.log(P / M)) + const * K.sum(Q * tf.math.log(Q / M))

def get_parameters_from_json(model_path, verbose=1):
    """Get the parameters for the nn from the model.json file"""
    if verbose:
        print("Loading model: " + model_path)
    with open(model_path, "r") as read_file:
        params = json.load(read_file)
    return params

def network(n_inputs, parameters, verbose=1):
    """Get the model for neural network"""

    if type(n_inputs) == int:
        if n_inputs == 0:
            raise ValueError("Number of inputs must be non-zero")
        n_inputs = [n_inputs]
    else:
        if len(n_inputs) < 1 :
            raise ValueError("Must specifiy number of inputs")

    n_neurons = parameters["neurons"]
    try:
        n_mixed_neurons = parameters["mixed_neurons"]
    except:
        n_mixed_neurons = parameters["mixed neurons"]
    n_layers = parameters["layers"]
    try:
        n_mixed_layers = parameters["mixed_layers"]
    except:
        n_mixed_layers = parameters["mixed layers"]
    dropout_rate = parameters["dropout"]
    try:
        mixed_dropout_rate = parameters["mixed_dropout"]
    except:
        mixed_dropout_rate = parameters["mixed dropout"]
    try:
        batch_norm = parameters["batch_norm"]
    except:
        batch_norm = parameters["batch norm"]
    activation = parameters["activation"]
    regularization = parameters["regularization"]

    if not isinstance(n_neurons, (list, tuple, np.ndarray)):
        n_neurons = n_neurons * np.ones(n_layers, dtype=int)
    if not isinstance(n_mixed_neurons, (list, tuple, np.ndarray)):
        n_mixed_neurons = n_mixed_neurons * np.ones(n_mixed_layers, dtype=int)
    if not len(n_neurons) is n_layers:
        raise ValueError("Specified more layers than neurons")

    if activation == "erf":
        activation = tf.erf
    elif activation == "probit":
        def probit(x):
            """return probit of x"""
            normal = tf.distributions.Normal(loc=0., scale=1.)
            return normal.cdf(x)
        actiavtion = probit

    if regularization == "l1":
        reg = regularizers.l1(parameters["lambda"])
    elif regularization == "l2":
        reg = regularizers.l2(parameters["lambda"])
    else:
        print("Proceeding with no regularization")
        reg = None

    w_init = RandomNormal(mean=0.0, stddev=0.5)
    b_init = RandomUniform(minval=-0.1, maxval=0.1)

    inputs = []
    block_outputs = []
    for i, n in enumerate(n_inputs):
        layer = Input(shape=(n,), name="input_" + str(i + 1))
        inputs.append(layer)
        for j in range(n_layers):
            layer = Dense(n_neurons[i], activation=activation, kernel_regularizer=reg, name="p{}_dense_{}".format(i, j),
                          kernel_initializer=w_init, bias_initializer=b_init)(layer)
            if dropout_rate:
                layer = Dropout(dropout_rate)(layer)
            if batch_norm:
                layer = BatchNormalization()(layer)
        block_outputs.append(layer)
    if len(block_outputs) > 1:
        layer = concatenate(block_outputs, name="concat_blocks")
        for i in range(n_mixed_layers):
            layer = Dense(n_mixed_neurons[i], activation=activation, kernel_regularizer=reg, name="mix_dense_{}".format(i))(layer)
            if mixed_dropout_rate:
                layer = Dropout(mixed_dropout_rate)(layer)
            if batch_norm:
                outputs = BatchNormalization()(layer)
    # make final layer
    output_layer = Dense(1, activation="linear", name="output_dense")(layer)
    model = Model(inputs=inputs, outputs=output_layer)
    if verbose:
        model.summary()

    return model
