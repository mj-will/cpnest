from __future__ import division, print_function

import numpy as np
from .trainer import Trainer

class FATrainer(Trainer):

    def __init__(self, input_dict, manager=None, output='./'):
        self.input_dict = input_dict
        super(FATrainer, self).__init__(manager=manager, output=output)

    def initialise(self):
        """
        Import functions that use tensorflow and initialise the function approximator
        """
        from .fa_utils import set_keras_device
        set_keras_device("gpu0")
        self.input_dict["parameters"]["tmpdir"] = self.output + "tmpdir_fa/"
        from .function_approximator import FunctionApproximator
        self.fa = FunctionApproximator(input_dict=self.input_dict, verbose=1)
        self.fa.setup_normalisation(self.fa.priors, normalise_output=False)
        # save so it's easier to init other fa
        self.fa._make_run_dir(self.output)
        #self.attr_dict = self.fa.save_approximator()    # TODO: Speicify path correctly
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
        true_stats, pred_stats = self.fa.train_on_data(x, y, accumulate="all", plot=True)

        mean_ratio = true_stats[0] / pred_stats[0]
        std_ratio = true_stats[1] / pred_stats[1]

        if (np.abs(1. - mean_ratio) < 0.05) and (np.abs(1. - std_ratio) < 0.05):
            self.manager.use_fa.value = 1

        self.manager.trained.value = 1
        self.producer_pipe.send(self.fa.weights_file)
