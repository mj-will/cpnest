
import numpy as np

from ..flowtrainer import FlowTrainer
from .reparameterise import GWReparam

class GWFlowTrainer(GWReparam, FlowTrainer):
    """
    A normalising flow trainer specifically for gravitational wave parameter
    esimation

    * psi: (0, pi) -> scale = 2.0
    * phase: (0, 2pi) -> scale=1.0

    """
    def __init__(self, **kwargs):
        parameters = kwargs['cpnest_model'].names
        reparameterisations = kwargs['trainer_dict']['reparameterisations']

        #if 'q_inversion' in kwargs['trainer_dict'].keys():
        #        kwargs['q_inversion'] = kwargs['trainer_dict']['q_inversion']

        super(GWFlowTrainer, self).__init__(parameters=parameters,
                reparameterisations=reparameterisations, **kwargs)

        # update mask to array
        #if 'mask' in self.model_dict.keys():
        #    self.mask = self.model_dict['mask'].copy()

    def setup_normalisation(self):
        """
        Setup normalisation using the priors

        ra, dec, psi, phase -> no normalisation
        dL -> [0, 1]
        masses -> [-1, 1]
        """
        self.setup_parameter_indices()
        # need to have updated input size for setting up model
        self.n_inputs = self.reparam_dim
        self.model_dict['n_inputs'] = self.reparam_dim

        self._prior_min = self.prior_bounds[self.defaults, 0]
        self._prior_max = self.prior_bounds[self.defaults, 1]

        self.normalise_samples = self.reparameterise
        self.rescale_samples = self.inverse_reparameterise
