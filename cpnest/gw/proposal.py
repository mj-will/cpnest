
import numpy as np
from scipy.stats import chi, multivariate_normal

from ..proposal import RandomFlowProposal, EnsembleProposal
from ..flowtrainer import plot_samples
from .reparameterise import GWReparam

import matplotlib.pyplot as plt

class GWNaiveProposal(EnsembleProposal):
    """
    Draw from GW priors for un-informed rejection sample

    Inhereits from EnsembleProposal for `set_ensemble` method
    """
    def __init__(self, model=None, **kwargs):
        """Intialise"""
        # Bilby adds new point method to draw from prior
        # use this to draw from GW priors without need to check which
        # parameters are being used
        self.new_point = model.new_point
        self.empty = False

    def get_sample(self, old_sample):
        """Draw a sample from the GW priors"""
        return self.new_point()

class GWFlowProposal(GWReparam, RandomFlowProposal):

    def __init__(self, bilby_priors=False, **kwargs):
        parameters =  kwargs['names']   # used for GWReparam
        print('Proposal parameters:', parameters)
        super(GWFlowProposal, self).__init__(parameters=parameters,
                initialise=False, **kwargs)
        # setup normalisation with GWReparam functions aviablable
        print('Proposal dim:', self.reparam_dim)
        print('Bilby priors:', bilby_priors)
        if bilby_priors:
            self.enable_bibly_priors(bilby_priors, parameters)
            self.default_priors = False
        else:
            self.default_priors = True


        self.swap_enabled=False
        # update mask to array
        if 'mask' in self.model_dict.keys():
            mask = self.model_dict['mask']
            self.model_dict['mask'] = self.get_mask(mask)
        self.initialise()


    def enable_bibly_priors(self, bilby_priors, base_parameters):
        """Use bibly priors to reduce cost of computing log_prior"""

        for k in base_parameters:
            if k not in bilby_priors.keys():
                bilby_priors.pop(k)

        def _log_prior(theta):
            """Log prior that takes numpy arrays"""
            log_p = 0
            for i, p in enumerate(base_parameters):
                log_p += bilby_priors[p].ln_prob(theta[:, i])

            return log_p

        self.log_prior_array = _log_prior

    def setup_normalisation(self, prior_bounds):
        """Setup the normalisation given the priors"""
        self.setup_parameter_indices()
        # need to have updated input size for setting up model
        self.ndims = self.reparam_dim
        self.model_dict['n_inputs'] = self.reparam_dim

        self._prior_min = prior_bounds[self.defaults, 0]
        self._prior_max = prior_bounds[self.defaults, 1]

        # over-ride defaults using functions from GWReparam
        self.rescale_input = self.reparameterise
        self.rescale_output = self.inverse_reparameterise

    def log_prior_radials(self, radial_components):
        """
        Compute the log prior for the radial components
        """
        log_p = 0.
        if 'psi' in self.reparameterisations:
            log_p += chi.logpdf(radial_components[:, self.psi_radial-self.base_dim], 2)
        if 'phase' in self.reparameterisations:
            log_p += chi.logpdf(radial_components[:, self.phase_radial-self.base_dim], 2)
        if 'sky' in self.reparameterisations:
            if self.sky_radial:
                log_p += chi.logpdf(radial_components[:, self.sky_radial-self.base_dim], 3)
        if self.angle_indices:
            for i in self.angle_indices:
                if not i[2]:
                    log_p + chi.logpdf(radial_components[:, i[1]-self.base_dim], 2)

        return log_p

    def compute_weights(self, theta, z, radial_components, log_J):
        """
        Compute the weight for a given set of samples
        """
        log_q = self.log_proposal_prob(z, log_J)
        # This is the bottle neck in the whole process
        # Would be geod to vectorise this function if possible
        if self.default_priors:
            log_p_phys = np.array([self.log_prior(t) for t in theta])
        else:
            log_p_phys = self.log_prior_array(theta)
        log_p_radial = self.log_prior_radials(radial_components)
        log_p = log_p_phys + log_p_radial
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def reset_proposal(self):
        """Reset proposal to default"""
        if self.proposal == 'uniform':
            self.prior = self.uniform_prior
            self.draw = self.draw_uniform
        else:
            self.prior = self.gaussian_prior
            self.draw = self.draw_trunc

    def swap_proposal(self, acceptance, old_r2, N):
        """Swap to the other proposal distribution if the accpetance is higher"""
        orig_prior = self.prior
        orig_draw = self.draw
        if self.proposal == 'gaussian':
            self.prior = self.uniform_prior
            self.draw = self.draw_uniform
        else:
            self.prior = self.gaussian_prior
            self.draw = self.draw_trunc

        while True:
            z = self.draw(old_r2, N, fuzz=self.fuzz)
            if z.size:
                break
        samples, log_J = self.backward_pass(z)
        log_J *= -1.
        log_J += self.compute_log_jacobian(samples)
        samples = self.rescale_output(samples)
        radial_components = samples[:, self.base_dim:]
        if self.default_priors:
            samples = [self.make_live_point(p) for p in samples[:, :self.base_dim]]
        log_w = self.compute_weights(samples, z, radial_components, log_J)
        if not self.default_priors:
            samples = [self.make_live_point(p) for p in samples[:, :self.base_dim]]
        log_u = np.log(np.random.rand(len(samples)))
        indices = np.where((log_w - log_u) >= 0)[0]
        if (len(indices) / len(samples)) > acceptance:
            self.logger.debug(f'Switching proposal with improved acceptance: {len(indices) / len(samples)}')
            self.samples = []
            self.z = np.empty([0, self.ndims])
            return True
        else:
            self.logger.debug('Not switching proposal')
            self.reset_proposal()
            return False


    def populate(self, old_r2, N=10000):
        """Populate a pool of latent points"""
        self.logger.debug(f"Populating proposal for r2: {old_r2}")
        # draw n samples and sort by radius
        # only proceed if the draw has produced points
        # mainly an issue with drawing from the gaussian
        self.samples = []
        self.z = np.empty([0, self.ndims])
        pn = 0
        swap = False
        while len(self.samples) < N:
            while True:
                z = self.draw(old_r2, N, fuzz=self.fuzz)
                if z.size:
                    break
            samples, log_J = self.backward_pass(z)
            log_J *= -1.
            log_J += self.compute_log_jacobian(samples)
            #if not pn:

            #    #plot_samples(z, samples, output=self.output,
            #            filename=f'samples_{self.count}_{pn}.png',
            #            names=self.re_parameters, c=log_J)
            #    # rescale given priors used intially, need for priors
            samples = self.rescale_output(samples)
            #if not pn:
            #    plot_samples(z, samples, output=self.output,
            #            filename=f'rescaled_samples_{self.count}_{pn}.png',
            #            names=self.parameters, c=log_J)
            radial_components = samples[:, self.base_dim:]
            # seperate physical samples from radial components
            if self.default_priors:
                samples = [self.make_live_point(p) for p in samples[:, :self.base_dim]]
            # TODO: will this fail if self.base_dim == samples.shape[0]

            # this is the bottle neck
            # TODO: fix this!!
            log_w = self.compute_weights(samples, z, radial_components, log_J)
            if not self.default_priors:
                samples = [self.make_live_point(p) for p in samples[:, :self.base_dim]]
            # rejection sampling
            log_u = np.log(np.random.rand(len(samples)))

            pn += 1
            indices = np.where((log_w - log_u) >= 0)[0]
            acceptance = len(indices)/len(samples)
            #self.logger.debug(f'Proposal acceptance: {acceptance:.3}')
            if not len(indices):
                self.logger.error('Rejection sampling produced zero samples!')
                raise RuntimeError('Rejection sampling produced zero samples!')
            if acceptance < 0.01:
                #self.logger.warning('Rejection sampling accepted less than 1 percent of samples!')
                if not swap and pn < 5:
                    if self.swap_enabled:
                        self.logger.debug('Swapping to other proposal')
                        swap = True
                        flag = self.swap_proposal(acceptance, old_r2, N)
                        # if swapped reset counter
                        if flag:
                            pn = 0
            # array of indices to take random draws from
            self.samples += [samples[i] for i in indices]
            self.z = np.concatenate([self.z, z[indices]], axis=0)
            #self.logger.debug(f'Proposal progress: {len(self.samples)} / {N} total points accepted')

        #self.samples = self.samples[:N]
        #self.z = self.z[:N]
        self.indices = np.random.permutation(len(self.samples))
        self.populated = True
        if swap:
            self.reset_proposal()
        self.logger.debug('Proposal populated: {} / {} points accepted'.format(len(self.samples), N))
