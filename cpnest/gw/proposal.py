
import numpy as np
from scipy.stats import chi, multivariate_normal

from ..proposal import RandomFlowProposal, NaiveProposal
from ..flowtrainer import plot_samples
from .reparameterise import GWReparam

import matplotlib.pyplot as plt

class GWNaiveProposal(NaiveProposal):

    def __init__(self, **kwargs):
        super(GWNaiveProposal, self).__init__(**kwargs)

        all_gaussian_parameters = ['q_0', 'q_1', 'q_2', 'q_3']
        print(self.prior_bounds)
        self.gaussian_mask = list()
        self.uniform_mask = list()
        for i, p in enumerate(self.names):
            if p in all_gaussian_parameters:
                self.gaussian_mask.append(i)
            else:
                self.uniform_mask.append(i)

        self.prior_bounds = self.prior_bounds[self.uniform_mask]
        if len(self.prior_bounds):
            self.prior_denom = np.ptp(self.prior_bounds)

        if len(self.gaussian_mask):
            self.normal = multivariate_normal(np.zeros(len(self.gaussian_mask)),
                    np.eye(len(self.gaussian_mask)))

        if np.isnan(self.prior_bounds).any():
            raise RuntimeError('Supplied parameter with no bounds that is not gaussian!')

        if not len(self.uniform_mask):
            self.draw_proposal = self.draw_gaussian
            self.log_proposal = self.gaussian_proposal
        elif not len(self.gaussian_mask):
            self.draw_proposal = self.draw_uniform
            self.log_proposal = self.uniform_proposal
        else:
            self.draw_proposal = self.draw_mixed
            self.log_proposal = self.mixed_proposal

        # update denominator for uniform priors

    def draw_uniform(self):
        """Draw from a uniform distribution"""
        return np.random.uniform(self.prior_bounds[:,0], self.prior_bounds[:,1],
                [self.N, len(self.uniform_mask)])

    def draw_gaussian(self):
        """Draw from a normal distribution"""
        return np.random.randn(self.N, len(self.gaussian_mask))

    def draw_mixed(self):
        """Draw from the proposal distribution"""
        samples = np.empty([self.N, self.dims], dtype=np.float)
        samples[:, self.gaussian_mask] = self.draw_gaussian()
        samples[:, self.uniform_mask] = self.draw_uniform()
        return samples

    def uniform_proposal(self, theta):
        """Uniform proposal probability"""
        return - np.log(self.prior_denom)

    def gaussian_proposal(self, theta):
        """Gaussian proposal probability"""
        theta = self.unpack_live_point(theta)
        return self.normal.logpdf([theta[self.gaussian_mask]])

    def mixed_proposal(self, theta):
        """Compute probability for mixed proposal"""
        return self.uniform_proposal(theta) + self.gaussian_proposal(theta)

class GWFlowProposal(GWReparam, RandomFlowProposal):

    def __init__(self, **kwargs):
        parameters =  kwargs['names']   # used for GWReparam
        print('Proposal parameters:', parameters)
        super(GWFlowProposal, self).__init__(parameters=parameters,
                initialise=False, **kwargs)
        # setup normalisation with GWReparam functions aviablable
        print('Proposal dim:', self.reparam_dim)
        # update mask to array
        if 'mask' in self.model_dict.keys():
            mask = self.model_dict['mask']
            self.model_dict['mask'] = self.get_mask(mask)
        self.initialise()

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
        return log_p

    def compute_weights(self, theta, z, radial_components, log_J):
        """
        Compute the weight for a given set of samples
        """
        log_q = self.log_proposal_prob(z, log_J)
        log_p_phys = np.array([self.log_prior(t) for t in theta])
        log_p_radial = self.log_prior_radials(radial_components)
        log_p = log_p_phys + log_p_radial
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def populate(self, old_r2, N=10000):
        """Populate a pool of latent points"""
        self.logger.debug(f"Populating proposal for r2: {old_r2}")
        # draw n samples and sort by radius
        # only proceed if the draw has produced points
        # mainly an issue with drawing from the gaussian
        self.samples = []
        self.z = np.empty([0, self.ndims])
        pn = 0
        while len(self.samples) < N:
            while True:
                z = self.draw(old_r2, N, fuzz=self.fuzz)
                if z.size:
                    break
            samples, log_J = self.backward_pass(z)
            log_J *= -1.
            log_J += self.compute_log_jacobian(samples)
            if not pn:
                plot_samples(z, samples, output=self.output,
                        filename=f'samples_{self.count}_{pn}.png',
                        names=self.re_parameters, c=log_J)
                # rescale given priors used intially, need for priors
            samples = self.rescale_output(samples)
            if not pn:
                plot_samples(z, samples, output=self.output,
                        filename=f'rescaled_samples_{self.count}_{pn}.png',
                        names=self.parameters, c=log_J)
            radial_components = samples[:, self.base_dim:]
            # seperate physical samples from radial components
            samples = [self.make_live_point(p) for p in samples[:, :self.base_dim]]
            # TODO: will this fail if self.base_dim == samples.shape[0]
            log_w = self.compute_weights(samples, z, radial_components, log_J)
            finite = np.isfinite(log_w)
            #log_w = log_[f]
            #samples = [s for f,s in zip(finite, samples)]
            #print(log_w)
            # rejection sampling
            log_u = np.log(np.random.rand(len(samples)))
            if not pn:
                fig = plt.figure()
                plt.hist(log_w[finite], alpha=0.5, label='log w')
                plt.hist(log_u[finite], alpha=0.5, label='log u')
                plt.legend()
                fig.savefig(self.output + f'/weights_hist_{self.count}_{pn}.png')

            pn += 1
            indices = np.where((log_w - log_u) >= 0)[0]
            if not len(indices):
                self.logger.error('Rejection sampling produced zero samples!')
                raise RuntimeError('Rejection sampling produced zero samples!')
            if len(indices) / len(samples) < 0.01:
                self.logger.error('Rejection sampling accepted less than 1 percent of samples!')
                #raise RuntimeError('Rejection sampling accepted less than 1 percent of samples!')
            # array of indices to take random draws from
            self.samples += [samples[i] for i in indices]
            self.z = np.concatenate([self.z, z[indices]], axis=0)
            self.logger.debug(f'Proposal: accpetance: {len(indices)/len(samples):.3}, {len(self.samples)} / {N} total points accepted')

        self.samples = self.samples[:N]
        self.z = self.z[:N]
        self.indices = np.random.permutation(len(self.samples))
        self.populated = True
        self.logger.debug('Proposal populated: {} / {} points accepted'.format(len(self.samples), N))
