from __future__ import division
from functools import reduce
import logging
import numpy as np
from math import log,sqrt,fabs,exp, pi
from abc import ABCMeta,abstractmethod
import random
import os
from random import sample,gauss,randrange,uniform
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal, truncnorm
from scipy.special import logsumexp

import matplotlib.pyplot as plt

from cpnest.parameter import LivePoint


class Proposal(object):
    """
    Base abstract class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        Must be implemented by user

        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        pass

class EnsembleProposal(Proposal):
    """
    Base class for ensemble proposals
    """
    ensemble=None
    def set_ensemble(self,ensemble):
        """
        Set the ensemble of points to use
        """
        self.ensemble=ensemble

class ProposalCycle(EnsembleProposal):
    """
    A proposal that cycles through a list of
    jumps.

    Initialisation arguments:

    proposals : A list of jump proposals
    weights   : Weights for each type of jump

    Optional arguments:
    cyclelength : length of the proposal cycle. Default: 100

    """
    idx=0 # index in the cycle
    N=0   # number of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__()
        assert(len(weights)==len(proposals))
        self.cyclelength = cyclelength
        self.weights = weights
        self.proposals = proposals
        self.set_cycle()

    def set_cycle(self):
        # The cycle is a list of indices for self.proposals
        self.cycle = np.random.choice(self.proposals, size=self.cyclelength,
                                      p=self.weights, replace=True)
        self.N=len(self.cycle)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = self.normalise_weights(weights)

    def normalise_weights(self, weights):
        norm = sum(weights)
        for i, _ in enumerate(weights):
            weights[i]=weights[i] / norm
        return weights

    def get_sample(self,old,**kwargs):
        # Call the current proposal and increment the index
        self.idx = (self.idx + 1) % self.N
        p = self.cycle[self.idx]
        new = p.get_sample(old,**kwargs)
        self.log_J = p.log_J
        return new

    def set_ensemble(self,ensemble):
        """
        Updates the ensemble statistics
        by calling it on each :obj:`EnsembleProposal`
        """
        self.ensemble=ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

    def add_proposal(self, proposal, weight):
        self.proposals = self.proposals + [proposal]
        self.weights = self.weights + [weight]
        self.set_cycle()


class EnsembleWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65

    Draws a step by evolving along the
    direction of the center of mass of
    3 points in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        subset = sample(list(self.ensemble),self.Npoints)
        center_of_mass = reduce(type(old).__add__,subset)/float(self.Npoints)
        out = old
        for x in subset:
            out += (x - center_of_mass)*gauss(0,1)
        return out

class EnsembleStretch(EnsembleProposal):
    """
    The Ensemble "stretch" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65
    """
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        scale = 2.0 # Will stretch factor in (1/scale,scale)
        # Pick a random point to move toward
        a = random.choice(self.ensemble)
        # Pick the scale factor
        x = uniform(-1,1)*log(scale)
        Z = exp(x)
        out = a + (old - a)*Z
        # Jacobian
        self.log_J = out.dimension * x
        return out

class DifferentialEvolution(EnsembleProposal):
    """
    Differential evolution move:
    Draws a step by taking the difference vector between two points in the
    ensemble and adding it to the current point.
    See e.g. Exercise 30.12, p.398 in MacKay's book
    http://www.inference.phy.cam.ac.uk/mackay/itila/

    We add a small perturbation around the exact step
    """
    log_J = 0.0 # Symmetric jump
    def get_sample(self,old):
        """
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        a,b = sample(list(self.ensemble),2)
        sigma = 1e-4 # scatter around difference vector by this factor
        out = old + (b-a)*gauss(1.0,sigma)
        return out

class EnsembleEigenVector(EnsembleProposal):
    """
    A jump along a randomly-chosen eigenvector
    of the covariance matrix of the ensemble
    """
    log_J = 0.0
    eigen_values=None
    eigen_vectors=None
    covariance=None
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        super(EnsembleEigenVector,self).set_ensemble(ensemble)
        self.update_eigenvectors()

    def update_eigenvectors(self):
        """
        Recompute the eigenvectors and eigevalues
        of the covariance matrix of the ensemble
        """
        n=len(self.ensemble)
        dim = self.ensemble[0].dimension
        cov_array = np.zeros((dim,n))
        if dim == 1:
            name=self.ensemble[0].names[0]
            self.eigen_values = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.covariance = self.eigen_values
            self.eigen_vectors = np.eye(1)
        else:
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n): cov_array[i,j] = self.ensemble[j][name]
            self.covariance = np.cov(cov_array)
            self.eigen_values,self.eigen_vectors = np.linalg.eigh(self.covariance)

    def get_sample(self,old):
        """
        Propose a jump along a random eigenvector
        Parameters
        ----------
        old : :obj:`cpnest.parameter.LivePoint`

        Returns
        ----------
        out: :obj:`cpnest.parameter.LivePoint`
        """
        out = old
        # pick a random eigenvector
        i = randrange(old.dimension)
        jumpsize = sqrt(fabs(self.eigen_values[i]))*gauss(0,1)
        for k,n in enumerate(out.names):
            out[n]+=jumpsize*self.eigen_vectors[k,i]
        return out

class FlowProposal(EnsembleProposal):

    torch = __import__('torch')
    log_J = 0.0

    def __init__(self, model_dict=None, names=None, log_prior=None, device='cpu', prior_range=None, pool_size=10000, fuzz=1.0):

        super(FlowProposal, self).__init__()

        from .flowtrainer import FlowModel
        self.logger = logging.getLogger("CPNest")
        self.model = FlowModel(**model_dict, device=device)                     # Flow model
        self.ndims = model_dict["n_inputs"]
        self.mu = np.zeros(self.ndims)
        self.sigma = np.identity(self.ndims)
        self.worst_r2 = np.inf
        self.samples = None                    # Stored samples
        self.r2 = None                         # Radius squared
        self.populated = False                 # Is sample list populated
        self.names = names                     # Names of samples for LivePoint
        self.fuzz = fuzz

        self.prior = self.gaussian_prior

        self.log_prior = log_prior
        if prior_range is not None:
            self.setup_normalisation(prior_range)
        self.previous_sample = None

        self.pool_size = pool_size

    @staticmethod
    def unpack_live_point(point):
        """Return the necessary information from an instance of cpnest.parameter.LivePoint"""
        return np.frombuffer(point.values)

    def make_live_point(self, theta):
        """Create an instance of LivePoint with the given inputs"""
        return LivePoint(self.names, d=theta)

    def setup_normalisation(self, priors):
        """Setup the normalisation given the priors"""
        p_max = np.max(priors, axis=0)
        p_min = np.min(priors, axis=0)

        def rescale_input(theta):
            """Redefine the input rescaling"""
            return 2. * ((theta - p_min) / (p_max- p_min)) - 1

        def rescale_output(theta):
            """Redifine the output rescaling"""
            return ((p_max - p_min) * (theta + 1.) / 2.) + p_min

        # over-ride defaults
        self.rescale_input = rescale_input
        self.rescale_output = rescale_output

    @staticmethod
    def rescale_input(theta):
        """Placeholder method"""
        return theta

    @staticmethod
    def rescale_output(theta):
        """Placeholder method"""
        return theta

    def forward_pass(self, theta):
        """Pass a vector of points through the model"""
        theta_tensor = self.torch.Tensor(theta.astype(np.float32)).to(self.model.device)
        z, log_J = self.model(theta_tensor)
        z = z.detach().cpu().numpy()
        log_J = log_J.detach().cpu().numpy()
        return z, np.squeeze(log_J)

    def backward_pass(self, z):
        """A backwards pass from the model (latent -> real)"""
        z_tensor = self.torch.Tensor(z.astype(np.float32)).to(self.model.device)
        theta, log_J = self.model(z_tensor, mode='inverse')
        theta = theta.detach().cpu().numpy()
        log_J = log_J.detach().cpu().numpy()
        return theta, np.squeeze(log_J)

    def load_weights(self, weights_file):
        """Update the model from a weights file"""
        self.model.load_state_dict(self.torch.load(weights_file))
        self.model.eval()
        self.populated = False

    def radius2(self, z):
        """Calculate the radius of a latent_point"""
        return np.sum(z ** 2., axis=-1)

    def random_surface_nsphere(self, r=1, N=1000):
        """
        Draw N points uniformly on an n-sphere of radius r

        See Marsaglia (1972)
        """
        x = np.array([np.random.randn(N) for _ in range(self.ndims)])
        R = np.sqrt(np.sum(x ** 2., axis=0))
        z = x / R
        return r * z.T

    def random_nsphere(self, r=1, N=1000, fuzz=1.0):
        """
        Draw N points uniformly within an n-sphere of radius r
        """
        x = self.random_surface_nsphere(r=1, N=N)
        R = np.random.uniform(0, 1, N)
        z = R ** (1 / self.ndims) * x.T
        return fuzz * r * z.T

    def draw_uniform(self, r2, N=1000, fuzz=1.0):
        """Draw N samples from a region within the worst point"""
        z = self.random_nsphere(r=np.sqrt(r2), N=N, fuzz=fuzz)
        return z

    def draw_trunc(self, r2, N=1000, fuzz=1.0):
        """Draw N samples from a region within the worst point"""
        r = np.sqrt(r2)
        samples = np.concatenate([[truncnorm.rvs(-r, r, size=N)] for _ in range(self.ndims)], axis=0).T
        # remove points outside bounds
        samples = samples[np.where(np.sum(samples**2., axis=-1) < r2 )]
        return samples


    def uniform_prior(self, z):
        """
        Uniform prior for use with points drawn uniformly with an n-shpere
        """
        return 0.0

    def gaussian_prior(self, z):
        """
        Gaussian prior
        """
        return np.sum(-0.5 * (z ** 2.) - 0.5 * log(2. * pi), axis=-1)

    def log_proposal_prob(self, z, log_J):
        """
        Compute the proposal probaility for a given point assuming the latent
        distribution is a unit gaussian

        q(theta)  = q(z)|dz/dtheta|
        """
        log_q_z = self.prior(z)
        return log_q_z + log_J

    def compute_weights(self, theta, z, log_J):
        """
        Compute the weight for a given set of samples
        """
        log_q = self.log_proposal_prob(z, log_J)
        log_p = np.array([self.log_prior(t) for t in theta])
        return log_p - log_q


class RandomFlowProposal(FlowProposal):


    def __init__(self, model_dict=None, names=None, log_prior=None, device='cpu', prior_range=None, pool_size=10000, fuzz=1.0, output=None):

        super(RandomFlowProposal, self).__init__(model_dict, names, log_prior, device, prior_range, pool_size, fuzz)
        self.output = os.path.join(output, '')
        os.makedirs(self.output, exist_ok=True)
        self.count = 0
        self.draw = self.draw_trunc

    def save_samples(self, z, samples, accepted):
        """Save the samples generated"""
        samples = np.array([s.values for s in samples])
        flag = np.zeros(z.shape[0], dtype=bool)
        flag[accepted] = True
        flag = np.array([flag, flag])
        np.save(self.output + 'samples_{}.npy'.format(self.count), [z, samples, flag])
        self.count += 1


    def populate(self, old_r2, N=10000):
        """Populate a pool of latent points"""
        self.logger.debug("Populating proposal")
        # draw n samples and sort by radius
        if not self.count:
            fuzz = self.fuzz
        else:
            fuzz = self.fuzz
        z = self.draw(old_r2, N, fuzz=fuzz)
        self.r2 = self.radius2(z)
        samples, log_J = self.backward_pass(z)

        # rescale given priors used intially, need for priors
        samples = self.rescale_output(samples)
        samples = [self.make_live_point(p) for p in samples]
        alpha = self.compute_weights(samples, z, -log_J)

        # rejection sampling
        u = np.log(np.random.rand(len(samples)))
        indices = np.where((alpha - u) >= 0)[0]
        if len(indices):
            self.save_samples(z, samples, indices)
            # array of indices to take random draws from
            self.indices = np.random.permutation(len(indices))
            self.samples = [samples[i] for i in indices]
            self.z = z[indices]
            self.populated = True
        self.logger.debug('Proposal populated: {} / {} points accepted'.format(len(self.samples), len(alpha)))

    def get_sample(self, old_sample):
        """Get a new sample within the contour defined by the old sample"""
        count = 0
        while not self.populated:
            # Populate if unpopulated
            # This requires pass through the flows
            old_sample = self.unpack_live_point(old_sample) # unpack LivePoint
            old_sample = self.rescale_input(old_sample)     # rescale (-1, 1)
            old_z, _ = self.forward_pass(old_sample)        # to latent space
            old_r2 = self.radius2(old_z)                    # contour
            self.populate(old_r2, N=self.pool_size)         # points within contour

            # get samples with index given by first entry in possible indices
            samples = np.array([s.values for s in self.samples])
            fig, axs = plt.subplots(1, 2, figsize=(10,5))
            axs = axs.ravel()
            old_sample = self.rescale_output(old_sample)
            assert len(self.z) == len(samples)
            axs[0].plot(self.z[:, 0][self.indices], self.z[:,1][self.indices], '.')
            axs[0].plot(old_z[0], old_z[1], 'o')
            axs[1].plot(samples[:, 0], samples[:, 1], '.')
            axs[1].plot(old_sample[0], old_sample[1], 'o', label='Worst point: {}'.format(old_sample))
            plt.legend()
            fig.savefig(self.output + 'samples_{}.png'.format(self.count-1))
            plt.close(fig)
            count += 1
            if count == 20:
                break
            # TODO: fix this



        new_sample = self.samples[self.indices[0]]
        self.indices = self.indices[1:]
        if not self.indices.size:
            self.populated = False
            self.logger.debug('Pool of points is empty')
        # make live point and return
        return new_sample


class DefaultProposalCycle(ProposalCycle):
    """
    A default proposal cycle that uses the
    :obj:`cpnest.proposal.EnsembleWalk`, :obj:`cpnest.proposal.EnsembleStretch`,
    :obj:`cpnest.proposal.DifferentialEvolution`, :obj:`cpnest.proposal.EnsembleEigenVector`
    ensemble proposals.
    """
    def __init__(self):

        proposals = [EnsembleWalk(),
                     EnsembleStretch(),
                     DifferentialEvolution(),
                     EnsembleEigenVector()]
        weights = [3,
                   3,
                   1,
                   10]
        super(DefaultProposalCycle,self).__init__(proposals, weights)

class HamiltonianProposalCycle(ProposalCycle):
    def __init__(self, model=None):
        """
        A proposal cycle that uses the hamiltonian :obj:`ConstrainedLeapFrog`
        proposal.
        Requires a :obj:`cpnest.Model` to be passed for access to the user-defined
        :obj:`cpnest.Model.force` (the gradient of :obj:`cpnest.Model.potential`) and
        :obj:`cpnest.Model.log_likelihood` to define the reflective
        """
        weights = [1]
        proposals = [ConstrainedLeapFrog(model=model)]
        super(HamiltonianProposalCycle,self).__init__(proposals, weights)

class HamiltonianProposal(EnsembleEigenVector):
    """
    Base class for hamiltonian proposals
    """
    mass_matrix = None
    inverse_mass_matrix = None
    momenta_distribution = None

    def __init__(self, model=None, **kwargs):
        """
        Initialises the class with the kinetic
        energy and the :obj:`cpnest.Model.potential`.
        """
        super(HamiltonianProposal, self).__init__(**kwargs)
        self.T              = self.kinetic_energy
        self.V              = model.potential
        self.normal         = None
        self.dt             = 0.3
        self.base_dt        = 0.3
        self.scale          = 1.0
        self.L              = 20
        self.base_L         = 20
        self.TARGET         = 0.500
        self.ADAPTATIONSIZE = 0.001
        self._initialised   = False
        self.c              = self.counter()
        self.DEBUG          = 0

    def set_ensemble(self, ensemble):
        """
        override the set ensemble method
        to update masses, momenta distribution
        and to heuristically estimate the normal vector to the
        hard boundary defined by logLmin.
        """
        super(HamiltonianProposal,self).set_ensemble(ensemble)
        self.update_mass()
        self.update_normal_vector()
        self.update_momenta_distribution()

    def update_normal_vector(self):
        """
        update the constraint by approximating the
        loglikelihood hypersurface as a spline in
        each dimension.
        This is an approximation which
        improves as the algorithm proceeds
        """
        n = self.ensemble[0].dimension
        tracers_array = np.zeros((len(self.ensemble),n))
        for i,samp in enumerate(self.ensemble):
            tracers_array[i,:] = samp.values
        V_vals = np.atleast_1d([p.logL for p in self.ensemble])

        self.normal = []
        for i,x in enumerate(tracers_array.T):
            # sort the values
#            self.normal.append(lambda x: -x)
            idx = x.argsort()
            xs = x[idx]
            Vs = V_vals[idx]
            # remove potential duplicate entries
            xs, ids = np.unique(xs, return_index = True)
            Vs = Vs[ids]
            # pick only finite values
            idx = np.isfinite(Vs)
            Vs  = Vs[idx]
            xs  = xs[idx]
            # filter to within the 90% range of the Pvals
            Vl,Vh = np.percentile(Vs,[5,95])
            (idx,) = np.where(np.logical_and(Vs > Vl,Vs < Vh))
            Vs = Vs[idx]
            xs = xs[idx]
            # Pick knots for this parameters: Choose 5 knots between
            # the 1st and 99th percentiles (heuristic tuning WDP)
            knots = np.percentile(xs,np.linspace(1,99,5))
            # Guesstimate the length scale for numerical derivatives
            dimwidth = knots[-1]-knots[0]
            delta = 0.1 * dimwidth / len(idx)
            # Apply a Savtzky-Golay filter to the likelihoods (low-pass filter)
            window_length = len(idx)//2+1 # Window for Savtzky-Golay filter
            if window_length%2 == 0: window_length += 1
            f = savgol_filter(Vs, window_length,
                              5, # Order of polynominal filter
                              deriv=1, # Take first derivative
                              delta=delta, # delta for numerical deriv
                              mode='mirror' # Reflective boundary conds.
                              )
            # construct a LSQ spline interpolant
            self.normal.append(LSQUnivariateSpline(xs, f, knots, ext = 3, k = 3))
            if self.DEBUG: np.savetxt('dlogL_spline_%d.txt'%i,np.column_stack((xs,Vs,self.normal[-1](xs),f)))

    def unit_normal(self, q):
        """
        Returns the unit normal to the iso-Likelihood surface
        at x, obtained from the spline interpolation of the
        directional derivatives of the likelihood
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        n: :obj:`numpy.ndarray` unit normal to the logLmin contour evaluated at q
        """
        v               = np.array([self.normal[i](q[n]) for i,n in enumerate(q.names)])
        v[np.isnan(v)]  = -1.0
        n               = v/np.linalg.norm(v)
        return n

    def gradient(self, q):
        """
        return the gradient of the potential function as numpy ndarray
        Parameters
        ----------
        q : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        dV: :obj:`numpy.ndarray` gradient evaluated at q
        """
        dV = self.dV(q)
        return dV.view(np.float64)

    def update_momenta_distribution(self):
        """
        update the momenta distribution using the
        mass matrix (precision matrix of the ensemble).
        """
        self.momenta_distribution = multivariate_normal(cov=self.mass_matrix)#

    def update_mass(self):
        """
        Update the mass matrix (covariance matrix) and
        inverse mass matrix (precision matrix)
        from the ensemble, allowing for correlated momenta
        """
        self.d                      = self.covariance.shape[0]
        self.inverse_mass_matrix    = np.atleast_2d(self.covariance)
        self.mass_matrix            = np.linalg.inv(self.inverse_mass_matrix)
        self.inverse_mass           = np.atleast_1d(np.squeeze(np.diag(self.inverse_mass_matrix)))
        _, self.logdeterminant      = np.linalg.slogdet(self.mass_matrix)
        if self._initialised == False:
            self.set_integration_parameters()

    def set_integration_parameters(self):
        """
        Set the integration length according to the N-dimensional ellipsoid
        shortest and longest principal axes. The former sets to base time step
        while the latter sets the trajectory length
        """
        ranges = [self.prior_bounds[j][1] - self.prior_bounds[j][0] for j in range(self.d)]

        l, h = np.min(ranges), np.max(ranges)

        self.base_L         = 10+int((h/l)*self.d**(1./4.))
        self.base_dt        = (1.0/self.base_L)*l/h
        self._initialised   = True


    def update_time_step(self, acceptance):
        """
        Update the time step according to the
        acceptance rate
        Parameters
        ----------
        acceptance : :obj:'numpy.float'
        """
        diff = acceptance - self.TARGET
        new_log_scale = np.log(self.scale) + self.ADAPTATIONSIZE * diff
        self.scale = np.exp(new_log_scale)
        self.dt = self.base_dt * self.scale

    def update_trajectory_length(self,nmcmc):
        """
        Update the trajectory length according to the estimated ACL
        Parameters
        ----------
        nmcmc :`obj`:: int
        """
        self.L = self.base_L + np.random.randint(nmcmc,5*nmcmc)

    def kinetic_energy(self,p):
        """
        kinetic energy part for the Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum

        Returns
        ----------
        T: :float: kinetic energy
        """
        return 0.5 * np.dot(p,np.dot(self.inverse_mass_matrix,p))-self.logdeterminant-0.5*self.d*np.log(2.0*np.pi)

    def hamiltonian(self, p, q):
        """
        Hamiltonian.
        Parameters
        ----------
        p : :obj:`numpy.ndarray`
            momentum
        q : :obj:`cpnest.parameter.LivePoint`
            position
        Returns
        ----------
        H: :float: hamiltonian
        """
        return self.T(p) + self.V(q)

class LeapFrog(HamiltonianProposal):
    """
    Leap frog integrator proposal for an unconstrained
    Hamiltonian Monte Carlo step
    """
    def __init__(self, model=None, **kwargs):
        """
        Parameters
        ----------
        model : :obj:`cpnest.Model`
        """
        super(LeapFrog, self).__init__(model=model, **kwargs)
        self.dV             = model.force
        self.prior_bounds   = model.bounds

    def get_sample(self, q0, *args):
        """
        Propose a new sample, starting at q0

        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        # generate a canonical momentum
        p0 = np.atleast_1d(self.momenta_distribution.rvs())
        initial_energy = self.hamiltonian(p0,q0)
        # evolve along the trajectory
        q, p = self.evolve_trajectory(p0, q0, *args)
        # minus sign from the definition of the potential
        final_energy   = self.hamiltonian(p,q)
        self.log_J = min(0.0, initial_energy-final_energy)
        return q

    def evolve_trajectory(self, p0, q0, *args):
        """
        Hamiltonian leap frog trajectory subject to the
        hard boundary defined by the parameters prior bounds.
        https://arxiv.org/pdf/1206.1901.pdf

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        # Updating the momentum a half-step
        p = p0 - 0.5 * self.dt * self.gradient(q0)
        q = q0.copy()

        for i in range(self.L):
            # do a step
            for j,k in enumerate(q.names):
                u,l = self.prior_bounds[j][1], self.prior_bounds[j][0]
                q[k] += self.dt * p[j] * self.inverse_mass[j]
                # check and reflect against the bounds
                # of the allowed parameter range
                while q[k] <= l or q[k] >= u:
                    if q[k] > u:
                        q[k] = u - (q[k] - u)
                        p[j] *= -1
                    if q[k] < l:
                        q[k] = l + (l - q[k])
                        p[j] *= -1

            dV = self.gradient(q)
            # take a full momentum step
            p += - self.dt * dV
        # Do a final update of the momentum for a half step
        p += - 0.5 * self.dt * dV

        return q, -p

class ConstrainedLeapFrog(LeapFrog):
    """
    Leap frog integrator proposal for a costrained
    (logLmin defines a reflective boundary)
    Hamiltonian Monte Carlo step.
    """
    def __init__(self, model=None, **kwargs):
        """
        Parameters
        ----------
        model : :obj:`cpnest.Model`
        """
        super(ConstrainedLeapFrog, self).__init__(model=model, **kwargs)
        self.log_likelihood = model.log_likelihood

    def get_sample(self, q0, logLmin=-np.inf):
        """
        Generate new sample with constrained HMC, starting at q0.

        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
            position

        logLmin: hard likelihood boundary

        Returns
        ----------
        q: :obj:`cpnest.parameter.LivePoint`
            position
        """
        return super(ConstrainedLeapFrog,self).get_sample(q0, logLmin)

    def counter(self):
        n = 0
        while True:
            yield n
            n += 1

    def evolve_trajectory_one_step_position(self, p, q):
        """
        One leap frog step in position

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint` position
        """
        for j,k in enumerate(q.names):
            u, l  = self.prior_bounds[j][1], self.prior_bounds[j][0]
            q[k]  += self.dt * p[j] * self.inverse_mass[j]
            # check and reflect against the bounds
            # of the allowed parameter range
            while q[k] < l or q[k] > u:
                if q[k] > u:
                    q[k] = u - (q[k] - u)
                    p[j] *= -1
                if q[k] < l:
                    q[k] = l + (l - q[k])
                    p[j] *= -1
        return p, q

    def evolve_trajectory_one_step_momentum(self, p, q, logLmin, half = False):
        """
        One leap frog step in momentum

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position
        logLmin: :obj:`numpy.float64`
            loglikelihood constraint
        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint` position
        """
        reflected = 0
        dV        = self.gradient(q)
        if half is True:
            p += - 0.5 * self.dt * dV
            return p, q, reflected
        else:
            c = self.check_constraint(q, logLmin)
            if c > 0:
                p += - self.dt * dV
            else:
                normal = self.unit_normal(q)
                p += - 2.0*np.dot(p,normal)*normal
                reflected = 1
        return p, q, reflected

    def check_constraint(self, q, logLmin):
        """
        Check the likelihood

        Parameters
        ----------
        q0 : :obj:`cpnest.parameter.LivePoint`
        position
        logLmin: :obj:`numpy.float64`
        loglikelihood constraint
        Returns
        ----------
        c: :obj:`numpy.float64` value of the constraint
        """
        q.logP  = -self.V(q)
        q.logL  = self.log_likelihood(q)
        return q.logL - logLmin

    def evolve_trajectory(self, p0, q0, logLmin):
        """
        Evolve point according to Hamiltonian method in
        https://arxiv.org/pdf/1005.0157.pdf

        Parameters
        ----------
        p0 : :obj:`numpy.ndarray`
            momentum
        q0 : :obj:`cpnest.parameter.LivePoint`
            position

        Returns
        ----------
        p: :obj:`numpy.ndarray` updated momentum vector
        q: :obj:`cpnest.parameter.LivePoint` position
        """

        trajectory = [(q0,p0)]
        # evolve forward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.L):
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy()))
            i += 1

        # evolve backward in time
        i = 0
        p, q, reflected = self.evolve_trajectory_one_step_momentum(-p0.copy(), q0.copy(), logLmin, half = True)
        while (i < self.L):
            p, q            = self.evolve_trajectory_one_step_position(p, q)
            p, q, reflected = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = False)
            trajectory.append((q.copy(),p.copy()))
            i += 1
#            if i == 3*self.L: break
        p, q, reflected     = self.evolve_trajectory_one_step_momentum(p, q, logLmin, half = True)

        if self.DEBUG: self.save_trajectory(trajectory, logLmin)
        return self.sample_trajectory(trajectory)
#        print("dt:",self.dt,"L:",self.L,"actual L:",i,"maxL:",3*self.L)
#        return trajectory[-1]

    def sample_trajectory(self, trajectory):
        """

        """
        logw = np.array([-self.hamiltonian(p,q) for q,p in trajectory[1:-1]])
        norm = logsumexp(logw)
        idx  = np.random.choice(range(1,len(trajectory)-1), p = np.exp(logw  - norm))
        return trajectory[idx]

    def save_trajectory(self, trajectory, logLmin, filename = None):
        """
        save trajectory for diagnostic purposes
        """
        if filename is None:
            filename = 'trajectory_'+str(next(self.c))+'.txt'
        f = open(filename,'w')
        names = trajectory[0][0].names

        for n in names:
            f.write(n+'\t'+'p_'+n+'\t')
        f.write('logPrior\tlogL\tlogLmin\n')

        for j,step in enumerate(trajectory):
            q = step[0]
            p = step[1]
            for j,n in enumerate(names):
                f.write(repr(q[n])+'\t'+repr(p[j])+'\t')
            f.write(repr(q.logP)+'\t'+repr(q.logL)+'\t'+repr(logLmin)+'\n')
        f.close()
        if self.c == 3: exit()
