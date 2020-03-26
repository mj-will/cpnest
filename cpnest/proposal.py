from __future__ import division
from functools import reduce
import numpy as np
from math import log,sqrt,fabs,exp
from abc import ABCMeta,abstractmethod
import random
from random import sample,gauss,randrange,uniform

from scipy.stats import truncnorm

from cpnest.parameter import LivePoint
from cpnest.flows import setup_model

class Proposal(object):
    """
    Base class for jump proposals
    """
    __metaclass__ = ABCMeta
    log_J = 0.0 # Jacobian of this jump proposal
    @abstractmethod
    def get_sample(self,old):
        """
        Returns a new proposed sample given the old one.
        Must be implemented by user
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
    cyclelength : length of the propsal cycle

    """
    idx=0 # index in the cycle
    N=0   # numer of proposals in the cycle
    def __init__(self,proposals,weights,cyclelength=100,*args,**kwargs):
        super(ProposalCycle,self).__init__(*args,**kwargs)
        assert(len(weights)==len(proposals))
        # Normalise the weights
        norm = sum(weights)
        for i,_ in enumerate(weights):
            weights[i]=weights[i]/norm
        self.proposals=proposals
        # The cycle is a list of indices for self.proposals
        self.cycle = np.random.choice(self.proposals, size = cyclelength, p=weights, replace=True)
        self.N=len(self.cycle)

    def get_sample(self,old):
        # Call the current proposal and increment the index
        p = self.cycle[self.idx]
        new = p.get_sample(old)
        self.log_J = p.log_J
        self.idx = (self.idx + 1) % self.N
        return new

    def set_ensemble(self,ensemble):
        # Calls set_ensemble on each proposal that is of ensemble type
        self.ensemble=ensemble
        for p in self.proposals:
            if isinstance(p,EnsembleProposal):
                p.set_ensemble(self.ensemble)

class EnsembleWalk(EnsembleProposal):
    """
    The Ensemble "walk" move from Goodman & Weare
    http://dx.doi.org/10.2140/camcos.2010.5.65

    Draws a step with the sample covariance of the points
    in the ensemble.
    """
    log_J = 0.0 # Symmetric proposal
    Npoints = 3
    def get_sample(self,old):
        subset = sample(self.ensemble,self.Npoints)
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
        a,b = sample(self.ensemble,2)
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
    def set_ensemble(self,ensemble):
        """
        Over-ride default set_ensemble so that the
        eigenvectors are recomputed when it is updated
        """
        super(EnsembleEigenVector,self).set_ensemble(ensemble)
        self.update_eigenvectors()

    def update_eigenvectors(self):
        """
        Recompute the eigenvectors of the covariance matrix
        from the ensemble
        """
        n=len(self.ensemble)
        dim = self.ensemble[0].dimension
        cov_array = np.zeros((dim,n))
        if dim == 1:
            name=self.ensemble[0].names[0]
            self.eigen_values = np.atleast_1d(np.var([self.ensemble[j][name] for j in range(n)]))
            self.eigen_vectors = np.eye(1)
        else:
            for i,name in enumerate(self.ensemble[0].names):
                for j in range(n): cov_array[i,j] = self.ensemble[j][name]
            covariance = np.cov(cov_array)
            self.eigen_values,self.eigen_vectors = np.linalg.eigh(covariance)

    def get_sample(self,old):
        """
        Propose a jump along a random eigenvector
        """
        out = old
        # pick a random eigenvector
        i = randrange(old.dimension)
        jumpsize = sqrt(fabs(self.eigen_values[i]))*gauss(0,1)
        for k,n in enumerate(out.names):
            out[n]+=jumpsize*self.eigen_vectors[k,i]
        return out


class DefaultProposalCycle(ProposalCycle):
    """
    A default proposal cycle that uses the Walk, Stretch, Differential Evolution
    and Eigenvector proposals
    """
    def __init__(self,*args,**kwargs):
        proposals = [EnsembleWalk(), EnsembleStretch(), DifferentialEvolution(), EnsembleEigenVector()]
        weights = [1.0,1.0,3.0,10.0]
        super(DefaultProposalCycle,self).__init__(proposals,weights,*args,**kwargs)

class NaiveProposal(Proposal):

    def __init__(self, names=None, log_prior=None, bounds=None, N=1000, **kwargs):
        super(NaiveProposal, self).__init__(**kwargs)

        self.dims = len(names)
        self.names = names
        self.log_prior = log_prior
        self.bounds = bounds
        self.prior_denom = np.ptp(bounds)
        self.populated = False
        self.N = N

    def make_live_point(self, theta):
        """Create an instance of LivePoint with the given inputs"""
        return LivePoint(self.names, d=theta)

    def log_proposal(self, theta):
        """Proposal probability"""
        return - np.log(self.prior_denom)

    def get_weights(self, theta):
        """Get weights for the samples"""
        log_q = np.array([self.log_proposal(t) for t in theta])
        log_p = np.array([self.log_prior(t) for t in theta])
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w

    def get_sample(self, old_sample):
        """Propose a new sample"""
        if not self.populated:
            theta = np.random.uniform(self.bounds[0], self.bounds[1], [self.N, self.dims])
            theta = [self.make_live_point(t) for t in theta]
            log_w = self.get_weights(theta)
            # rejection sampling
            log_u = np.log(np.random.rand(self.N))
            indices = np.where((log_w - log_u) >= 0)[0]
            self.samples = [theta[i] for i in indices]
            self.populated = True
        # get new sample
        new_sample = self.samples.pop()
        if not self.samples:
            self.populated = False
        return new_sample


class FlowProposal(EnsembleProposal):

    torch = __import__('torch')
    log_J = 0.0

    def __init__(self, model_dict=None, names=None, log_prior=None, device='cpu', prior_range=None, proposal_size=10000, fuzz=1.0, setup=None, normalise=True):

        super(FlowProposal, self).__init__()
        print(model_dict)
        #from .flowtrainer import FlowModel
        self.logger = logging.getLogger("CPNest")
        #self.model = FlowModel(**model_dict, device=device)                     # Flow model
        if setup is None:
            self.model = setup_model(**model_dict, device=device)
        else:
            self.model = setup(**model_dict, device=device)
        self.update_count = 0                  # times flows have been updated
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
        if prior_range is not None and normalise:
            self.setup_normalisation(prior_range)
        self.previous_sample = None

        self.proposal_size = proposal_size

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
        z, log_J = self.model(theta_tensor, mode='direct')
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
        self.update_count += 1

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
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        return log_w


class RandomFlowProposal(FlowProposal):

    def __init__(self, output='./', **kwargs):

        super(RandomFlowProposal, self).__init__(**kwargs)
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

    def load_weights(self, weights_file):
        """Update the model from a weights file"""
        self.model.load_state_dict(self.torch.load(weights_file))
        self.model.eval()
        self.populated = False
        self.update_count += 1

    def populate(self, old_r2, N=10000):
        """Populate a pool of latent points"""
        self.logger.debug("Populating proposal")
        # draw n samples and sort by radius
        # only proceed if the draw has produced points
        # mainly an issue with drawing from the gaussian
        self.samples = []
        self.z = np.empty([0, self.ndims])
        while len(self.samples) < N:
            while True:
                z = self.draw(old_r2, N, fuzz=self.fuzz)
                if z.size:
                    break

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
                # array of indices to take random draws from
                self.samples += [samples[i] for i in indices]
                self.z = np.concatenate([self.z, z[indices]], axis=0)

        self.samples = self.samples[:N]
        self.z = self.z[:N]
        self.indices = np.random.permutation(len(self.samples))
        self.populated = True
        self.logger.debug('Proposal populated: {} / {} points accepted'.format(len(self.samples), N))

    def get_sample(self, old_sample):
        """Get a new sample within the contour defined by the old sample"""
        count = 0
        if not self.populated:
            while not self.populated:
                # Populate if unpopulated
                # This requires pass through the flows
                old_sample_np = self.unpack_live_point(old_sample) # unpack LivePoint
                old_sample_np = self.rescale_input(old_sample_np)  # rescale (-1, 1)
                old_z, _ = self.forward_pass(old_sample_np.reshape(1, -1))        # to latent space
                old_r2 = self.radius2(old_z)                    # contour
                self.populate(old_r2, N=self.proposal_size)         # points within contour

                # get samples with index given by first entry in possible indices
                count += 1
                if count == 20:
                    self.logger.error('Could not populate proposal')
                    break

            self.save_samples(self.z, self.samples, self.indices)
            samples = np.array([s.values for s in self.samples])
            plot_samples(self.z, samples, output=self.output, filename='samples_{}.png'.format(self.count-1))

        # new sample is drawn randomly from proposed points
        new_sample = self.samples[self.indices[0]]
        self.indices = self.indices[1:]
        if not self.indices.size:
            self.populated = False
            self.logger.debug('Pool of points is empty')
        # make live point and return
        return new_sample
