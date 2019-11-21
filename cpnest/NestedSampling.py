from __future__ import division, print_function
import sys
import os
import pickle
import time
import numpy as np
import multiprocessing as mp
from numpy import logaddexp, exp
from numpy import inf
from math import isnan
try:
    from queue import Empty
except ImportError:
    from Queue import Empty # For python 2 compatibility
import types
from . import nest2pos
from .nest2pos import logsubexp
from operator import attrgetter
from .cpnest import CheckPoint, RunManager
from .cpthread import CPCommand

from tqdm import tqdm


class _NSintegralState(object):
  """
  Stores the state of the nested sampling integrator
  """
  def __init__(self, nlive):
    self.nlive = nlive
    self.reset()
  def reset(self):
    """
    Reset the sampler to its initial state at logZ = -infinity
    """
    self.iteration=0
    self.logZ=-inf
    self.oldZ=-inf
    self.logw=0
    self.info=0
    # Start with a dummy sample enclosing the whole prior
    self.logLs=[-inf] # Likelihoods sampled
    self.log_vols=[0.0] # Volumes enclosed by contours
  def increment(self, logL, nlive=None):
    """
    Increment the state of the evidence integrator
    Simply uses rectangle rule for initial estimate
    """
    if(logL<=self.logLs[-1]):
      print('WARNING: NS integrator received non-monotonic logL. {0:.5f} -> {1:.5f}'.format(self.logLs[-1],logL))
    if nlive is None:
      nlive = self.nlive
    oldZ = self.logZ
    logt=-1.0/nlive
    Wt = self.logw + logL + logsubexp(0,logt)
    self.logZ = logaddexp(self.logZ,Wt)
    # Update information estimate
    if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
        self.info = np.exp(Wt - self.logZ)*logL + np.exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
        if isnan(self.info):
            self.info=0

    # Update history
    self.logw += logt
    self.iteration += 1
    self.logLs.append(logL)
    self.log_vols.append(self.logw)
  def finalise(self):
    """
    Compute the final evidence with more accurate integrator
    Call at end of sampling run to refine estimate
    """
    from scipy import integrate
    # Trapezoidal rule
    self.logZ=nest2pos.log_integrate_log_trap(np.array(self.logLs),np.array(self.log_vols))
    return self.logZ
  def plot(self,filename):
    """
    Plot the logX vs logL
    """
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    fig=plt.figure()
    plt.plot(self.log_vols,self.logLs)
    plt.title('{0} iterations. logZ={1:.2f} H={2:.2f} bits'.format(self.iteration,self.logZ,self.info*np.log2(np.e)))
    plt.grid(which='both')
    plt.xlabel('log prior_volume')
    plt.ylabel('log likelihood')
    plt.xlim([self.log_vols[-1],self.log_vols[0]])
    plt.savefig(filename)
    print('Saved nested sampling plot as {0}'.format(filename))


class NestedSampler(object):
    """
    Nested Sampler class.
    Initialisation arguments:

    model: :obj:`cpnest.Model` user defined model

    manager: `multiprocessing` manager instance which controls
        the shared objects.
        Default: None

    Nlive: int
        number of live points to be used for the integration
        Default: 1024

    output: string
        folder where the output will be stored
        Default: None

    verbose: int
        0: Nothing
        1: display information on screen
        2: (1) + diagnostic plots
        Default: 1

    seed: int
        seed for the initialisation of the pseudorandom chain
        Default: 1234

    prior_sampling: boolean
        produce Nlive samples from the prior.
        Default: False

    stopping: float
        Stop when remaining samples wouldn't change logZ estimate by this much.
        Deafult: 0.1

    n_periodic_checkpoint: int
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    """

    def __init__(self,
                 model,
                 manager        = None,
                 nlive          = 1024,
                 output         = None,
                 verbose        = 1,
                 seed           = 1,
                 prior_sampling = False,
                 stopping       = 0.1,
                 trainer = False,
                 trainer_type   = None,
                 trainer_dict   = None,
                 n_periodic_checkpoint = None):
        """
        Initialise all necessary arguments and
        variables for the algorithm
        """
        self.model          = model
        self.manager        = manager
        self.prior_sampling = prior_sampling
        self.setup_random_seed(seed)
        self.verbose        = verbose
        self.acceptance     = 1.0
        self.accepted       = 0
        self.rejected       = 1
        self.queue_counter  = 0
        self.Nlive          = nlive
        self.params         = [None] * self.Nlive
        self.n_periodic_checkpoint = n_periodic_checkpoint
        self.tolerance      = stopping
        self.condition      = np.inf
        self.worst          = 0
        self.logLmax        = -np.inf
        self.logLmin        = self.manager.logLmin
        self.iteration      = 0
        self.nested_samples = []
        self.logZ           = None
        self.state          = _NSintegralState(self.Nlive)
        sys.stdout.flush()
        self.output_folder  = output
        self.output_file,self.evidence_file,self.resume_file = self.setup_output(output)
        header              = open(os.path.join(output,'header.txt'),'w')
        header.write('\t'.join(self.model.names))
        header.write('\tlogL\n')
        header.close()
        self.trainer = trainer
        self.trainer_type = trainer_type
        if trainer:
            self.n_training_samples = trainer_dict['n_training_samples']
            if trainer_type == 'function_approximator':
                self.n_likelihood_evaluations = trainer_dict['n_likelihood_evaluations']
            print("CPnest Trainer: Training enabled in nested sampling")
        self.initialised    = False

    def setup_output(self,output):
        """
        Set up the output folder

        -----------
        Parameters:
        output: string
            folder where the results will be stored
        -----------
        Returns:
            output_file, evidence_file, resume_file: tuple
                output_file:   file where the nested samples will be written
                evidence_file: file where the evidence will be written
                resume_file:   file used for checkpointing the algorithm
        """
        os.system("mkdir -p {0!s}".format(output))
        chain_filename = "chain_"+str(self.Nlive)+"_"+str(self.seed)+".txt"
        output_file   = os.path.join(output,chain_filename)
        evidence_file = os.path.join(output,chain_filename+"_evidence.txt")
        resume_file  = os.path.join(output,"nested_sampler_resume.pkl")

        return output_file, evidence_file, resume_file


    def write_chain_to_file(self):
        """
        Outputs a `cpnest.parameter.LivePoint` to the
        output_file
        """
        with open(self.output_file,"w") as f:
            f.write('{0:s}\n'.format(self.model.header().rstrip()))
            for ns in self.nested_samples:
                f.write('{0:s}\n'.format(self.model.strsample(ns).rstrip()))

    def write_evidence_to_file(self):
        """
        Write the evidence logZ and maximum likelihood to the evidence_file
        """
        with open(self.evidence_file,"w") as f:
            f.write('{0:.5f} {1:.5f}\n'.format(self.state.logZ, self.logLmax))

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)

    def consume_sample(self):
        """
        consumes a sample from the consumer_pipes
        and updates the evidence logZ
        """
        # Increment the state of the evidence integration
        logLmin = self.get_worst_n_live_points(len(self.manager.consumer_pipes))
        logLtmp = []
        for k in self.worst:
            self.state.increment(self.params[k].logL)
            self.manager.consumer_pipes[k].send(CPCommand('sample', self.params[k]))
            self.nested_samples.append(self.params[k])
            logLtmp.append(self.params[k].logL)
        self.condition = logaddexp(self.state.logZ,self.logLmax - self.iteration/(float(self.Nlive))) - self.state.logZ

        # Replace the points we just consumed with the next acceptable ones
        # Make sure we are mixing the chains
        np.random.shuffle(self.worst)
        for k in self.worst:
            self.iteration += 1
            loops           = 0
            while(True):
                loops += 1
                acceptance, sub_acceptance, self.jumps, proposed = self.manager.consumer_pipes[self.queue_counter].recv()
                if proposed.logL > self.logLmin.value:
                    # replace worst point with new one
                    self.params[k]     = proposed
                    self.queue_counter = (self.queue_counter + 1) % len(self.manager.consumer_pipes)
                    self.accepted += 1
                    break
                else:
                    # resend it to the producer
                    self.manager.consumer_pipes[self.queue_counter].send(CPCommand('sample', self.params[k]))
                    self.rejected += 1
            self.acceptance = float(self.accepted)/float(self.accepted + self.rejected)
            if self.verbose:
                sys.stderr.write("{0:d}: n:{1:4d} NS_acc:{2:.3f} S{3:d}_acc:{4:.3f} sub_acc:{5:.3f} H: {6:.2f} logL {7:.5f} --> {8:.5f} dZ: {9:.3f} logZ: {10:.3f} logLmax: {11:.2f}\n"\
                .format(self.iteration, self.jumps*loops, self.acceptance, k, acceptance, sub_acceptance, self.state.info,\
                  logLtmp[k], self.params[k].logL, self.condition, self.state.logZ, self.logLmax))
                sys.stderr.flush()

    def get_worst_n_live_points(self, n):
        """
        selects the lowest likelihood N live points
        for evolution
        """
        self.params.sort(key=attrgetter('logL'))
        self.worst = np.arange(n)
        self.logLmin.value = np.float128(self.params[n-1].logL)
        self.logLmax = self.params[-1].logL
        return np.float128(self.logLmin.value)

    def reset(self):
        """
        Initialise the pool of `cpnest.parameter.LivePoint` by
        sampling them from the `cpnest.model.log_prior` distribution
        """
        # send all live points to the samplers for start
        i = 0
        nthreads=self.manager.nthreads
        with tqdm(total=self.Nlive, disable= not self.verbose, desc='CPNEST: populate samplers', position=nthreads) as pbar:
            while i < self.Nlive:
                for j in range(nthreads): self.manager.consumer_pipes[j].send(CPCommand(ctype='sample', payload=self.model.new_point()))
                for j in range(nthreads):
                    while i < self.Nlive:
                        acceptance,sub_acceptance,self.jumps,self.params[i] = self.manager.consumer_pipes[self.queue_counter].recv()
                        self.queue_counter = (self.queue_counter + 1) % len(self.manager.consumer_pipes)
                        if self.params[i].logP!=-np.inf and self.params[i].logL!=-np.inf:
                            i+=1
                            pbar.update()
                            break
        if self.verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()

        if False:
            print("Trainer network: Pre-Training on intial live poiints")
            self.manager.trainer_consumer_pipe.send(CPCommand('train', payload=self.params))
            self.manager.training.value = 1
        self.initialised=True

    def nested_sampling_loop(self):
        """
        main nested sampling loop
        """
        if not self.initialised:
            self.reset()
        if self.prior_sampling:
            for i in range(self.Nlive):
                self.nested_samples.append(self.params[i])
            self.write_chain_to_file()
            self.write_evidence_to_file()
            self.logLmin.value = np.inf
            for c in self.manager.consumer_pipes:
                c.send(CPCommand('exit'))
            print("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        try:
            i=0
            training_data = list()  # Data that has yet to be trained on
            retrain = False         # Retrain flag
            using_fa = False        # using fa instead of analytic likelihood
            fa_count = 0            # count of points computed with approximator likelihood
            nthreads = self.manager.nthreads
            while self.condition > self.tolerance:
                if self.trainer:
                    #print("CPnest trainer: training statement")
                    if self.manager.trained.value:
                        print("Trainer: training complete")
                        if self.trainer_type == 'function_approximator':
                            weights_file = self.manager.trainer_consumer_pipe.recv()
                            for c in self.manager.consumer_pipes:
                                c.send(CPCommand('set_weights', payload=weights_file))
                            self.manager.trained.value = 0
                            self.manager.training.value = 0
                            if self.manager.use_fa.value:
                                print("Function approximator: switching to approximate likelihood")
                                for c in self.manager.consumer_pipes:
                                    c.send(CPCommand('switch', payload='fa'))
                                print("Function approxiamtor: using approximate likelihood for {0} iterations".format(self.n_likelihood_evaluations))
                                using_fa = True
                                self.manager.use_fa.value = 0
                            else:
                                print("Function approximator: peformance not high enough, retraining")
                                retrain = True
                        elif self.trainer_type == 'flow':
                            print("Trainer: flows does not affect sampler")
                            self.manager.trained.value = 0
                            self.manager.training.value = 0
                        else:
                            raise NotImplementedError("Trainer not implemented, choose from function approximator or flow")

                    if (len(self.nested_samples) % self.n_training_samples <= nthreads) or (retrain and len(self.nested_samples) % (self.n_training_samples / 2) <= nthreads):
                        retrain = False
                        if not len(self.nested_samples) > nthreads:
                            pass
                        elif not self.manager.training.value:
                            print("Trainer: training")
                            if len(training_data):
                                training_data += self.params
                                if self.trainer_type == 'function_approximator':
                                    training_data += self.nested_samples[self.Nlive:]     # include nested samples exclusing initial batch
                                self.manager.trainer_consumer_pipe.send(CPCommand('train', payload=training_data))
                                training_data = list()
                            else:
                                if self.trainer_type == 'function_approximator':
                                    self.manager.trainer_consumer_pipe.send(CPCommand('train', payload=self.params + self.nested_samples[self.Nlive:]))
                                else:
                                    self.manager.trainer_consumer_pipe.send(CPCommand('train', payload=self.params))
                            self.manager.training.value = 1
                        else:
                            training_data += self.params

                self.consume_sample()
                if using_fa:
                    fa_count += nthreads
                    if fa_count >= self.n_likelihood_evaluations:
                        print("Function approximator: switching to analytic likelihood")
                        for c in self.manager.consumer_pipes:
                            c.send(CPCommand('switch', payload='model'))
                        using_fa = False
                        fa_count = 0

                if self.n_periodic_checkpoint is not None and i % self.n_periodic_checkpoint == 1:
                    self.checkpoint()

                i += 1

        except CheckPoint:
            self.checkpoint()
            # Run each pipe to get it to checkpoint
            for c in self.manager.consumer_pipes:
                c.send(CPCommand('checkpoint'))
            sys.exit(130)

        # Signal worker threads to exit
        self.logLmin.value = np.inf
        for c in self.manager.consumer_pipes:
            c.send(CPCommand('exit'))
        # signal nn worker thread to exit
        if self.trainer:
            print("Trainer: Waiting for training to end")
            self.manager.trainer_consumer_pipe.send(CPCommand('exit'))

        # final adjustments
        self.params.sort(key=attrgetter('logL'))
        for i,p in enumerate(self.params):
            self.state.increment(p.logL,nlive=self.Nlive-i)
            self.nested_samples.append(p)

        # Refine evidence estimate
        self.state.finalise()
        self.logZ = self.state.logZ
        # output the chain and evidence
        self.write_chain_to_file()
        self.write_evidence_to_file()
        print('Final evidence: {0:0.2f}\nInformation: {1:.2f}'.format(self.state.logZ,self.state.info))

        # Some diagnostics
        if self.verbose>1 :
          self.state.plot(os.path.join(self.output_folder,'logXlogL.png'))
        return self.state.logZ, self.nested_samples

    def checkpoint(self):
        """
        Checkpoint its internal state
        """
        print('Checkpointing nested sampling')
        with open(self.resume_file,"wb") as f:
            pickle.dump(self, f)

    @classmethod
    def resume(cls, filename, manager, usermodel):
        """
        Resumes the interrupted state from a
        checkpoint pickle file.
        """
        print('Resuming NestedSampler from '+filename)
        with open(filename,"rb") as f:
            obj = pickle.load(f)
        obj.manager = manager
        obj.logLmin = obj.manager.logLmin
        obj.logLmin.value = obj.llmin
        obj.model = usermodel
        del obj.__dict__['llmin']
        return(obj)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['llmin']=self.logLmin.value
        # Remove the unpicklable entries.
        del state['logLmin']
        del state['manager']
        del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
