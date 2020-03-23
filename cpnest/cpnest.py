#! /usr/bin/env python
# coding: utf-8

import multiprocessing as mp
from ctypes import c_double, c_int, c_char_p, c_char
import numpy as np
import os
import sys
import signal

from multiprocessing import Lock
from multiprocessing.sharedctypes import Value, Array
from multiprocessing.managers import SyncManager

import cProfile

from .logger import start_logger


class CheckPoint(Exception):
    print("Checkpoint exception raise")
    pass


def sighandler(signal, frame):
    print("Handling signal {}".format(signal))
    raise CheckPoint()

class CPNest(object):
    """
    Class to control CPNest sampler
    cp = CPNest(usermodel,nlive=100,output='./',verbose=0,seed=None,maxmcmc=100,nthreads=None,balanced_sampling = True)

    Input variables:
    =====

    usermodel: :obj:`cpnest.Model`
        a user-defined model to analyse

    nlive: `int`
        Number of live points (100)

    poolsize: `int`
        Number of objects in the sampler pool (100)

    output : `str`
        output directory (./)

    verbose: `int`
        Verbosity, 0=silent, 1=progress, 2=diagnostic, 3=detailed diagnostic

    seed: `int`
        random seed (default: 1234)

    maxmcmc: `int`
        maximum MCMC points for sampling chains (100)

    nthreads: `int` or `None`
        number of parallel samplers. Default (None) uses mp.cpu_count() to autodetermine

    nhamiltomnian: `int`
        number of sampler threads using an hamiltonian samplers. Default: 0

    resume: `boolean`
        determines whether cpnest will resume a run or run from scratch. Default: False.

    proposal: `dict`
        dictionary of lists with custom jump proposals.
        key 'mhs' for the Metropolis-Hastings sampler,
        'hmc' for the Hamiltonian Monte-Carlo sampler. Default: None

    n_periodic_checkpoint: `int`
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    """
    def __init__(self,
                 usermodel,
                 nlive        = 100,
                 poolsize     = 100,
                 output       = './',
                 verbose      = 0,
                 seed         = None,
                 maxmcmc      = 100,
                 nthreads     = None,
                 nhamiltonian = 0,
                 resume       = False,
                 proposals    = None,
                 trainer_class= None,
                 trainer_dict = None,
                 trainer_type = None,
                 n_periodic_checkpoint = None):
        if nthreads is None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads

        output = os.path.join(output, '')
        os.system("mkdir -p {0!s}".format(output))

        self.logger = start_logger(output, verbose=verbose)

        self.logger.critical('Running with {0} parallel threads'.format(self.nthreads))
        from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler, RejectionSampler
        from .NestedSampling import NestedSampler
        from .proposal import DefaultProposalCycle, HamiltonianProposalCycle
        if proposals is None:
            proposals = dict(mhs=DefaultProposalCycle,
                             hmc=HamiltonianProposalCycle)
        elif type(proposals) == list:
            proposals = dict(mhs=proposals[0],
                             hmc=proposals[1])
        self.nlive    = nlive
        self.verbose  = verbose
        self.output   = output
        self.poolsize = poolsize
        self.posterior_samples = None
        self.user     = usermodel
        self.resume = resume
        if trainer_class is not None:
            if None in [trainer_class, trainer_dict, trainer_type]:
                raise ValueError("If using a trainer must specifiy three arguments: trainer_class, trainer_dict, trainer_type")
            else:
                self.trainable = True
                self.logger.info("Trainer enabled")
        else:
            self.trainable = False
        if self.trainable:
            trainer_count = 1
            self.manager = RunManager(nthreads=self.nthreads-trainer_count, trainer=True)
            self.manager.start()
            SamplerClass = RejectionSampler
        else:
            trainer_count = 0
            self.manager = RunManager(nthreads=self.nthreads-trainer_count)
            self.manager.start()
            SamplerClass = MetropolisHastingsSampler

        if seed is None: self.seed=1234
        else:
            self.seed=seed

        self.process_pool = []

        if nthreads == 1 and self.trainable:
            raise RuntimeError("Using a trainer requieres more than one thread")

        # instantiate the nested sampler class
        resume_file = os.path.join(output, "nested_sampler_resume.pkl")
        if not os.path.exists(resume_file) or resume == False:
            self.NS = NestedSampler(self.user,
                        nlive          = nlive,
                        output         = output,
                        verbose        = verbose,
                        seed           = self.seed,
                        prior_sampling = False,
                        manager        = self.manager,
                        trainer = self.trainable,
                        trainer_type = trainer_type,
                        trainer_dict = trainer_dict,
                        n_periodic_checkpoint = n_periodic_checkpoint)
        else:
            self.NS = NestedSampler.resume(resume_file, self.manager, self.user)

        # instaniate the neural network if used
        if self.trainable:
            if not os.path.exists(resume_file) or resume == False:
                trainer = trainer_class(trainer_dict,
                        manager=self.manager,
                        output=output)
            else:
                raise NotImplementedError("Can't restore neural network from resume file")

            self.trainer_process = mp.Process(target=trainer.receiver)
        # instantiate the sampler class
        for i in range(self.nthreads-nhamiltonian-trainer_count):
            resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
            if not os.path.exists(resume_file) or resume == False:
                sampler = SamplerClass(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = proposals['mhs'](),
                                  resume_file = resume_file,
                                  manager     = self.manager,
                                  trainer_type = trainer_type,
                                  trainer_dict = trainer_dict
                                  )
            else:
                sampler = SamplerClass.resume(resume_file,
                                                           self.manager,
                                                           self.user)

            p = mp.Process(target=sampler.receiver)
            self.process_pool.append(p)

        for i in range(self.nthreads-nhamiltonian-trainer_count,self.nthreads-trainer_count):
            resume_file = os.path.join(output, "sampler_{0:d}.pkl".format(i))
            if not os.path.exists(resume_file) or resume == False:
                sampler = HamiltonianMonteCarloSampler(self.user,
                                  maxmcmc,
                                  verbose     = verbose,
                                  output      = output,
                                  poolsize    = poolsize,
                                  seed        = self.seed+i,
                                  proposal    = proposals['hmc'](model=self.user),
                                  resume_file = resume_file,
                                  manager     = self.manager,
                                  trainer_type = trainer_type,
                                  trainer_dict = trainer_dict
                                  )
            else:
                sampler = HamiltonianMonteCarloSampler.resume(resume_file,
                                                              self.manager,
                                                              self.user)
            p = mp.Process(target=sampler.receiver)
            self.process_pool.append(p)


    def run(self):
        """
        Run the sampler
        """
        if self.resume:
            signal.signal(signal.SIGTERM, sighandler)
            signal.signal(signal.SIGALRM, sighandler)
            signal.signal(signal.SIGQUIT, sighandler)
            signal.signal(signal.SIGINT, sighandler)
            signal.signal(signal.SIGUSR1, sighandler)
            signal.signal(signal.SIGUSR2, sighandler)

        #self.p_ns.start()
        if self.trainable:
            self.trainer_process.start()
        for each in self.process_pool:
            each.start()
        try:
            self.NS.nested_sampling_loop()
            if self.trainable:
                self.logger.warning("Waiting for training to end")
                self.manager.stop_training.value = 1
                self.trainer_process.join()
            for each in self.process_pool:
                each.join()
        except CheckPoint:
            self.checkpoint()
            sys.exit(130)

        self.posterior_samples = self.get_posterior_samples(filename=None)
        if self.verbose>1: self.plot()

        #TODO: Clean up the resume pickles

    def get_nested_samples(self, filename='nested_samples.dat'):
        """
        returns nested sampling chain
        Parameters
        ----------
        filename : string
                   If given, file to save nested samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays(
                [s.asnparray()
                    for s in self.NS.nested_samples]
                ,usemask=False)
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder,'nested_samples.dat'),
                self.nested_samples.ravel(),
                header=' '.join(self.nested_samples.dtype.names),
                newline='\n',delimiter=' ')
        return self.nested_samples

    def get_posterior_samples(self, filename='posterior.dat'):
        """
        Returns posterior samples

        Parameters
        ----------
        filename : string
                   If given, file to save posterior samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy as np
        import os
        from .nest2pos import draw_posterior_many
        from . import plot
        nested_samples = self.get_nested_samples()
        posterior_samples = draw_posterior_many([nested_samples],[self.nlive],verbose=self.verbose)
        posterior_samples = np.array(posterior_samples)
        # TODO: Replace with something to output samples in whatever format
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder,'posterior.dat'),
                self.posterior_samples.ravel(),
                header=' '.join(posterior_samples.dtype.names),
                newline='\n',delimiter=' ')
        return posterior_samples

    def plot(self, corner = True):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        pos = self.posterior_samples

        from . import plot
        for n in pos.dtype.names:
            plot.plot_hist(pos[n].ravel(),name=n,filename=os.path.join(self.output,'posterior_{0}.png'.format(n)))
        for n in self.nested_samples.dtype.names:
            plot.plot_chain(self.nested_samples[n],name=n,filename=os.path.join(self.output,'nschain_{0}.png'.format(n)))
        plot.plot_corner_samples(self.nested_samples, filename=os.path.join(self.output, 'nested_samples.png'))
        plot.plot_corner_samples(pos, filename=os.path.join(self.output, 'posterior_samples.png'), cmap=False)
        import numpy as np
        plotting_posteriors = np.squeeze(pos.view((pos.dtype[0], len(pos.dtype.names))))
        if corner: plot.plot_corner(plotting_posteriors,labels=pos.dtype.names,filename=os.path.join(self.output,'corner.png'))

        plot.plot_proposal_stats(self.output)


    def worker_sampler(self, producer_pipe, logLmin):
        cProfile.runctx('self.sampler.produce_sample(producer_pipe, logLmin)', globals(), locals(), 'prof_sampler.prof')

    def worker_ns(self):
        cProfile.runctx('self.NS.nested_sampling_loop(self.consumer_pipes)', globals(), locals(), 'prof_nested_sampling.prof')

    def profile(self):
        for i in range(0,self.NUMBER_OF_PRODUCER_PROCESSES):
            p = mp.Process(target=self.worker_sampler, args=(self.queues[i%len(self.queues)], self.NS.logLmin ))
            self.process_pool.append(p)
        for i in range(0,self.NUMBER_OF_CONSUMER_PROCESSES):
            p = mp.Process(target=self.worker_ns, args=(self.queues, self.port, self.authkey))
            self.process_pool.append(p)
        for each in self.process_pool:
            each.start()

    def checkpoint(self):
        self.manager.checkpoint_flag=1


class RunManager(SyncManager):
    def __init__(self, nthreads=None, trainer=False, **kwargs):
        super(RunManager,self).__init__(**kwargs)
        self.nconnected=mp.Value(c_int,0)
        self.producer_pipes = list()
        self.consumer_pipes = list()
        for i in range(nthreads):
            consumer, producer = mp.Pipe(duplex=True)
            self.producer_pipes.append(producer)
            self.consumer_pipes.append(consumer)
        self.logLmin=None
        self.logLmax = None
        self.nthreads=nthreads

        if trainer:
            self.trainer = True   # enables trainer fucntions
            self.trainer_consumer_pipe, self.trainer_producer_pipe = mp.Pipe(duplex=True)
        else:
            self.trainer = False

    def start(self):
        super(RunManager, self).start()
        self.logLmin = mp.Value(c_double,-np.inf)
        self.logLmax = mp.Value(c_double,-np.inf)
        self.checkpoint_flag=mp.Value(c_int,0)

        if self.trainer:
            self.trained = mp.Value(c_int,0)
            self.training = mp.Value(c_int,0)
            self.enable_flow = mp.Value(c_int, 0)
            self.stop_training = mp.Value(c_int, 0)
            self.send_weights = mp.Value(c_int, 0)
            self.weights_file = mp.Array(c_char, range(128))

    def connect_producer(self, trainer=False):
        """
        Returns the producer's end of the pipe
        """
        if trainer:
            return self.trainer_producer_pipe
        else:
            with self.nconnected.get_lock():
                n = self.nconnected.value
                pipe = self.producer_pipes[n]
                self.nconnected.value+=1
            return pipe, n
