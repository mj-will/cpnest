import unittest
import numpy as np
from scipy import integrate,stats
import cpnest.model
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

class GaussianModel(cpnest.model.Model):
    """
    A simple gaussian model with parameters mean and sigma
    """
    def __init__(self):
        self.distr = stats.norm(loc=0,scale=1.0)
    names=['x']
    bounds=[[-10,10]]
    analytic_log_Z=0.0 - np.log(bounds[0][1] - bounds[0][0])

    def log_likelihood(self,p):
        return self.distr.logpdf(p['x'])
        #return -0.5*(p['x']**2) - 0.5*np.log(2.0*np.pi)


class GaussianTestCase(unittest.TestCase):
    """
    Test the gaussian model
    """
    def setUp(self):
        self.model=GaussianModel()
        self.work=cpnest.CPNest(self.model,verbose=2,Nthreads=1,Nlive=500,maxmcmc=5000)
        self.work.run()

    def test_evidence(self):
        # 2 sigma tolerance
        tolerance = 2.0*np.sqrt(self.work.NS.state.info/self.work.NS.Nlive)
        print('Tolerance: {0:0.3f}'.format(tolerance))
        self.assertTrue(np.abs(self.work.NS.logZ - GaussianModel.analytic_log_Z)<tolerance, 'Incorrect evidence for normalised distribution: {0:.3f} instead of {1:.3f}'.format(self.work.NS.logZ,GaussianModel.analytic_log_Z ))
        
    def test_posterior_ks(self):
        pos=self.work.posterior_samples['x']
        t,pval=stats.kstest(pos,self.model.distr.cdf)
        plt.figure()
        plt.hist(pos.ravel(),normed=True)
        plt.title('KS test = %f'%(pval))
        plt.savefig('posterior.png')
        self.assertTrue(pval>0.01,'KS test failed: KS stat = {0}'.format(pval))



def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
    unittest.main(verbosity=2)
 
