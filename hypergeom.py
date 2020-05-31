import numpy as np
import theano.tensor as tt
from scipy import stats
import warnings

from pymc3.util import get_variable_name
from pymc3.math import tround, sigmoid, logaddexp, logit, log1pexp

from pymc3 import Discrete, floatX, intX
from pymc3.distributions.bound import bound
from pymc3.distributions.discrete import binomln, betaln
from pymc3.distributions import draw_values, generate_samples


class HyperGeometric(Discrete):
    R"""
    HyperGeometric log-likelihood.

    The probability of x successes in a sequence of n Bernoulli
    trials (That is, sample size = n) - where the population 
    size is N, containing a total of k successful individuals.
    The process is carried out without replacement.
    

    The pmf of this distribution is

    .. math:: f(x \mid N, n, k) = \frac{\binom{k}{x} \binom{N-k}{N-x}}{\binom{N}{n}} 

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(1, 15)
        N = 50
        k = 10
        for n in [20, 25]:
            pmf = st.hypergeom.pmf(x, N, k, n)
            plt.plot(x, pmf, '-o', label='n = {}'.format(n))
        plt.plot(x, pmf, '-o', label='N = {}'.format(N))
        plt.plot(x, pmf, '-o', label='k = {}'.format(k))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{>0}`
    Mean      :math:`\dfrac{n.k}{N}`
    Variance  :math:`\dfrac{(N-n).n.k.(N-k)}{(N-1).N^2}`
    ========  =============================

    Parameters
    ----------
    N : integer
        Total size of the population
    n : integer
        Number of samples drawn from the population
    k : integer
        Number of successful individuals in the population
    """

    def __init__(self, N,  k, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k = tt.as_tensor_variable(intX(k))
        self.N = N = tt.as_tensor_variable(intX(N))
        self.n = n = tt.as_tensor_variable(intX(n))
        self.mode = 1

    def random(self, point=None, size=None):
        """
        Draw random values from HyperGeometric distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        N, n, k = draw_values([self.N, self.n, self.k], point=point, size=size)
        return generate_samples(np.random.hypergeometric, N, n, k,
                                dist_shape=self.shape,
                                size=size)
    def logp(self, value):
        """
        Calculate log-probability of HyperGeometric distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        k = self.k
        N = self.N
        n = self.n
        # return bound(binomln(k, value) + binomln(N - k, value) - binomln(N, n),
        #              0 <= k, k <= N, 0 <= n, 0 <= N, max(0, n - N + k) <= value, value <= min(k, n))
        return bound(binomln(k, value) + binomln(N - k, n - value) - binomln(N, n),
                     tt.le(0, k), tt.le(k, N), tt.le(0, n), 
                     tt.le(0, N), tt.le(tt.maximum(0, n - N + k), value), 
                     tt.le(value, tt.minimum(k, n)))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        N = dist.N
        n = dist.n
        k = dist.k
        name = r'\text{%s}' % name
        return r'${} \sim \text{{HyperGeometric}}(\mathit{{N}}={},~\mathit{{n}}={}, ,~\mathit{{k}}={})$'.format(name,
                                                get_variable_name(N),
                                                get_variable_name(n),
                                                get_variable_name(k))

