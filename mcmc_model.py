import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os

import arviz as az
import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

from datetime import date
from datetime import datetime

from hypergeom import HyperGeometric


class MCMCModel(object):
    def __init__(self, region, num_positive, num_tests, I_0=25, window=50, verbose=1):
        
        # Just for identification purposes
        self.region = region
        
        # For the model, we'll only look at the last N
        self.num_positive = num_positive.iloc[-window:]
        self.num_tests = num_tests.iloc[-window:]
        self.I_0 = I_0 if window == -1 else self.num_positive.iloc[0]
        self.verbose = verbose

        # Where we store the results
        self.trace = None
        self.trace_index = self.num_tests.index

    def run(self, chains=1, tune=10_000, draws=1_000, target_accept=.95):

        with pm.Model() as model:

            # Random walk magnitude
            step_size = pm.HalfNormal('step_size', sigma=.03)

            # Theta random walk
            theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
            theta_raw_steps = pm.Normal('theta_raw_steps', shape=len(self.num_positive)-2) * step_size
            theta_raw = tt.concatenate([[0., theta_raw_init], theta_raw_steps])
            theta = pm.Deterministic('theta', theta_raw.cumsum())
            theta_cumulative = pm.Deterministic('theta_c', theta.cumsum())

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta/gamma + 1)

            # Up until here is fine.
            N_t = 100_000 * np.ones_like(self.num_positive) # Some large numbers, think of them as candidates
            
            I_t_mu = self.I_0 * pm.math.exp(theta_cumulative)
            # Ensure cases stay above zero for poisson
            I_t = pm.math.maximum(.1, I_t_mu)
            cases = pm.Poisson('cases', mu=I_t, shape=self.num_positive.shape)

            # Random walk magnitude
            step_size = pm.HalfNormal('step_size', sigma=.03)

            # Theta random walk
            theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
            theta_raw_steps = pm.Normal('theta_raw_steps', shape=len(self.num_positive)-2) * step_size
            theta_raw = tt.concatenate([[0., theta_raw_init], theta_raw_steps])
            theta = pm.Deterministic('theta', theta_raw.cumsum())
            theta_cumulative = pm.Deterministic('theta_c', theta.cumsum())

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta/gamma + 1)

            # Up until here is fine.
            N_t = 100_000 * np.ones_like(self.num_positive) # Some large numbers, think of them as candidates
            
            I_t_mu = self.I_0 * pm.math.exp(theta_cumulative)
            # Ensure cases stay above zero for poisson
            I_t = pm.math.maximum(.1, I_t_mu)
            cases = pm.Poisson('cases', mu=I_t, shape=self.num_positive.shape)

            
            observed = self.num_positive
            # positives = HyperGeometric('positives', 
            #                            N = N_t, n = self.num_tests, k = cases, )
                                    #    observed=self.num_positive,
                                    #    testval=self.num_positive, 
                                    #    shape=self.num_positive.shape)
            
            # Approximation taken from
            # https://www.vosesoftware.com/riskwiki/ApproximationstotheHypergeometricdistribution.php
            approx_mu = pm.Deterministic('approx_mu', cases * (self.num_tests / N_t))
            approx_positives = pm.Poisson('approx_positives', 
                                          mu=approx_mu, 
                                          shape=self.num_positive.shape,
                                          observed=self.num_positive)

            assert np.all(self.num_positive < self.num_tests)

            # prob_success = pm.Deterministic('success_rate', cases/N_t)
            # positives = pm.distributions.discrete.Binomial(
            #     'positives', n = self.num_tests, p = prob_success,
            #     testval=self.num_positive, shape=self.num_positive.shape
            # )

            # Extra noise for smoothing out the discreteness
            # diff = pm.Normal('error', approx_positives - self.num_positive, sd=5., 
            #                  observed=np.zeros_like(self.num_positive))

            if self.verbose:
                print('Built model, sampling...')

            for RV in model.basic_RVs:
                print(RV.name, RV.logp(model.test_point))

            self.trace = pm.sample(
                chains=chains,
                tune=tune,
                draws=draws,
                # target_accept=target_accept
                nuts={"target_accept": target_accept},
                init='advi+adapt_diag'
            )
            
            return self
    
    def run_gp(self):
        with pm.Model() as model:
            gp_shape = len(self.onset) - 1

            length_scale = pm.Gamma("length_scale", alpha=3, beta=.4)

            eta = .05
            cov_func = eta**2 * pm.gp.cov.ExpQuad(1, length_scale)

            gp = pm.gp.Latent(mean_func=pm.gp.mean.Constant(c=0), 
                              cov_func=cov_func)

            # Place a GP prior over the function f.
            theta = gp.prior("theta", X=np.arange(gp_shape)[:, None])

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta / gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            self.trace = pm.sample(chains=1, tune=1000, draws=1000, target_accept=.8)
        return self


def df_from_model(model):
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    hpd_90 = az.hdi(r_t, .9)
    hpd_50 = az.hdi(r_t, .5)

    approx_positives = model.trace['approx_positives']
    median_approx_pos = np.median(approx_positives, axis=0)
    hpd_90_approx_pos = az.hdi(approx_positives, .9)

    cases = model.trace['cases']
    median_cases = np.median(cases, axis=0)
    hpd_90_cases = az.hdi(cases, .9)

    idx = pd.MultiIndex.from_product([
            [model.region],
            model.trace_index
        ], names=['region', 'date'])
        
    df = pd.DataFrame(data=np.c_[mean, median, hpd_90, hpd_50, 
                                 median_approx_pos, hpd_90_approx_pos,
                                 median_cases, hpd_90_cases], index=idx,
                 columns=['mean', 'median', 'lower_90', 'upper_90', 'lower_50','upper_50',
                          'median_approx_pos', 'lower_90_approx_pos', 'upper_90_approx_pos',
                          'median_cases', 'lower_90_cases', 'upper_90_cases'])
    return df


def create_and_run_model(args, verbose=True):
    data = pd.read_csv(args.infile)
    data = data[data.P_t >= args.cutoff]
    if verbose:
        print(f'Data loaded, total datapoints {len(data)}')
    return MCMCModel(args.infile, data.P_t, data.T_t, window=args.window).run()


def parse_args():
    parser = argparse.ArgumentParser(description='Rt estimation with MCMC')

    parser.add_argument(
        '--infile',
        type=str,
        default='synthetic_data/bd.csv',
        help='File from which to read P_t and T_t (default: %(default)s)',
        nargs='?',
    )

    parser.add_argument(
        '--cutoff', type=int, 
        help='Minimum number of positive tests from which to start inference (default: %(default)d)', 
        nargs='?', default=25
    )

    parser.add_argument(
        '--window', type=int, 
        help='Number of days to compute R_t for (default: %(default)d)', 
        nargs='?', default=-1
    )

    parser.add_argument(
        '--outfile',
        type=str,
        help='Output file to save data to. \nFormat should be <country>_<last date of observation>.csv (default: %(default)s)',
        nargs='?',
        default='rt/synthetic/bd.csv',
    )

    return parser.parse_args()

def main():
    args = parse_args()
    model = create_and_run_model(args)
    results = df_from_model(model)
    results.to_csv(args.outfile)

if __name__ == '__main__':
    main()