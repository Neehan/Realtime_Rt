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

from scipy.stats import kurtosis, skew

from hypergeom import HyperGeometric


class MCMCModel(object):
    def __init__(self, region, num_positive, num_tests,
                 last_tests, last_positives,
                 R_log_I_mu, R_log_I_cov, 
                 N_t, N_t_1, fixed_N_t = 100_000, use_real_nt=False,
                 R_t_drift = 0.05, verbose=1, extra_evidence=True):

        # Just for identification purposes
        self.region = region

        # For the model, we'll only look at the last N
        self.num_positive = num_positive
        self.num_tests = num_tests
        self.last_tests, self.last_positives = last_tests, last_positives
        self.R_log_dI_mu = R_log_I_mu
        self.R_log_dI_cov = R_log_I_cov
        self.R_t_drift = R_t_drift

        self.fixed_N_t = fixed_N_t
        self.N_t = N_t if use_real_nt else -1
        self.N_t_1 = N_t_1 if use_real_nt else -1   
        self.extra_evidence = extra_evidence     

        self.verbose = verbose

        # Where we store the results
        self.trace = None

    def run(self, chains=1, tune=3_000, draws=500, target_accept=.9, cores=1):

        with pm.Model() as model:
            # Figure out the new R_t, log I_t
            R_log_I_t = pm.MvNormal('R_log_I_t', 
                                    mu=self.R_log_dI_mu, 
                                    cov=self.R_log_dI_cov,
                                    shape=2, testval=self.R_log_dI_mu)
            R_t = pm.Deterministic('R_t', R_log_I_t[0])
            R_t_drift = pm.Normal('R_t_drift', mu=0, sigma=self.R_t_drift)
            R_t_1 = pm.Deterministic('R_t_1', R_t + R_t_drift)

            # Now, take the new I_t_1
            # Effective serial_interval is basically 9, from empirical tests.
            serial_interval = 9.
            gamma = 1/serial_interval
            log_dI_t = R_log_I_t[1]
            dI_t = pm.Deterministic('dI_t', pm.math.exp(log_dI_t))
            exp_rate = pm.math.exp((R_t_1 - 1) * gamma)
            # Restrict I_t to be nonzero
            dI_t_1_mu = pm.math.maximum(0.1, dI_t * exp_rate)
            dI_t_1 = pm.Poisson('dI_t_1', mu=dI_t_1_mu)

            # From here, find the expected number of positive cases
            N_t_1 = self.fixed_N_t if self.N_t_1 == -1 else self.N_t_1 # For now, assume random tests among a large set.
            positives = HyperGeometric(name='positives',
                                       N = N_t_1, n=self.num_tests, k=dI_t_1,
                                       observed=self.num_positive)

            if self.extra_evidence:
                # Add this extra observation
                N_t = self.fixed_N_t if self.N_t == -1 else self.N_t # For now, assume random tests among a large set.
                yesterday_positives = HyperGeometric(name='yesterday_positives',
                                                    N = N_t, n=self.last_tests, k=dI_t,
                                                    observed=self.last_positives)


            if self.verbose > 2:
                print('Built model, sampling...')
                print(model.test_point)
                for RV in model.basic_RVs:
                    print(RV.name, RV.logp(model.test_point))

            self.trace = pm.sample(
                chains=chains,
                tune=tune,
                draws=draws,
                nuts={"target_accept": target_accept},
                cores=cores
            )
            return self


def create_and_run_models(args):
    verbose = args.verbose
    if verbose:
        print(vars(args))
    data = pd.read_csv(args.infile)
    data_start = data[data.P_t >= args.cutoff].index[0]
    data = data.loc[data_start:]
    # Now, from the start date, we will project forward and
    # compute our Rts and Its.
    R_t_mu, R_t_sigma = args.rt_init_mu, args.rt_init_sigma
    I_t_mu = data.iloc[0].P_t
    log_I_t_mu = np.log(I_t_mu)
    n_days = len(data) if args.window == -1 else args.window

    R_t_mus, R_t_lows, R_t_highs = [R_t_mu], [R_t_mu - R_t_sigma * 1.96], [R_t_mu + R_t_sigma * 1.96]
    I_t_mus, I_t_lows, I_t_highs = [I_t_mu], [-np.inf], [np.inf]
    # create the joint distribution of (R, I)
    R_log_I_mu = np.array([R_t_mu, log_I_t_mu])
    R_log_I_cov = np.array([[R_t_sigma, 0], 
                            [0, R_t_sigma]]) # Start with even
    for i in range(1, n_days):
        day = data.iloc[i]
        yesterday = data.iloc[i-1]
        last_tests, last_positives = yesterday.T_t, yesterday.P_t
        model = MCMCModel(args.infile, R_t_drift=args.rt_drift,
                          num_positive=day.P_t, num_tests=day.T_t,
                          last_tests=last_tests, last_positives=last_positives,
                          N_t_1=day.N_t, N_t=yesterday.N_t,
                          use_real_nt=args.real_nt, fixed_N_t=args.fixed_nt,
                          R_log_I_mu=R_log_I_mu, R_log_I_cov=R_log_I_cov,
                          verbose=args.verbose).run(
                              chains=args.chains,
                              tune=args.tune,
                              draws=args.draw,
                              cores=args.cores
                          )

        I_t = model.trace['dI_t']
        I_t_1 = model.trace['dI_t_1']        
        log_I_t_1 = np.log(I_t_1 + 1e-9)
        R_t_1 = model.trace['R_t_1']

        R_log_I_t = np.stack((R_t_1, log_I_t_1))
        R_log_I_mu = np.mean(R_log_I_t, axis=1)
        R_log_I_cov = np.nan_to_num(np.cov(R_log_I_t), nan=0.01)

        R_t_mu = np.mean(R_t_1)
        R_t_bounds = az.hdi(R_t_1, 0.95)
        R_t_low, R_t_high = R_t_bounds[0], R_t_bounds[1]
        I_t_mu = np.mean(I_t_1)
        I_t_bounds = az.hdi(I_t_1, 0.95)
        I_t_low, I_t_high = I_t_bounds[0], I_t_bounds[1]

        if verbose:
            print(i)
            print(f'R_t: {(R_t_mu, R_t_low, R_t_high)}')
            print(f'I_t: {(I_t_mu, I_t_low, I_t_high)}')
            if verbose > 1:
                print(f'R_t_sigma: {(np.std(R_t_1))}')
                print('Skew, kurtosis: ', skew(R_t_1), kurtosis(R_t_1))
            if verbose > 5:
                print(f'Sample R_t: {R_t_1[:10]}')
                print(f'Sample I_t: {I_t[:10]}')
                print(f'Sample I_t_1: {I_t_1[:10]}')
            print(R_log_I_cov)

        R_t_mus.append(R_t_mu)
        R_t_highs.append(R_t_high)
        R_t_lows.append(R_t_low)
        I_t_mus.append(I_t_mu)
        I_t_highs.append(I_t_high)
        I_t_lows.append(I_t_low)

        if args.save_every:
            results = pd.DataFrame({
                'R_t_mean': np.array(R_t_mus),
                'R_t_low': np.array(R_t_lows),
                'R_t_high': np.array(R_t_highs),
                'I_t_mean': np.array(I_t_mus),
                'I_t_low': np.array(I_t_lows),
                'I_t_high': np.array(I_t_highs),
            })

            results.to_csv(args.outfile)

    results = pd.DataFrame({
        'R_t_mean': np.array(R_t_mus),
        'R_t_low': np.array(R_t_lows),
        'R_t_high': np.array(R_t_highs),
        'I_t_mean': np.array(I_t_mus),
        'I_t_low': np.array(I_t_lows),
        'I_t_high': np.array(I_t_highs),
    })

    results.index = data.index[:n_days]
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='R_t estimation with MCMC')

    parser.add_argument(
        '--infile',
        type=str,
        default='synthetic_data/new_testing/bd_smooth.csv',
        help='File from which to read P_t and T_t (default: %(default)s)',
        nargs='?',
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (default: %(default)d)',
        nargs='?',
    )

    parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of cores to use in compute (default: %(default)d)',
        nargs='?',
    )

    parser.add_argument(
        '--rt_init_mu',
        type=float,
        default=2.0,
        help='Initial mean for Rt on day 0.  (default: %(default)f)'
    )

    parser.add_argument(
        '--real_nt',
        action='store_true',
        help='Use the real N(t) value for synthetic data. (default: False)'
    )

    parser.add_argument(
        '--save_every',
        action='store_true',
        help='Save at every step (default: False)'
    )

    parser.add_argument(
        '--fixed_nt',
        type=int,
        default=10_000_000,
        help='Fixed N_t to use for estimation (default: %(default)d)'
    )

    parser.add_argument(
        '--rt_drift',
        type=float,
        default=0.05,
        help='Variance of drifting R_t (default: %(default)f)'
    )

    parser.add_argument(
        '--rt_init_sigma',
        type=float,
        default=0.5,
        help='Initial variance for Rt on day 0.  (default: %(default)f)'
    )

    parser.add_argument(
        '--cutoff', type=int,
        help='Minimum number of positive tests from which to start inference (default: %(default)d)',
        nargs='?', default=10
    )

    parser.add_argument(
        '--window', type=int,
        help='Number of days to compute R_t for (default: %(default)d)',
        nargs='?', default=-1
    )

    parser.add_argument(
        '--chains', type=int,
        help='Number of chains to use in the MCMC (default: %(default)d)',
        nargs='?', default=1
    )

    parser.add_argument(
        '--tune', type=int,
        help='Number of steps to tune MCMC for (default: %(default)d)',
        nargs='?', default=1_000
    )

    parser.add_argument(
        '--draw', type=int,
        help='Number of samples to draw using MCMC (default: %(default)d)',
        nargs='?', default=1_000
    )

    parser.add_argument(
        '--outfile',
        type=str,
        help='Output file to save data to. (default: %(default)s)',
        nargs='?',
        default='/tmp/bd.csv',
    )

    return parser.parse_args()

def main():
    args = parse_args()
    results = create_and_run_models(args)
    results.to_csv(args.outfile)

if __name__ == '__main__':
    main()
