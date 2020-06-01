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
    def __init__(self, region, num_positive, num_tests,
                 dI_t_mu, R_t_mu, R_t_sigma, N_t, use_real_nt=False,
                 R_t_drift = 0.05, verbose=1):

        # Just for identification purposes
        self.region = region

        # For the model, we'll only look at the last N
        self.num_positive = num_positive
        self.num_tests = num_tests
        self.dI_t_mu = dI_t_mu
        self.R_t_mu = R_t_mu
        self.R_t_sigma = R_t_sigma
        self.R_t_drift = R_t_drift
        self.N_t = N_t if use_real_nt else -1

        self.verbose = verbose

        # Where we store the results
        self.trace = None

    def run(self, chains=1, tune=3_000, draws=500, target_accept=.9, cores=1):

        with pm.Model() as model:
            # Figure out the new R_t
            R_t = pm.Normal('R_t', mu=self.R_t_mu, sigma=self.R_t_sigma)
            R_t_drift = pm.Normal('R_t_drift', mu=0, sigma=self.R_t_drift)
            R_t_1 = pm.Deterministic('R_t_1', R_t + R_t_drift)

            # Now, take the new I_t_1
            serial_interval = 5.2
            gamma = 1/serial_interval
            dI_t = pm.Poisson('dI_t', mu=self.dI_t_mu)
            exp_rate = pm.Deterministic('exp_rate', pm.math.exp((R_t_1 - 1) * gamma))
            # Restrict I_t to be nonzero
            dI_t_1_mu = pm.math.maximum(0.1, dI_t * exp_rate)
            dI_t_1 = pm.Poisson('dI_t_1', mu=dI_t_1_mu)

            # From here, find the expected number of positive cases
            N_t_1 = 100_000 if self.N_t == -1 else self.N_t # For now, assume random tests among a large set.
            positives = HyperGeometric(name='positives',
                                       N = N_t_1, n=self.num_tests, k=dI_t_1,
                                       observed=self.num_positive)


            if self.verbose > 1:
                print('Built model, sampling...')
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
    data = pd.read_csv(args.infile)
    data_start = data[data.P_t >= args.cutoff].index[0]
    data = data.loc[data_start:]
    # Now, from the start date, we will project forward and
    # compute our Rts and Its.
    R_t_mu, R_t_sigma = args.rt_init_mu, args.rt_init_sigma
    I_t_mu = args.cutoff # Change this later
    n_days = len(data) if args.window == -1 else args.window

    R_t_mus, R_t_lows, R_t_highs = [R_t_mu], [R_t_mu - R_t_sigma * 1.96], [R_t_mu + R_t_sigma * 1.96]
    I_t_mus, I_t_lows, I_t_highs = [I_t_mu], [-np.inf], [np.inf]
    for i in range(1, n_days):
        day = data.iloc[i]
        model = MCMCModel(args.infile, R_t_drift=args.R_t_drift,
                          num_positive=day.P_t, num_tests=day.T_t,
                          dI_t_mu=I_t_mu, N_t=day.N_t, use_real_nt=args.real_nt,
                          R_t_mu=R_t_mu, R_t_sigma=R_t_sigma,
                          verbose=args.verbose).run(
                              chains=args.chains,
                              tune=args.tune,
                              draws=args.draw,
                              cores=args.cores
                          )

        I_t_1 = model.trace['dI_t_1']
        R_t_1 = model.trace['R_t_1']

        R_t_mu = np.mean(R_t_1)
        R_t_sigma = np.std(R_t_1)
        I_t_mu = np.mean(I_t_1)

        R_t_bounds = az.hdi(R_t_1, 0.95)
        R_t_low, R_t_high = R_t_bounds[0], R_t_bounds[1]
        I_t_bounds = az.hdi(I_t_1, 0.95)
        I_t_low, I_t_high = I_t_bounds[0], I_t_bounds[1]

        if verbose:
            print(i)
            print(f'R_t: {(R_t_mu, R_t_low, R_t_high)}')
            print(f'I_t: {(I_t_mu, I_t_low, I_t_high)}')

        R_t_mus.append(R_t_mu)
        R_t_highs.append(R_t_high)
        R_t_lows.append(R_t_low)
        I_t_mus.append(I_t_mu)
        I_t_highs.append(I_t_high)
        I_t_lows.append(I_t_low)

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
    parser = argparse.ArgumentParser(description='Rt estimation with MCMC')

    parser.add_argument(
        '--infile',
        type=str,
        default='synthetic_data/new_testing/bd.csv',
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
        default=3.5,
        help='Initial mean for Rt on day 0.  (default: %(default)f)'
    )

    parser.add_argument(
        '--real_nt',
        type=bool,
        default=False,
        help='Use the real N(t) value for synthetic data. (default: False)'
    )

    parser.add_argument(
        '--R_t_drift',
        type=float,
        default=0.05,
        help='Variance of drifting R_t (default: %(default)f)'
    )

    parser.add_argument(
        '--rt_init_sigma',
        type=float,
        default=3.,
        help='Initial variance for Rt on day 0.  (default: %(default)f)'
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
        '--chains', type=int,
        help='Number of chains to use in the MCMC (default: %(default)d)',
        nargs='?', default=1
    )

    parser.add_argument(
        '--tune', type=int,
        help='Number of steps to tune MCMC for (default: %(default)d)',
        nargs='?', default=500
    )

    parser.add_argument(
        '--draw', type=int,
        help='Number of samples to draw using MCMC (default: %(default)d)',
        nargs='?', default=500
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
    results = create_and_run_models(args)
    results.to_csv(args.outfile)

if __name__ == '__main__':
    main()
