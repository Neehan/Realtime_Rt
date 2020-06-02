import argparse

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    # Loading the death rate from the EJI
    ifr_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSqRiIR-jRruEO7LckXpFO8T-1LNtbOXro5gdPGgxOZ7XorqDQHod5WsCQPD89wyX0OO3Wvayd2au7H/pub?gid=0&single=true&output=csv")
    ifr_data = ifr_data.set_index('Country')
    ifr_data.fp = ifr_data.fp / 100.
    ifr_data.fu = ifr_data.fu / 100.
    ifr_data.fd = ifr_data.fd / 100.
    return ifr_data

def generate(country_name, num_tests_start, num_tests_end, 
             N_days = 180, lowest_rt=0.01, highest_rt=12,
             base_rt=2.2, rt_var=0.05, smooth=False):
    # Setting time to run until
    end_time = N_days

    # Loading data
    ifr_data = load_data()
    country = ifr_data.loc[country_name]

    # Set up the global constants
    sigma = 1/5.2 # incubation period
    gamma = 1/2.3 # infectious period
    omega = 1/2.7 # delay between severe infection and hospitalization
    kappa = 1/8.0 # duration of non-ICU hospital stays
    delta = 1/8.0 # duration of ICU'd hospital stays

    # rt constants
    beta_func = lambda country, rt: (gamma * rt / country.population)

    def generate_rt_sequence(N=N_days, mean=base_rt, var=rt_var, seed=0):
        # Generate rt as a random walk.
        start = np.random.normal(mean, 5*var)
        # Put a small downwards bias
        steps = np.random.normal(-0.003, var, N_days)
        all_steps = np.concatenate((np.array([start]), steps))
        final_rt = np.clip(np.cumsum(all_steps), lowest_rt, highest_rt)
        return final_rt

    def diff_eqs_for_country(country, rt_sequence):
        N = country.population
        beta = lambda time: beta_func(country, rt_sequence[int(time)])
        def diff_eq(time, y):
            # First, seperate out the columns of y
            S, E, I, P, H, U, R, D = (y[i] for i in range(8))
            results = np.zeros_like(y)
            results[0] = -beta(time) * I * S
            results[1] = beta(time) * I * S - sigma * E
            results[2] = sigma * E - gamma * I
            results[3] = country.fp * gamma * I - omega * P
            results[4] = omega * P - kappa * H
            results[5] = country.fu * kappa * H - delta * U
            results[6] = ((1-country.fp) * gamma * I + 
                        (1-country.fu) * kappa * H + 
                        (1-country.fd) * delta * U)
            results[7] = country.fd * delta * U
            return results

        y0 = np.array([country.population - country.initial_infections, 0, 
                       country.initial_infections, 0, 0, 0, 0, 0])
        return diff_eq, y0

    rt = generate_rt_sequence()
    if smooth: rt = savgol_filter(rt, window_length=7, polyorder=2)
    diff_eqs, initial_values = diff_eqs_for_country(country, rt_sequence=rt)
    ivp_solution = solve_ivp(fun=diff_eqs, y0=initial_values, vectorized=True,
                             t_span=(0, end_time), t_eval=np.arange(end_time))

    S, E, I, P, H, U, R, D = (ivp_solution.y[i] for i in range(8))

    # Now, we combine I with the seasonal diseases data
    # First, we take the 2 month-in data, and estimate that with proper testing
    # 10% would be positive.
    median_seasonal = 26_027 # Estimated by US flu statistics 
    # (total medical visits scaled by population divided by 365)
    flu_period = 60 # flu oscillation periods
    flu_series = (median_seasonal * np.ones_like(I) * 0.9 +
                  np.sin(np.arange(N_days) * 2 * np.pi / flu_period) * 0.1)

    # Now, we figure out how testing scales, and positives scale with it.
    new_cases = -np.diff(S)
    new_cases = np.concatenate((np.array([country.initial_infections]), new_cases))
    total_eligibles = (flu_series + new_cases).astype(int)
    total_daily_infected = new_cases.astype(int)

    num_tests= np.linspace(num_tests_start, num_tests_end, num=N_days).astype(int)

    num_positives = np.random.hypergeometric(ngood=total_daily_infected,
                                             nbad=total_eligibles - total_daily_infected,
                                             nsample=np.minimum(num_tests, total_eligibles))

    return I, total_eligibles, rt, num_tests, num_positives, total_daily_infected
    

def set_up_parser():
    parser = argparse.ArgumentParser(description='Rt estimation data generation')

    parser.add_argument(
        '--country',
        type=str,
        default='Bangladesh',
        help='Country whose population model we are using (default: %(default)s)',
        nargs='?',
    )
    parser.add_argument(
        '--num_days', type=int, 
        help='Number of days to generate data with (default: %(default)d)', 
        nargs='?', default=180
    )

    parser.add_argument('--smooth', action='store_true', help='Smooth the Rt curve.')

    parser.add_argument(
        '--num_tests_start', type=int, 
        help='Number of tests the country starts with (default: %(default)d)', 
        nargs='?', default=1000
    )

    parser.add_argument(
        '--num_tests_end', type=int, 
        help='Number of tests the country ends with (default: %(default)d)', 
        nargs='?', default=20_000
    )

    parser.add_argument(
        '--seed', type=int, 
        help='Seed for random generation (default: %(default)d)', 
        nargs='?', default=0
    )

    parser.add_argument(
        '--rt_var',
        type=float,
        default=0.05,
        help='Variance of drifting R_t (default: %(default)f)'
    )

    parser.add_argument(
        '--outfile',
        type=str,
        help='Output file to save data to. Format should be <country>_<last date of observation>.csv (default: %(default)s)',
        nargs='?',
        default='synthetic_data/new_testing/bd.csv',
    )

    return parser.parse_args()

def main():
    args = set_up_parser()
    I, total_eligibles, rt, num_tests, num_positives, new_cases = generate(args.country, args.num_tests_start, 
                                                                           args.num_tests_end, args.num_days, 
                                                                           rt_var=args.rt_var, smooth=args.smooth)

    results_df = pd.DataFrame({
        'R_t': rt[:-1],
        'I_t': (I),
        'N_t': (total_eligibles),
        'T_t': (num_tests),
        'P_t': (num_positives),
        'pct_positive': 100. * num_positives/num_tests,
        'cases': new_cases,
    })

    results_df.to_csv(args.outfile)

main()