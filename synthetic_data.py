import argparse

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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
             base_rt=2.2):
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

    def generate_rt_sequence(N=N_days, mean=base_rt, var=0.05, seed=0):
        # Generate rt as a random walk.
        start = np.random.normal(mean, 8*var)
        # Put a small downwards bias
        steps = np.random.normal(-0.005, var, N_days)
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

        y0 = np.array([country.population - 1, 0, 
                       1, 0, 0, 0, 0, 0])
        return diff_eq, y0

    rt = generate_rt_sequence()
    diff_eqs, initial_values = diff_eqs_for_country(country, rt_sequence=rt)
    ivp_solution = solve_ivp(fun=diff_eqs, y0=initial_values, vectorized=True,
                             t_span=(0, end_time), t_eval=np.arange(end_time))

    S, E, I, P, H, U, R, D = (ivp_solution.y[i] for i in range(8))

    # Now, we combine I with the seasonal diseases data
    # First, we take the 2 month-in data, and estimate that with proper testing
    # 10% would be positive.
    median_seasonal = I[60] * 9
    flu_period = 60 # flu oscillation periods
    flu_series = (median_seasonal * np.ones_like(I) * 0.9 +
                  np.sin(np.arange(N_days) * 2 * np.pi / flu_period) * 0.1)
    
    # Now, we figure out how testing scales, and positives scale with it.
    total_eligibles = (flu_series + I).astype(int)
    total_infected = I.astype(int)

    num_tests= np.linspace(num_tests_start, num_tests_end, num=N_days).astype(int)

    num_positives = np.random.binomial(num_tests, 
                                       total_infected / total_eligibles)

    return total_infected, rt, num_tests, num_positives
    

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
        '--outfile',
        type=str,
        help='Output file to save data to. Format should be <country>_<last date of observation>.csv (default: %(default)s)',
        nargs='?',
        default='synthetic_data/bd.csv',
    )

    return parser.parse_args()

def main():
    args = set_up_parser()
    I, rt, num_tests, num_positives = generate(args.country, args.num_tests_start, 
                                               args.num_tests_end, args.num_days)

    results_df = pd.DataFrame({
        'rt': rt[:-1],
        'total_infected': np.cumsum(I),
        'total_tests': np.cumsum(num_tests),
        'total_positives': np.cumsum(num_positives),
        'pct_positive': 100. * num_positives/num_tests
    })

    results_df.to_csv(args.outfile)

main()