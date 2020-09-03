import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_data, mcf, mcfcost, plot_data, plot_mcf
from scipy.stats import gamma, uniform, expon, norm

# Cook (2008, pg. 367) D.5 Artificial Field Repair Data
# In Example 8.1 some results were presented on the analysis of an artificial dataset on field repairs. The data were
# generated as follows. The time-homogeneous event rate for subject i was gamma distributed with mean 2 and variance 0.5
# and this was used to generate events over (0, ti], where ti ~ Unif(1,3). At the jth event time for subject i, the cost
# was generated independently as Cij ~ N(10, 2.52).
def generate_data(n=134, mean_gamma=2, var_gamma=0.5, min_unif=1, max_unif=3, mean_norm=10, std_norm=2.52):
    rows = []
    for i in range(1, n+1):
        rate = gamma.rvs(mean_gamma**2/var_gamma, scale=var_gamma/mean_gamma)
        t_i = uniform.rvs()*(max_unif-min_unif) + min_unif
        t_ij = expon.rvs(scale=1./rate)
        while t_ij <= t_i:
            cost = norm.rvs(loc=mean_norm, scale=std_norm)
            rows.append((i, t_ij, 1, cost))
            t_ij += expon.rvs(scale=1./rate)
        rows.append((i, t_i, 0, np.nan))
    return pd.DataFrame(rows, columns=['Sample', 'Time', 'Event', 'Cost'])

if __name__ == '__main__':

    # Cook (2008, pg. 299) Example 8.1: Field Repair data
    #   This dataset (see Appendix D) gives simulated data on unscheduled repairs
    #   for a fleet of m = 134 large utility vehicles operated by a city. The data were
    #   collected over a three-year period on new vehicles which were purchased and
    #   placed in service over the first two years of the study. Time is measured in
    #   years from the start of the study, and costs are in hundreds of dollars.
    raw_data = generate_data()
    df = transform_data(raw_data)

    # Table D.4.

    # Marked point process (8.14) as mcf1 and cumulative cost process (8.12) as mcf2
    lt1 = mcf(df, robust=True, positive=False)
    mcf1 = mcfcost(df, mcf_compound=lt1, robust=False, positive=False)
    mcf2 = mcfcost(df, robust=True, positive=False)

    # Plot data
    plot_data(df, alpha=0.5, marker='.', plot_costs=False)

    # Plot MCFs
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)
    plot_mcf(mcf1, ax=ax1, cost=True, label='Events * Cost')
    plot_mcf(mcf2, ax=ax2, cost=True, label='Cost')

    # Table 8.1.
    mcf1['E[C].Std'] = np.sqrt(mcf1['E[C].Var'])
    mcf2['E[C].Std'] = np.sqrt(mcf2['E[C].Var'])
    mcf1 = mcf1[['Time', 'E[C]', 'E[C].Std']]
    mcf2 = mcf2[['Time', 'E[C]', 'E[C].Std']]
    mcf1 = mcf1.set_index('Time').reindex([0.50, 1.00, 1.50, 2.00, 2.50], method='nearest')
    mcf2 = mcf2.set_index('Time').reindex([0.50, 1.00, 1.50, 2.00, 2.50], method='nearest')
    mcf1.rename(columns={'E[C]': 'EST. (8.14)', 'E[C].Std': 'S.E. (8.15)'}, inplace=True)
    mcf2.rename(columns={'E[C]': 'EST. (8.12)', 'E[C].Std': 'S.E. (8.13)'}, inplace=True)
    mcfC = pd.concat((mcf1, mcf2), axis=1)
    print mcfC

    plt.show()
