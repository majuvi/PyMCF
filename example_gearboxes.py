import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_population, mcf, mcfcost, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs, population_reverse, population_reverse_costs

if __name__ == '__main__':

    # Gearboxes Example
    raw_data = pd.read_csv('datas/gearbox_recurrent.csv', sep=';')#, usecols=[0,1,2])
    raw_data = raw_data[raw_data['SaleMonth'].isin([1,2,3,4])]
    df = transform_population(raw_data)
    lt = mcf(df, robust=False, positive=False, interval=True)
    #ct = mcfcost(df, robust=False, positive=False, interval=True)

    #lt_cost = mcf_cost(fr, mcf_compound=lt)
    #print lt_cost.head()
    #plot_cost(mcf_cost)

    covariate = 'SaleMonth'
    group_df = raw_data.groupby(covariate).apply(lambda sfr: transform_population(sfr))
    dfs = list(group_df.groupby(level=0))
    lts = [(cohort, mcf(cohort_df, robust=False, positive=False)) for cohort, cohort_df in dfs]
    cts = [(cohort, mcfcost(cohort_df, mcf_compound=cohort_mcf, robust=False, positive=False)) for\
           (cohort, cohort_df), (cohort, cohort_mcf) in zip(dfs,lts)]
    datas = [(cohort, population_reverse_costs(cohort_df)) for cohort, cohort_df in dfs]

    plot_datas(datas, alpha=0.01)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plot_mcfs(lts, ax=ax1, CI=False)
    plot_mcfs(cts, ax=ax2, cost=True, CI=False)
    ax1.set_ylim([0.0, 0.01])
    plt.show()
