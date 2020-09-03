import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_population, mcf, mcfcost, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs, population_reverse, population_reverse_costs

if __name__ == '__main__':

    # Fan Example
    raw_data = pd.read_csv('datas/fan_population.csv', sep=';')#, usecols=[0,1,2])
    raw_data.rename(columns={'Cost': 'Costs'}, inplace=True)
    df = transform_population(raw_data)
    fr = population_reverse(df)
    fr2 = population_reverse_costs(df)

    lt1 = mcf(df, robust=False, positive=False)
    lt2 = mcf(df, robust=False, positive=True)

    mcf1 = mcfcost(df, mcf_compound=lt1, robust=False, positive=False)
    mcf2 = mcfcost(df, mcf_compound=lt2, robust=False, positive=True)

    plot_data(fr2, alpha=0.5)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=False)
    plot_mcf(lt1, ax=ax1, label='Normal limits')
    plot_mcf(lt2, ax=ax2, label='Normal limits (positive)')
    plot_mcf(mcf1, ax=ax3, cost=True, label='Normal limits')
    plot_mcf(mcf2, ax=ax4, cost=True, label='Normal limits (positive)')
    plt.show()
    ax1.set_ylim([-0.01, 0.51])
    ax2.set_ylim([-0.01, 0.51])
    #lt_cost = mcf_cost(fr, mcf_compound=lt)
    #print lt_cost.head()
    #plot_cost(mcf_cost)
