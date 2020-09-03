import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_population, mcf, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs, population_reverse

if __name__ == '__main__':

    # Compressor Example
    raw_data = pd.read_csv('datas/compressor_population.csv', sep=';')#, usecols=[0,1,2])
    #raw_data = raw_data.groupby('Time', as_index=False).sum()
    df = transform_population(raw_data)
    print population_reverse(df)
    group_df = raw_data.groupby('Building').apply(lambda sdata: population_reverse(transform_population(sdata)))
    dfs = list(group_df.groupby(level=0))

    lt1 = mcf(df, robust=False, positive=False)
    lt2 = mcf(df, robust=False, positive=True)

    plot_datas(dfs, alpha=0.1)

    fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True)
    plot_mcf(lt1, ax=ax1, label='Normal limits')
    plot_mcf(lt2, ax=ax2, label='Normal limits (positive)')
    plt.ylim([-0.005, 0.1])
    plt.show()

    #lt_cost = mcf_cost(fr, mcf_compound=lt)
    #print lt_cost.head()
    #plot_cost(mcf_cost)
