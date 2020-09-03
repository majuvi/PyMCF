import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_population, mcf, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs, population_reverse

if __name__ == '__main__':

    # Defrost Control Example
    raw_data = pd.read_csv('datas/defrost_interval.csv', sep=';')#, usecols=[0,1,2])
    df = transform_population(raw_data)
    lt1 = mcf(df, robust=False, positive=False, interval=True)
    print df.head()

    fr = population_reverse(df)
    plot_data(fr, alpha=0.005)

    fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True)
    plot_mcf(lt1, ax=ax1, interval=True)
    plt.ylim([0.0, 0.1])
    plt.show()

    #lt_cost = mcf_cost(fr, mcf_compound=lt)
    #print lt_cost.head()
    #plot_cost(mcf_cost)
