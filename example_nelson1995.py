import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_data, mcf, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs

if __name__ == '__main__':

    # Valve Seats (Nelson 1995) Example
    data = pd.read_csv('datas/nelson1995_recurrent.csv', sep=';')

    df = transform_data(data, format='recurrent')
    lt = mcf(df, robust=True, positive=True)
    print lt[['Time', 'Y', 'N', 'dE[N]', 'E[N]', 'E[N].lcl', 'E[N].ucl']].head()

    # Plot, no stratification
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_data(df, ax=ax1)
    plot_mcf(lt, ax=ax2)
    plt.tight_layout()

    plt.show()
