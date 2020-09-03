import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_population, mcf, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs

if __name__ == '__main__':

    # Childbirths Example
    raw_data = pd.read_csv('datas/childbirths_interval.csv', sep=';')#, usecols=[0,1,2])

    cohort1, cohort2 = 'Male', 'Female'
    df1 = transform_population(raw_data[raw_data['Sex'] == cohort1]).fillna(cohort1)
    df2 = transform_population(raw_data[raw_data['Sex'] == cohort2]).fillna(cohort2)
    df = pd.concat((df1, df2))

    lt1 = mcf(df1, robust=False, positive=False)
    lt2 = mcf(df2, robust=False, positive=False)
    lt_diff = mcfdiff(lt1, lt2)

    # Plot, stratified
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, sharex=True)
    plot_mcf(lt1, ax=ax1, interval=True, CI=True, label=cohort1)
    plot_mcfs([(cohort1, lt1), (cohort2, lt2)], ax=ax2, interval=True, CI=False)
    plot_mcfdiff(lt_diff, ax=ax3, label='%s vs. %s' % (cohort1, cohort2))
    ax1.set_ylim([0, 2])
    ax2.set_ylim([0, 2])
    ax3.set_ylim([-1, 1])
    plt.xlim([0, 60])

    # Compute p-values
    p_value1 = mcfequal(df1, df2)
    print "p-value: %.3f" % (p_value1)

    plt.show()
