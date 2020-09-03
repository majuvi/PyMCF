import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_data, mcf, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs

if __name__ == '__main__':

    # CGD example
    raw_data = pd.read_csv('datas/CGD_recurrent.csv', sep=';')

    df = transform_data(raw_data)
    lt = mcf(df, robust=True, positive=True)
    print lt.head()

    covariate = 'Treatment'
    cohort1, cohort2 = 'Placebo', 'Gamma Interferon'
    group_df = df.set_index(covariate)
    group_lt = df.groupby(covariate).apply(lambda sfr: mcf(sfr, robust=True, positive=True))

    df1, df2 = group_df.ix[cohort1], group_df.ix[cohort2]
    lt1, lt2 = group_lt.ix[cohort1], group_lt.ix[cohort2]

    lt_diff = mcfdiff(lt1, lt2)

    # Plot, no stratification
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_data(df, ax=ax1)
    plot_mcf(lt, ax=ax2)
    plt.tight_layout()

    # Plot, stratified
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_datas([(cohort1, df1), (cohort2, df2)], ax=ax1)
    plot_mcfs([(cohort1, lt1), (cohort2, lt2)], ax=ax2)

    # Comparison plot
    fig, ax1 = plt.subplots(1, 1)
    plot_mcfdiff(lt_diff, ax=ax1, label='%s vs. %s' % (cohort1, cohort2))
    plt.tight_layout()

    # Compute p-values
    p_value1 = mcfequal(df1, df2)
    p_value2 = mcfequal(df1, df2, robust=True)
    print "p-values: %.3f (robust %.3f)" % (p_value1, p_value2)

    plt.show()

