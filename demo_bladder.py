import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_operations import transform_data, mcf, mcfdiff, mcfequal, mcfequal_k, logrank, logrank_k,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs, print_data

if __name__ == '__main__':

    # Bladder Example
    raw_data = pd.read_csv('T45.csv', sep=';')
    raw_data.rename(columns={'Patient number': 'Sample', 'Event': 'Number', 'Censored': 'Event',
                             'Treatment group': 'Drug'}, inplace=True)
    raw_data['Drug'] = raw_data['Drug'].map({1: 'Placebo', 2: 'Pyridoxine', 3: 'Thiotepa'})
    raw_data['Event'] = 1 - raw_data['Event']
    print raw_data.head()

    df = transform_data(raw_data)
    lt = mcf(df, robust=True, positive=False)

    covariate = 'Drug'
    cohort1, cohort2 = 'Placebo', 'Thiotepa'

    group_df = df.set_index(covariate)
    group_lt = df.groupby(covariate).apply(lambda sfr: mcf(sfr, robust=True, positive=False))

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
    p_value0 = logrank(df1, df2)
    p_value1 = mcfequal(df1, df2)
    p_value2 = mcfequal(df1, df2, robust=True)
    print "p-values: %.3f / %.3f (robust %.3f)" % (p_value0, p_value1, p_value2)

    plt.show()

    group_df = raw_data.groupby(covariate).apply(lambda sfr: transform_data(sfr))
    dfs = list(df.groupby(covariate))
    lts = [(cohort, mcf(cohort_df, robust=False, positive=False)) for cohort, cohort_df in dfs]

    plot_datas(dfs, alpha=0.5)
    plot_mcfs(lts)

    cohort3 = 'Pyridoxine'
    df3 = group_df.ix[cohort3]
    p_value0 = logrank_k(df1, df2, df3)
    p_value1 = mcfequal_k(df1, df2, df3)
    p_value2 = mcfequal_k(df1, df2, df3)
    print "p-values: %.3f / %.3f (robust %.3f)" % (p_value0, p_value1, p_value2)

    plt.show()