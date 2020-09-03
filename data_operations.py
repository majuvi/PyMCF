# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from numpy.linalg import inv
from collections import OrderedDict
from itertools import cycle
from warnings import warn

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'D']
#group_colors = iter(plt.cm.rainbow(np.linspace(0, 1, 2)))
#c1, c2 = next(group_colors), next(group_colors)


def parse_data(fr, integer=True):

    fr['Time'] = fr['Time.str'].str.replace('(\-|\+)', '').astype(int if integer else float)
    fr['Event'] = np.where(fr['Time.str'].str[-1] == '-', 0, 1)

    if 'Cost' in fr:
        fr['Cost'] = fr['Cost'].fillna(0.00)

    fr = fr.drop('Time.str', axis=1)

    return fr


def transform_data(fr, format='recurrent'):
    #TODO: add custom left-truncation

    def idx_exit(fr):
        if format == 'recurrent':
            return (fr['Event'] == 0)
        elif format == 'single':
            return (fr['Event'] == 0) | (fr['Event'] == 1)

    fr_entry = fr[idx_exit(fr)].copy()
    fr_entry['Time'] = 0.0
    fr_entry['Event'] = 2
    fr = pd.concat((fr_entry, fr))
    fr.sort_values(['Sample', 'Time'] if 'Sample' in fr else 'Time', inplace=True)
    fr.index = np.arange(len(fr))

    fr['dY'] = 0
    fr['dN'] = 0

    fr.loc[idx_exit(fr), 'dY'] = -1
    fr.loc[fr['Event'] == 2, 'dY'] = 1
    fr.loc[fr['Event'] == 1, 'dN'] = 1

    if 'Cost' in fr:
        fr['dC'] = fr['Cost'].fillna(0)

    # Validity check: sample has entry & exit, entry < exit, all events in (entry, exit]
    def valid(sfr):
        stimes = sfr.loc[sfr['dY'] == 1, 'Time']
        etimes = sfr.loc[sfr['dY'] == -1, 'Time']
        events = sfr.loc[sfr['dN'] == 1, 'Time']
        enter, exit = stimes.iloc[0], etimes.iloc[0]
        return (len(stimes) == 1) & (len(etimes) == 1) & (enter < exit) & \
               (events > enter).all() & (events <= exit).all()

    #TODO: add population checks
    if 'Sample' in fr:
        # Ignore invalid samples and warn the user
        fr_valid = fr.groupby('Sample').filter(valid)
        if len(fr_valid) < len(fr):
            fr_invalid = fr.groupby('Sample').filter(lambda sfr: ~valid(sfr))
            warn("Ignoring invalid samples:\n" + str(fr_invalid))
            fr = fr_valid

    return fr

def transform_population(fr):

    fr = fr.copy()

    fr.rename(columns={'Events':'dN'}, inplace=True)
    entry = fr['Truncated'] if 'Truncated' in fr else 0
    exit = fr['Censored']
    fr['dY'] = entry - exit

    total = fr['dY'].sum()
    if total < 0:
        if (fr['Time'] == 0).any():
            fr.loc[fr['Time'] == 0, 'dY'] += -total
        else:
            initial = pd.DataFrame({'Time': 0, 'dN': 0, 'dY': -total}, index=[0])
            fr = pd.concat((initial, fr))
            fr.index = np.arange(len(fr))

    if 'Costs' in fr:
        fr['Costs'] = fr['Costs'].fillna(0)
        fr.rename(columns={'Costs': 'dC'}, inplace=True)

    fr.drop('Censored', axis=1, inplace=True)
    if 'Truncated' in fr:
        fr.drop('Truncated', axis=1, inplace=True)

    return fr

def population_reverse(fr):
    fr = fr.groupby('Time')[['dN','dY']].agg('sum').sort_index().reset_index()
    fr['Y'] = fr['dY'].cumsum().shift(1).fillna(0).astype(int)
    fr['Y_next'] = fr['dY'].cumsum().astype(int)

    ids = pd.Index(np.arange(1, max(fr['Y']) + 1), name='Sample')
    fr_time = pd.DataFrame(np.zeros((len(ids), len(fr))) * np.nan, index=ids, columns=fr['Time'])
    for idx, (time, dN, dY, Y, Y_next) in fr.iterrows():
        status = np.zeros(len(ids)) * np.nan
        status[:int(max(Y, Y_next))] = 0
        if Y > 0:
            idx = np.random.choice(np.arange(Y, dtype=int), int(dN))
            status[idx] = 1
        fr_time.loc[:, time] = status

    fr_time = fr_time.stack()
    fr_time.name = 'dN'
    fr_time = fr_time.reset_index()

    fr_enter = fr_time.groupby('Sample')['Time'].agg('min').reset_index()
    fr_enter['dY'] = 1
    fr_enter['dN'] = 0

    fr_exit = fr_time.groupby('Sample')['Time'].agg('max').reset_index()
    fr_exit['dY'] = -1
    fr_exit['dN'] = 0

    fr_event = fr_time[fr_time['dN'] > 0].reset_index()
    fr_event['dY'] = 0

    fr_sample = pd.concat((fr_enter, fr_event, fr_exit)).sort_values(['Sample', 'Time'])
    fr_sample.index = np.arange(len(fr_sample))
    fr_sample = fr_sample[['Sample', 'Time', 'dY', 'dN']]
    return fr_sample


def population_reverse_costs(fr):
    fr = fr.groupby('Time')[['dC', 'dN', 'dY']].agg('sum').sort_index().reset_index()
    fr['Y'] = fr['dY'].cumsum().shift(1).fillna(0).astype(int)
    fr['Y_next'] = fr['dY'].cumsum().astype(int)

    ids = pd.Index(np.arange(1, max(fr['Y']) + 1), name='Sample')
    times = pd.MultiIndex.from_product([['dC', 'dN'], fr['Time']], names=['Type', 'Time'])
    fr_time = pd.DataFrame(np.zeros((len(ids), 2*len(fr))) * np.nan, index=ids, columns=times)
    for idx, (time, dC, dN, dY, Y, Y_next) in fr.iterrows():
        status = np.zeros(len(ids)) * np.nan
        status[:int(max(Y, Y_next))] = 0
        costs = status.copy()
        if Y > 0:
            random_ids = np.random.choice(np.arange(Y, dtype=int), int(dN))
            status[random_ids] = 1
            costs[random_ids] = dC/dN if dN > 0 else 0
        fr_time['dN', time] = status
        fr_time['dC', time] = costs

    fr_time = fr_time.stack()
    fr_time = fr_time.reset_index()

    fr_enter = fr_time.groupby('Sample')['Time'].agg('min').reset_index()
    fr_enter['dY'] = 1
    fr_enter['dN'] = 0
    fr_enter['dC'] = 0

    fr_exit = fr_time.groupby('Sample')['Time'].agg('max').reset_index()
    fr_exit['dY'] = -1
    fr_exit['dN'] = 0
    fr_exit['dC'] = 0

    fr_event = fr_time[fr_time['dN'] > 0].reset_index()
    fr_event['dY'] = 0

    fr_sample = pd.concat((fr_enter, fr_event, fr_exit)).sort_values(['Sample', 'Time'])
    fr_sample.index = np.arange(len(fr_sample))
    fr_sample = fr_sample[['Sample', 'Time', 'dY', 'dN', 'dC']]
    return fr_sample

def print_data(fr, fmt='%.2f'):
    fr = fr.copy()
    fr['Number'] = fr.groupby(['Sample'])['Sample'].transform(lambda s: range(1,len(s)+1))
    fr['Time'] = fr['Time'].map('{:.2f}'.format) + fr['Event'].map({1: '', 0: '+'})

    if not 'Cost' in fr:
        return fr.pivot(index='Sample', columns='Number', values='Time').fillna('')
    else:
        fr['Cost'] = fr['Cost'].map('{:.2f}'.format, na_action='ignore')
        tb1 = fr.pivot(index='Sample', columns='Number', values='Time').fillna('')
        tb2 = fr.pivot(index='Sample', columns='Number', values='Cost').fillna('')
        tb = pd.concat((tb1, tb2), axis=0, keys=['Time', 'Value'], names=['Type', 'Sample'])
        tb = tb.swaplevel().sort_index()
        return tb

def plot_data(fr, ax=None, label='', offset=0, color='black', edge_color='black', marker='o', alpha=1.0, plot_costs=True,
              update_limits=True, order_followup=True):

    fr = fr.copy()

    # If no axis was given, create a new figure
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Order by censoring time
    if order_followup:
        sample_id = fr.loc[fr['dY'] < 0] .sort_values('Time')['Sample']
    else:
        sample_id = fr.loc[fr['dY'] > 0].sort_values('Time', ascending=False)['Sample']
    sample_idx = np.arange(offset + len(sample_id), offset, -1)
    map_to_nth = dict(zip(sample_id, sample_idx))
    fr['Sample.y'] = fr['Sample'].map(map_to_nth)

    # Plot marker at each event time and line from observable start to end
    fr_events = fr[fr['dN'] > 0]
    fr_limits = fr[(fr['dY'] == 1) | (fr['dY'] == -1)]
    fr_limits = fr_limits.pivot(index='Sample.y', columns='dY', values='Time').reset_index()
    fr_limits = fr_limits.rename(columns={1: 'Time.start', -1: 'Time.end'})

    # Plot events as black dots
    ax.scatter(fr_events['Time'], fr_events['Sample.y'], zorder=10, marker=marker, color=color, edgecolors='black', label=label)
    # Plot vertical lines where at risk starts and ends with horizontal line in between
    ax.hlines(fr_limits['Sample.y'], fr_limits['Time.start'], fr_limits['Time.end'], color=edge_color, label='', alpha=alpha)
    ax.scatter(fr_limits['Time.start'], fr_limits['Sample.y'], marker='|', color=edge_color, label='', alpha=alpha)
    ax.scatter(fr_limits['Time.end'], fr_limits['Sample.y'], marker='|', color=edge_color, label='', alpha=alpha)
    # Annotate events with cost
    if 'dC' in fr and plot_costs:
        for i, cost in enumerate(fr_events['dC']):
            ax.annotate("%.2f" % cost, (fr_events['Time'].iloc[i]+.02, fr_events['Sample.y'].iloc[i]+.10), fontsize=8,
                        bbox={'boxstyle':'square', 'fc':'1.0', 'ec':'0.0', 'pad': 0.2})

    if update_limits:
        # If axis were given, expand to fit previous plot
        ax_xmax, ax_ymax = ax.get_xlim()[1], ax.get_ylim()[1]
        ax.set_ylim(0, max(ax_ymax, fr['Sample.y'].max()+1))
        ax.set_xlim(0, max(ax_xmax, fr['Time'].max()*1.05))

    ax.set_title('Event History Plot')
    ax.set_ylabel('Sample')
    ax.set_xlabel('Time')
    if label:
        ax.legend()

    return ax

def plot_datas(group_fr, ax=None, alpha=1.0, plot_costs=True):

    # If no axis was given, create a new figure
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot timelines grouped by cohorts
    idx, marker = 0, cycle(markers)
    for cohort, cohort_fr in group_fr:
        plot_data(cohort_fr, ax, label=str(cohort), marker=marker.next(), offset=idx, alpha=alpha, plot_costs=plot_costs)
        idx += len(cohort_fr['Sample'].unique())

    return ax

def mcf(fr_sample, confidence=0.95, robust=False, positive=False, interval=False, Y_start=0):
    Z = norm.ppf((1.00+confidence)/2.0)
    fr = fr_sample.groupby('Time')[['dN', 'dY']].agg('sum').sort_index().reset_index()

    fr['N'] = fr['dN'].cumsum()

    if not interval:
        fr['Y'] = Y_start + fr['dY'].cumsum().shift(1).fillna(0)
    if interval:
        enter = Y_start + fr['dY'].cumsum().shift(1).fillna(0)
        exit = Y_start + fr['dY'].cumsum()
        fr['Y'] = (enter + exit) / 2.0

    fr['dE[N]'] = (fr['dN']/fr['Y']).fillna(0)
    fr['E[N]'] = fr['dE[N]'].cumsum()

    if robust:
        dYi = pd.pivot_table(fr_sample, values='dY', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
        dNi = pd.pivot_table(fr_sample, values='dN', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
        Yi = dYi.cumsum(axis=1).shift(axis=1).fillna(0)

        Y = Yi.sum(axis=0)
        dN = dNi.sum(axis=0)
        dH = (dN/Y).fillna(0)

        wi = Yi.div(Y, axis=1).fillna(0)
        di = dNi.subtract(dH, axis=1)
        Ti = wi.multiply(di, axis=1).cumsum(axis=1)**2
        Hvar = Ti.sum(axis=0)
        fr['E[N].Var'] = Hvar.values
    else:
        dvar = (fr['dE[N]']/fr['Y']).fillna(0)
        fr['E[N].Var'] = dvar.cumsum()

    if not positive:
        fr['E[N].ucl'] = fr['E[N]'] + Z*np.sqrt(fr['E[N].Var'])
        fr['E[N].lcl'] = fr['E[N]'] - Z*np.sqrt(fr['E[N].Var'])
    else:
        with np.errstate(divide='ignore'):
            fr['E[N].ucl'] = np.exp(np.log(fr['E[N]']) + Z*np.sqrt(fr['E[N].Var'])/fr['E[N]']).fillna(0)
            fr['E[N].lcl'] = np.exp(np.log(fr['E[N]']) - Z*np.sqrt(fr['E[N].Var'])/fr['E[N]']).fillna(0)

    #if exact:
    #    fr['E[N].ucl'] = 0.5*chi2.ppf((1.00+confidence)/2.0, 2*fr['N'] + 2)/fr['Y']
    #    fr['E[N].lcl'] = 0.5*chi2.ppf((1.00-confidence)/2.0, 2*fr['N'])/fr['Y']
    #    fr['E[N].ucl'].fillna(0, inplace=True)
    #    fr['E[N].lcl'].fillna(0, inplace=True)
    #    fr['E[N].ucl'] = fr['E[N].ucl'].where(fr['E[N].ucl'] != np.inf, 0)

    return fr


def mcfcost(fr_sample, mcf_compound=None, confidence=0.95, robust=False, positive=False, interval=False, Y_start=0):

    Z = norm.ppf((1.00+confidence)/2.0)
    fr = fr_sample.groupby('Time')[['dC', 'dN', 'dY']].agg('sum').sort_index().reset_index()

    if not interval:
        fr['Y'] = Y_start + fr['dY'].cumsum().shift(1).fillna(0)
    if interval:
        enter = Y_start + fr['dY'].cumsum().shift(1).fillna(0)
        exit = Y_start + fr['dY'].cumsum()
        fr['Y'] = (enter + exit) / 2.0

    fr['dE[C]'] = (fr['dC']/fr['Y']).fillna(0)
    fr['E[C]'] = fr['dE[C]'].cumsum()

    if mcf_compound is None:
        dYi = pd.pivot_table(fr_sample, values='dY', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
        dCi = pd.pivot_table(fr_sample, values='dC', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
        Yi = dYi.cumsum(axis=1).shift(axis=1).fillna(0)

        Y = Yi.sum(axis=0)
        dC = dCi.sum(axis=0)
        dEC = (dC/Y).fillna(0)

        wi = Yi.div(Y, axis=1).fillna(0)
        di = dCi.subtract(dEC, axis=1)
        Ti = wi.multiply(di, axis=1).cumsum(axis=1)**2 if robust else (wi.multiply(di, axis=1)**2).cumsum(axis=1)
        Cvar = Ti.sum(axis=0)
        fr['E[C].Var'] = Cvar.values
    else:
        cost_fr = fr_sample.loc[fr_sample['dN'] > 0, 'dC']
        N = len(cost_fr)
        E_cost = cost_fr.mean()
        Var_cost = cost_fr.var()

        fr['dE[C]'] = E_cost * mcf_compound['dE[N]'].fillna(0)
        fr['E[C]'] = fr['dE[C]'].cumsum()
        Var_EC = E_cost**2 * mcf_compound['E[N].Var'] + Var_cost/N * mcf_compound['E[N]']**2 #+ Varc**2/N * fr_compound['Ht.Var']
        fr['E[C].Var'] = Var_EC

    if not positive:
        fr['E[C].ucl'] = fr['E[C]'] + Z*np.sqrt(fr['E[C].Var'])
        fr['E[C].lcl'] = fr['E[C]'] - Z*np.sqrt(fr['E[C].Var'])
    else:
        with np.errstate(divide='ignore'):
            fr['E[C].ucl'] = np.exp(np.log(fr['E[C]']) + Z*np.sqrt(fr['E[C].Var'])/fr['E[C]']).fillna(0)
            fr['E[C].lcl'] = np.exp(np.log(fr['E[C]']) - Z*np.sqrt(fr['E[C].Var'])/fr['E[C]']).fillna(0)

    return fr


def plot_mcf(fr, ax=None, cost=False, label='', interval=False, color='black', edge_color='black', marker='o', CI=True):

    # If no axis was given, create a new figure
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # If axis were given, expand to fit previous plot
    ax_xmax, ax_ymax = ax.get_xlim()[1], ax.get_ylim()[1]

    dNA = fr[fr['dN'] > 0]
    dY = fr[fr['dY'] == -1]

    # Draw MCF as a solid step-plot with black dots at events and vertical lines at censoring events
    key = 'E[N]' if not cost else 'E[C]'
    drawstyle = 'steps-post' if not interval else 'default'
    if not interval:
        ax.scatter(dNA['Time'], dNA[key], s=25, marker=marker, color=color, label=label)
        ax.plot(dY['Time'], dY[key], marker='+', linestyle='', color=edge_color, label='')
    else:
        ax.plot(fr['Time'], fr[key], marker=marker, linestyle='', color=edge_color, label=label)
    ax.plot(fr['Time'], fr[key], marker='', linestyle='-', drawstyle=drawstyle, color=edge_color, label='')
    # Draw confidence intervals as dashed step-plots
    if CI:
        ax.plot(fr['Time'], fr[key+'.ucl'], marker='', linestyle='--', drawstyle=drawstyle, color=edge_color, label='')
        ax.plot(fr['Time'], fr[key+'.lcl'], marker='', linestyle='--', drawstyle=drawstyle, color=edge_color, label='')

    ax.set_xlim(0, max(ax_xmax, fr['Time'].max()*1.05))
    ax.set_ylim(0, max(ax_ymax, fr[key+'.ucl'].max()*1.1))
    ax.set_title('Mean Cumulative Function Plot')
    ax.set_ylabel('Mean Number of Events' if not cost else 'Mean Cost')
    ax.set_xlabel('Time')
    if label:
        ax.legend()
    ax.grid(True)

    return ax

def plot_mcfs(group_fr, ax=None, cost=False, robust=False, positive=False, interval=False, CI=True):

    # If no axis was given, create a new figure
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot timelines grouped by cohorts
    idx, marker = 0, cycle(markers)
    for cohort, cohort_fr in group_fr:
        plot_mcf(cohort_fr, ax, cost=cost, label=str(cohort) if cohort else '', marker=marker.next(), interval=interval, CI=CI)

    return ax

def mcfdiff(mcf1, mcf2, confidence=0.95, bonferroni_correction=2):

    if bonferroni_correction > 2:
        n_comparisons = bonferroni_correction*(bonferroni_correction-1)/2
        confidence = 1.00 - (1.00 - confidence)/float(n_comparisons)

    Z = norm.ppf((1.00+confidence)/2.0)

    mcf1 = mcf1.set_index('Time')
    mcf2 = mcf2.set_index('Time')
    idx = mcf1.index.union(mcf2.index)

    dN = (mcf1['dN'].reindex(idx, fill_value=0) != 0) | (mcf2['dN'].reindex(idx, fill_value=0) != 0)
    dY = (mcf1['dY'].reindex(idx, fill_value=0) != 0) | (mcf2['dY'].reindex(idx, fill_value=0) != 0)
    H_diff = mcf1['E[N]'].reindex(idx, method='ffill') - mcf2['E[N]'].reindex(idx, method='ffill')
    H_dvar = mcf1['E[N].Var'].reindex(idx, method='ffill') + mcf2['E[N].Var'].reindex(idx, method='ffill')

    fr_diff = pd.DataFrame({'dN': dN, 'dY': dY, 'E[N].diff': H_diff, 'E[N].diff.Var': H_dvar})
    fr_diff['E[N].diff.ucl'] = fr_diff['E[N].diff'] + Z*np.sqrt(fr_diff['E[N].diff.Var'])
    fr_diff['E[N].diff.lcl'] = fr_diff['E[N].diff'] - Z*np.sqrt(fr_diff['E[N].diff.Var'])

    return fr_diff.reset_index()

def plot_mcfdiff(fr, ax=None, label='', interval=False, CI=True, xlabel=True, ylabel=True, title=True,
                 color='black', edge_color='black', marker='o'):

    # If no axis was given, create a new figure
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # If axis were given, expand to fit previous plot
    ax_xmax, ax_ymin, ax_ymax = ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]

    dNA = fr[fr['dN'] == True]
    dY = fr[fr['dY'] == True]

    drawstyle = 'steps-post' if not interval else 'default'
    ax.scatter(dNA['Time'], dNA['E[N].diff'], s=25, marker='o', color=color, label=label)
    ax.plot(dY['Time'], dY['E[N].diff'], marker='+', linestyle='', color=edge_color, label='')
    ax.plot(fr['Time'], fr['E[N].diff'], marker='', linestyle='-', drawstyle=drawstyle, color=edge_color, label='')
    if CI:
        ax.plot(fr['Time'], fr['E[N].diff.ucl'], marker='', linestyle='--', drawstyle=drawstyle, color=edge_color, label='')
        ax.plot(fr['Time'], fr['E[N].diff.lcl'], marker='', linestyle='--', drawstyle=drawstyle, color=edge_color, label='')

    lim_ucl = fr['E[N].diff.ucl'].max()
    lim_lcl = fr['E[N].diff.lcl'].min()
    ymin, ymax = lim_lcl*1.1 if lim_lcl < 0 else lim_lcl*0.9, lim_ucl*1.1 if lim_ucl > 0 else lim_ucl*0.9
    xmax = fr['Time'].max()*1.05
    ax.set_ylim(min(ymin, ax_ymin), max(ymax, ax_ymax))
    ax.set_xlim(0, max(xmax, ax_xmax))
    if title: ax.set_title('Mean Cumulative Difference Plot')
    if ylabel: ax.set_ylabel('Mean Difference of Events')
    if xlabel: ax.set_xlabel('Time')
    if label:
        ax.legend()
    ax.grid()

    return ax

def plot_mcfdiffs(group_fr, axs=None, robust=False, positive=False, interval=False):

    # If no axis was given, create a new figure
    nrows, ncols = len(group_fr), len(group_fr[0])
    if axs is None:
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)

    # Plot timelines grouped by cohorts
    i, j = 0, 0
    idx, marker = 0, cycle(markers)
    for row in group_fr:
        for cohort_1, cohort_2, cohort_fr in row:
            if cohort_fr:
                plot_mcfdiff(cohort_fr, axs[i][j], label=str(cohort_1) + ' vs. ' + str(cohort_2),
                             marker=marker.next(), CI=(i != j), title=(i == 0), xlabel=(i == nrows-1), ylabel=(j == 0),
                             interval=interval)
            j +=1
        j = 0
        i += 1

    return axs


def var(fr, ws=None):
    dN = fr['dN']
    dY = fr['dY']
    if ws is not None:
        dY = dY.reindex(ws.index, fill_value=0)
        dN = dN.reindex(ws.index, fill_value=0)

    Y = dY.cumsum().shift(1).fillna(0)
    dH = (dN/Y).fillna(0)
    T = (dH/Y).fillna(0)
    if not ws is None:
        T = T.multiply(ws**2)

    Hvar = T.cumsum()
    return Hvar

def robust_var(fr_sample, ws=None):
    #TODO: possible to combine?
    dYi = pd.pivot_table(fr_sample, values='dY', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
    dNi = pd.pivot_table(fr_sample, values='dN', index='Sample', columns='Time', aggfunc='sum', fill_value=0)
    if not ws is None:
        dYi = dYi.reindex(columns=ws.index, fill_value=0)
        dNi = dNi.reindex(columns=ws.index, fill_value=0)
    Yi = dYi.cumsum(axis=1).shift(axis=1).fillna(0)

    dN = dNi.sum(axis=0)
    Y = Yi.sum(axis=0)
    dH = (dN/Y).fillna(0)

    di = dNi.subtract(dH, axis=1)
    wi = Yi.div(Y, axis=1).fillna(0)
    if not ws is None:
        wi = wi.multiply(ws, axis=1)
    Ti = wi.multiply(di, axis=1).cumsum(axis=1)**2

    Hvar = Ti.sum(axis=0)
    return Hvar

'''
def var(mcf, ws=None):
    dN = mcf.set_index('Time')['dN']
    dY = mcf.set_index('Time')['dY']
    if ws is not None:
        dY = dY.reindex(ws.index, fill_value=0)
        dN = dN.reindex(ws.index, fill_value=0)

    Y = dY.cumsum().shift(1).fillna(0)
    dH = (dN/Y).fillna(0)
    T = (dH/Y).fillna(0)
    if not ws is None:
        T = T.multiply(ws**2)

    Hvar = T.cumsum()
    return Hvar

def robust_var(fr_sample, ws=None):
    fr = fr_sample.groupby('Time')[['dN', 'dY']].agg('sum').sort_index().reset_index()
    fr['Y'] = fr['dY'].cumsum().shift(1).fillna(0)
    fr['ht'] = (fr['dN']/fr['Y']).fillna(0)

    def sample_var(gfr, Y, dH):
        dYi = gfr['dY'].reindex(ws.index).fillna(0)
        dNi = gfr['dN'].reindex(ws.index).fillna(0)

        Yi = dYi.cumsum().shift().fillna(0)
        wi = Yi / Y
        wi = wi * ws
        di = dNi - dH
        T = (wi * di).cumsum()**2
        return T

    df_sample = fr_sample.set_index('Time')
    Y = fr.set_index('Time')['Y']
    dH = fr.set_index('Time')['ht']

    T_var = pd.Series(np.zeros_like(Y), index=Y.index)
    i = 0
    for sample, gfr in df_sample.groupby('Sample'):
        if i % 100 == 0: print i
        T_var += sample_var(gfr, Y, dH)
        i += 1
    print
'''

def fr_agg(fr_sample):
    fr = fr_sample.groupby('Time')[['dN', 'dY']].agg('sum').sort_index().reset_index()
    fr['N'] = fr['dN'].cumsum()
    fr['Y'] = fr['dY'].cumsum().shift(1).fillna(0)
    fr['dE[N]'] = (fr['dN']/fr['Y']).fillna(0)
    return fr


def mcfequal(fr_sample1, fr_sample2, confidence=0.95, robust=False):
    #TODO: drop Y = 0, compare multiple
    Z = norm.ppf((1.00+confidence)/2.0)
    fr1 = fr_agg(fr_sample1)
    fr2 = fr_agg(fr_sample2)
    fr1 = fr1.set_index('Time')
    fr2 = fr2.set_index('Time')
    idx = fr1.index.union(fr2.index)

    Y1 = fr1['dY'].reindex(idx, fill_value=0).cumsum().shift(1).fillna(0)
    Y2 = fr2['dY'].reindex(idx, fill_value=0).cumsum().shift(1).fillna(0)
    h1 = fr1['dE[N]'].reindex(idx, fill_value=0)
    h2 = fr2['dE[N]'].reindex(idx, fill_value=0)

    w = (Y1 * Y2/(Y1 + Y2)).fillna(0)
    U_score = (w * (h1 - h2)).sum()

    if not robust:
        U_var = var(fr1, w).iloc[-1] + var(fr2, w).iloc[-1] #(w**2 * (h1/Y1 + h2/Y2)).sum()
    else:
        U_var = robust_var(fr_sample1, w).iloc[-1] + robust_var(fr_sample2, w).iloc[-1]

    p_value = chi2.sf(U_score**2 / U_var, 1)
    #print U_score, U_var, U_score**2/U_var, p_value
    return p_value



def mcfequal_k(*fr_samples):
    confidence = 0.95
    Z = norm.ppf((1.00+confidence)/2.0)
    k = len(fr_samples)

    cohorts = range(k)
    fr_aggs = [fr_agg(fr_cohort).set_index('Time') for fr_cohort in fr_samples]
    fr = {cohort: fr_cohort.set_index('Time') for cohort, fr_cohort in zip(cohorts, fr_samples)}

    dY = pd.DataFrame({idx: fr_agg['dY'] for idx, fr_agg in zip(cohorts, fr_aggs)}).fillna(0)
    dN = pd.DataFrame({idx: fr_agg['dN'] for idx, fr_agg in zip(cohorts, fr_aggs)}).fillna(0)
    dN_ = dN.sum(axis=1)
    Y = dY.cumsum(axis=0).shift(1, axis=0).fillna(0)
    Y_ = Y.sum(axis=1)

    Z = np.array([(Y[cohort] * (dN[cohort] / Y[cohort] - dN_ / Y_)).sum() for cohort in cohorts])

    V = np.zeros((k, k))
    for cohort_i in cohorts:
        for cohort_j in cohorts:
            var = 0.0
            for cohort in cohorts:
                c_i = int(cohort == cohort_i) - Y[cohort_i] / Y_
                c_j = int(cohort == cohort_j) - Y[cohort_j] / Y_
                for sample, fr_sample in fr[cohort].groupby('Sample'):
                    fr_sample = fr_sample.groupby(level=0).agg('sum')
                    Y_sample = fr_sample['dY'].reindex(Y.index, fill_value=0).cumsum().shift(1).fillna(0)
                    dN_sample = fr_sample['dN'].reindex(Y.index, fill_value=0)
                    var_sample = (Y_sample * c_i * (dN_sample - dN[cohort] / Y[cohort])).sum()*\
                                 (Y_sample * c_j * (dN_sample - dN[cohort] / Y[cohort])).sum()
                    var += var_sample
            V[cohort_i, cohort_j] = var

    Z = Z[:-1]
    V = V[:-1,:-1]

    score = np.dot(np.dot(Z, inv(V)), Z)
    p_value = chi2.sf(score, k - 1)
    #print Z, V, score, p_value
    return p_value


def logrank(fr_sample1, fr_sample2):
    confidence = 0.95
    Z = norm.ppf((1.00+confidence)/2.0)
    fr1 = fr_agg(fr_sample1)
    fr2 = fr_agg(fr_sample2)
    fr1 = fr1.set_index('Time')
    fr2 = fr2.set_index('Time')
    idx = fr1.index.union(fr2.index)

    Y1 = fr1['dY'].reindex(idx, fill_value=0).cumsum().shift(1).fillna(0)
    Y2 = fr2['dY'].reindex(idx, fill_value=0).cumsum().shift(1).fillna(0)
    h1 = fr1['dE[N]'].reindex(idx, fill_value=0)
    h2 = fr2['dE[N]'].reindex(idx, fill_value=0)
    dN1 = fr1['dN'].reindex(idx, fill_value=0)
    dN2 = fr2['dN'].reindex(idx, fill_value=0)

    w1 = (Y1 * Y2/(Y1 + Y2)).fillna(0)
    w2 = ((Y1 * Y2)/(Y1 + Y2)**2).fillna(0)
    U_score = (w1 * (h1 - h2)).sum()
    U_var = (w2 * (dN1 + dN2)).sum()

    score = U_score**2 / U_var
    p_value = chi2.sf(score, 1)
    #print U_score, U_var, score, p_value

    return p_value


def logrank_k(*fr_samples):
    confidence = 0.95
    Z = norm.ppf((1.00+confidence)/2.0)
    k = len(fr_samples)

    cohorts = range(k)
    fr_aggs = [fr_agg(fr_cohort).set_index('Time') for fr_cohort in fr_samples]

    dY = pd.DataFrame({idx: fr_agg['dY'] for idx, fr_agg in zip(cohorts, fr_aggs)}).fillna(0)
    dN = pd.DataFrame({idx: fr_agg['dN'] for idx, fr_agg in zip(cohorts, fr_aggs)}).fillna(0)
    dN_ = dN.sum(axis=1)
    Y = dY.cumsum(axis=0).shift(1, axis=0).fillna(0)
    Y_ = Y.sum(axis=1)

    K = Y.all(axis=1).astype(int)
    Z = np.array([(K * (dN[cohort] - Y[cohort] * (dN_ / Y_))).sum() for cohort in cohorts])
    V = np.array([[(K * Y[cohort_i] / Y_ * (int(cohort_i == cohort_j) - Y[cohort_j] / Y_) * dN_).sum()
                   for cohort_j in cohorts]
                  for cohort_i in cohorts])
    Z = Z[:-1]
    V = V[:-1,:-1]

    score = np.dot(np.dot(Z, inv(V)), Z)
    p_value = chi2.sf(score, k - 1)
    #print Z, V, score, p_value
    return p_value

# List of [(cohort tuple, cohort dataframe), ...] as defined by covariates
# group_fr = fr.groupby(covariates) if covariates else [('', fr)]


if __name__ == '__main__':
    pass