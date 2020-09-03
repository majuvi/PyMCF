# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_operations import transform_data, mcf, mcfcost, mcfdiff, mcfequal,\
    plot_data, plot_datas, plot_mcf, plot_mcfs, plot_mcfdiff, plot_mcfdiffs

# Recurrent event homogeneous Poisson process with a hidden terminal event
class MarkovRecurrentTerminal(object):

    def __init__(self, play_rate, purchase_rate, quit_rate, playtime_dist, purchase_dist):
        self.play_rate = play_rate
        self.purchase_rate = purchase_rate
        self.quit_rate = quit_rate
        self.playtime_dist = playtime_dist
        self.purchase_dist = purchase_dist

    def __str__(self):
        return "Play rate: %s\nPurchase rate: %s\nQuit rate: %s" % (self.play_rate, self.purchase_rate, self.quit_rate)

    def Sample(self, n, cens):
        T = []
        for i in range(1, n+1):
            Ti = []
            tmax = stats.uniform.rvs() * cens
            tiq = stats.expon.rvs() / self.quit_rate
            tij = stats.expon.rvs() / self.play_rate
            while tij <= tmax and tij < tiq:
                Ti.append((tij, 1, 0, self.playtime_dist.rvs()))
                tij = tij + stats.expon.rvs() / self.play_rate
            tij = stats.expon.rvs() / self.purchase_rate
            while tij <= tmax and tij < tiq:
                Ti.append((tij, 1, 1, self.purchase_dist.rvs()))
                tij = tij + stats.expon.rvs() / self.purchase_rate
            Ti.append((tmax, 0, 0, 0))
            T.extend([(i, tij, event, type, cost) for tij, event, type, cost in Ti])
        return pd.DataFrame(T, columns=['Sample', 'Time', 'Event', 'Type', 'Cost'])

if __name__ == '__main__':

    max_date = 180
    play_rate, purchase_rate, churn_rate = 0.1, 0.004, 0.02
    playtime_dist = stats.expon(scale=0.25)
    purchase_dist = stats.rv_discrete(values=((1, 4, 10, 20), (0.80, 0.15, 0.04, 0.01)))

    process = MarkovRecurrentTerminal(play_rate, purchase_rate, churn_rate, playtime_dist, purchase_dist)
    raw_data = process.Sample(300, max_date)

    df_sessions = transform_data(raw_data[(raw_data['Type'] == 0) | (raw_data['Event'] == 0)])
    df_purchases = transform_data(raw_data[(raw_data['Type'] == 1) | (raw_data['Event'] == 0)])

    mcf_sessions = mcfcost(df_sessions, robust=True, positive=True)
    mcf_purchases = mcfcost(df_purchases, robust=True, positive=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=False)
    plot_data(df_sessions, ax=ax1, marker='.', label='Session', alpha=0.2)
    plot_data(df_purchases, ax=ax2, marker='*', label='Purchase', alpha=0.2)
    plot_mcf(mcf_sessions, ax=ax3, cost=True).set_ylabel('Playtime')
    plot_mcf(mcf_purchases, ax=ax4, cost=True).set_ylabel('Lifetime Value')

    T = np.linspace(0, max_date, 100)
    E_playtime = playtime_dist.mean() * play_rate / churn_rate * (1 - np.exp(-churn_rate*T))
    E_purchases = purchase_dist.mean() * purchase_rate / churn_rate * (1 - np.exp(-churn_rate*T))
    ax3.plot(T, E_playtime, linestyle='-', alpha=0.5, color='black', label='True')
    ax4.plot(T, E_purchases, linestyle='-', alpha=0.5, color='black', label='True')
    ax3.set_ylim(0.0,1.8)
    ax4.set_ylim(0.0,0.6)

    plt.show()