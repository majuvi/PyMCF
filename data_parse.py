import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from data_operations import parse_data, transform_data

if __name__ == '__main__':


    # Transmission Example
    raw_data = pd.read_csv('datas_raw/transmission.csv', sep=';', usecols=[0,1,2])
    raw_data.rename(columns={'Car': 'Sample', 'Mileage': 'Time.str'}, inplace=True)

    df = parse_data(raw_data)
    df = df[['Sample', 'Time', 'Event', 'Transmission']]
    print df.head()
    df.to_csv('datas/transmission_recurrent.csv', index=False, sep=';')

    #df = pd.read_csv('datas/transmission_recurrent.csv', sep=';')
    #fr = transform_data(df)
    #print fr.head()


    # Bladder Example
    raw_data = pd.read_csv('datas_raw/bladder.csv', sep=';', usecols=[0,1,2])
    raw_data.rename(columns={'ID': 'Sample', 'Months': 'Time.str'}, inplace=True)

    df = parse_data(raw_data)
    df = df[['Sample', 'Time', 'Event', 'Drug']]
    print df.head()
    df.to_csv('datas/bladder_recurrent.csv', index=False, sep=';')

    #df = pd.read_csv('datas/bladder_recurrent.csv', sep=';')
    #fr = transform_data(df)
    #print fr.head()


    # CGD example
    raw_data = pd.read_csv('datas_raw/CGD.csv', sep=';')
    raw_data.rename(columns={'id': 'Sample'}, inplace=True)
    raw_data['Time.str'] = np.where(raw_data['status'] == 1, raw_data['tstop'].astype(str), raw_data['tstop'].astype(str) + '-')
    raw_data = raw_data[['Treatment', 'Sample', 'Time.str']]

    df = parse_data(raw_data)
    df = df[['Sample', 'Time', 'Event', 'Treatment']]
    print df.head()
    df.to_csv('datas/CGD_recurrent.csv', index=False, sep=';')

    #df = pd.read_csv('datas/CGD_recurrent.csv', sep=';')
    #fr = transform_data(df)
    #print fr.head()

    # Fan Example
    raw_data = pd.read_csv('datas_raw/fan.csv', sep=';')#, usecols=[0,1,2])
    raw_data.rename(columns={'Time': 'Time.str'}, inplace=True)

    df = parse_data(raw_data)
    population = lambda fr: pd.Series({'Censored': (fr['Event'] == 0).sum(),
                                       'Events': (fr['Event'] == 1).sum(),
                                       'Cost': fr['Cost'].sum()})
    df = df.groupby('Time').apply(population).reset_index()
    df['Censored'] = df['Censored'].astype(int)
    df['Events'] = df['Events'].astype(int)
    df = df[['Time', 'Events', 'Censored', 'Cost']]
    print df.head()
    df.to_csv('datas/fan_population.csv', index=False, sep=';')


    # Compressor Example
    raw_data = pd.read_csv('datas_raw/compressor.csv', sep=';')#, usecols=[0,1,2])

    df = raw_data.fillna(0)
    df['Truncated'] = np.where(df['Samples'] > 0, df['Samples'], 0).astype(int)
    df['Censored'] = np.where(df['Samples'] < 0, -df['Samples'], 0).astype(int)
    df['Events'] = df['Events'].astype(int)
    df = df[['Time', 'Events', 'Censored', 'Truncated', 'Building']]
    print df.head()
    df.to_csv('datas/compressor_population.csv', index=False, sep=';')

    # Proschan Example
    raw_data = pd.read_csv('datas_raw/proschan2.csv', sep=';')#, usecols=[0,1,2])
    raw_data.sort_values(['Sample', 'Time'], inplace=True)
    raw_data['Event'] = 1
    last_observation = raw_data.groupby('Sample', as_index=False).apply(lambda fr: fr.iloc[-1,:])
    last_observation['Event'] = 0
    df = pd.concat((raw_data, last_observation))
    df.sort_values(['Sample', 'Time'], inplace=True)
    print df.head()
    df.to_csv('datas/proschan_recurrent.csv', index=False, sep=';')

    # Valve Seats (Nelson 1995) Example
    raw_data = pd.read_csv('datas_raw/nelson1995.csv', sep=';')#, usecols=[0,1,2])
    df = raw_data
    print df.head()
    df.to_csv('datas/nelson1995_recurrent.csv', index=False, sep=';')


    # Childbirths Example
    raw_data = pd.read_csv('datas_raw/childbirths.csv', sep=';')#, usecols=[0,1,2])
    df = raw_data
    df = df[['Time', 'Events', 'Censored', 'Sex']]
    print df.head()
    df.to_csv('datas/childbirths_interval.csv', index=False, sep=';')

    # Defrost Example
    raw_data = pd.read_csv('datas_raw/defrost2.csv', sep=';')#, usecols=[0,1,2])
    df = raw_data
    df['Censored'] = df['AtRisk'] - df['AtRisk'].shift(-1).fillna(0).astype(int)
    df = df[['Time', 'Events', 'Censored']]
    print df.head()
    df.to_csv('datas/defrost_interval.csv', index=False, sep=';')

    # Traction Motor Example
    raw_data = pd.read_csv('datas_raw/traction.csv', sep=';')#, usecols=[0,1,2])
    df = raw_data.fillna(0).astype(int)

    censored = df[['Time', 'Censored']]
    censored = censored.rename(columns={'Censored': 'Total'})
    censored['Event'] = 'Censored'
    events = df.drop('Censored', axis=1)
    events.columns.name = 'Type'
    events = events.set_index('Time').stack()
    events.name = 'Total'
    events = events.reset_index()
    events['Event'] = 'Event'
    df = pd.concat((censored, events)).sort_values(['Time','Event','Type'])
    df = df[['Time', 'Event', 'Type', 'Total']]

    print df.head()
    df.to_csv('datas/traction_multiple.csv', index=False, sep=';')

    # Gearboxes example
    def process(df, type):
        df = raw_data
        df = df.set_index('Time')
        df.columns.name = 'SaleMonth'
        df = df.stack()
        df.name = 'Total'
        df = df.astype(int)
        df = df.reset_index()
        df['Type'] = type
        return df

    raw_data = pd.read_csv('datas_raw/gearbox_censored.csv', sep=';')#, usecols=[0,1,2])
    censored = process(raw_data, 'Censored')
    raw_data = pd.read_csv('datas_raw/gearbox_events.csv', sep=';')#, usecols=[0,1,2])
    events = process(raw_data, 'Events')
    raw_data = pd.read_csv('datas_raw/gearbox_cost.csv', sep=';')#, usecols=[0,1,2])
    costs = process(raw_data, 'Costs')
    df = pd.concat((censored, events, costs))
    df = pd.pivot_table(df, index=['SaleMonth', 'Time'], columns=['Type'], values='Total').fillna(0)
    df = df.reset_index().astype(int)
    df = df.sort_values(['SaleMonth', 'Time'])

    df.to_csv('datas/gearbox_recurrent.csv', index=False, sep=';')
