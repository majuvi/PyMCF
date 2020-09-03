# Byar, D., Blackard, C., & Veterans Administration Cooperative Urological Research Group. (1977). Comparisons of placebo, pyridoxine, and topical thiotepa in preventing recurrence of stage I bladder cancer. Urology, 10(6), 556-561.
# https://doi.org/10.1016/0090-4295(77)90101-7

# Download T45.man
#	Andrews, D.F. and Herzberg, A.M. (1985). A Collection of Problems from Many Fields for the Student and Research Worker
#	http://lib.stat.cmu.edu/datasets/Andrews/ (Table 45.1)

import requests
url = 'http://lib.stat.cmu.edu/datasets/Andrews/T45.1'
r = requests.get(url)
with open('datas_raw/T45.man', 'wb') as f:
    f.write(r.content)

# Process T45.man to
#	list1: rows of summary table
#	list2: rows of recurrent events at inspection table
import re
list1 = []
list2 = []
with open('datas_raw/T45.man') as f:
	for i, s in enumerate(f.readlines()):
		#print s[9:12]
		if i % 2 == 0:
			# split by whitespace
			list1.append(s[16:].split())
		else:
			# split 1. by two or more spaces 2. by whitespace
			events = re.split(r'\s{2,}', s[16:])
			events = [event.split() for event in events if event]
			list2.append(events)

# Table 45.1 patient statistics
import numpy as np
import pandas as pd
df1 = pd.DataFrame(list1, columns=['Patient number', 'Treatment group', 'Follow-up time, months', 'Survival status', 'No. of recurrences', 'Initial number', 'Initial size'])

# Table 45.1 recurrent events at inspection: month (M), number (#), size (S)
n, m = len(list2), max([len(l) for l in list2])
cols = [col for i in range(1, m+1) for col in ['M%d'%i, '#%d'%i, 'S%d'%i]]
df2 = pd.DataFrame(np.zeros((n, 3*m))*np.nan, columns=cols)
for i in range(n):
	for j in range(len(list2[i])):
		m, n, s = list2[i][j]
		df2.iloc[i,j*3+0] = m
		df2.iloc[i,j*3+1] = n
		df2.iloc[i,j*3+2] = s
df2 = pd.concat((df1['Patient number'], df2), axis=1)

# Recurrent events at inspection: patient x month (M) in wide and long format
events_wide = df2.loc[:,df2.columns.str.startswith('M') | (df2.columns == 'Patient number')]
events_long = pd.wide_to_long(events_wide, 'M', i='Patient number', j='Event').dropna().reset_index()

# Recurrent events at inspection: long format Treatment group, Patient number, Event, Time, Censored
events_rec = events_long.rename(columns={'M':'Time'})
events_rec['Censored'] = 0
events_end = df1[['Patient number', 'Follow-up time, months']]
events_end.rename(columns={'Follow-up time, months':'Time'}, inplace=True)
events_end['Censored'] = 1
events_next = events_long.groupby('Patient number')['Event'].agg(lambda s: str(s.astype(int).max()+1))
events_end['Event'] = events_end['Patient number'].map(events_next).fillna('1')
events = pd.concat((events_rec, events_end))
events = events.merge(df1[['Patient number', 'Treatment group']], how='left', on='Patient number')

# Final processing and saving
events['Treatment group'] = events['Treatment group'].astype(int)
events['Patient number'] = events['Patient number'].astype(int)
events['Event'] = events['Event'].astype(int)
events['Time'] = events['Time'].astype(int)
events.sort_values(['Patient number', 'Event'], inplace=True)
events = events[['Treatment group', 'Patient number', 'Event', 'Time', 'Censored']]
events.to_csv('datas/T45.csv', index=False, sep=';')