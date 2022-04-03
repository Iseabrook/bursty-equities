# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 07:56:11 2022

@author: iseabrook1
"""
#This script contains the code used in Seabrook et. al., Modelling equity 
#transactions as bursty processes. 

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Modelling equity 
#transactions as bursty processes. 
#Specifically, this script contains various code for additional results presented 
#in the referenced paper. 
##############################################################################
# Instructions for use. This script produces the following: 
#1. plots of the assortativity across time for both the original data and the data
#following a rewiring of edges.
#2. plots of the parameter values estimated including the confidence intervals. 
#The user should input the parameter values required manually, along with the 
#confidence intervals. 
#3. timeaveraged burstiness across 4 different days
#4. timeaveraged density across 4 different days. 

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def burstiness(inter_trade_times):
    '''
    This function takes in a series of inter-trade times, and returns the burstiness
    as defined in Holme et. al., Temporal Networks (2011)
    '''
    sigma = inter_trade_times.std()
    mu = inter_trade_times.mean()
    B = (sigma-mu)/(sigma+mu)
    return(B)

def transaction_density(data):
    return(len(data)/(data.max()-data.min()))
    
def temporal_assortativity_rewire(data, n, c, title):
    data.sort_values('trade date time', inplace=True)
    data['buyer id shuff'] =  np.random.permutation(data['buyer id'].values)
    rol = data['trade date time'].rolling(window=n)
    def ass_rolling(ser):
        G_raw = nx.from_pandas_edgelist(data.loc[ser.index], source='seller id', target='buyer id', create_using = nx.Graph)
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        G_raw.remove_nodes_from(list(nx.isolates(G_raw)))
        assortativity = nx.degree_pearson_correlation_coefficient(G_raw)
        return assortativity
    def ass_rolling_swap(ser):
        G_raw = nx.from_pandas_edgelist(data.loc[ser.index], source='seller id', target='buyer id shuff', create_using = nx.Graph)
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        G_raw.remove_nodes_from(list(nx.isolates(G_raw)))
        #G_swap = nx.double_edge_swap(G_raw, nswap=int(G_raw.number_of_nodes()), max_tries=10000, seed=101)
        ass_swap = nx.degree_pearson_correlation_coefficient(G_raw)
        return ass_swap
    rolling_node_ass = rol.apply(ass_rolling, raw=False).dropna()  
    rolling_node_ass_swap = rol.apply(ass_rolling_swap, raw=False).dropna()  

    print(len(rolling_node_ass))
    axs.plot(range(len(rolling_node_ass)), rolling_node_ass.values, color=c, label = 'real' )
    axs.plot(range(len(rolling_node_ass_swap)), rolling_node_ass_swap.values, color='b', label = 'rewired' )
    axs.set_ylim([-0.9, 0])
    axs.set_xlabel("Time")
    axs.set_ylabel("Assortativity")
    
if __name__ == "__main__":

    path_to_data = 'path_to_data'
    data_subset = 'INSERT SUBSET HERE'
    save_data_name = 'INSERT NAME TO SAVE DATA'
    raw_data = pd.read_csv(path_to_data+"_allvenues.csv", index_col=0)
    if not data_subset.isna():
        raw_data = raw_data[raw_data['venue id']==data_subset]
    
    raw_data.index = range(len(raw_data))
    
    fig, axs=plt.subplots(1,1, figsize=(10,5))
    temporal_assortativity_rewire(raw_data, int(len(raw_data)/10), 'r', 'real data')
    plt.legend()
    plt.show()
    
    #plots for paper - example for plot 1. 
    #plot 1: univariate parameter values 
    param_vals = [0.45, 0.09, 0.12, 0.90, 0.23, 0.99, 0.41, 0.42, 0.01, 0.10, 0.23, 2.84, 0.03, 0.01, 0.04, 0.55, 0.35, 0.36, 0.36, 0.56, 0.48, 0.72, 1.05, 0.27, 0.52, 0.56, 0.75, 1.29, 0.93, 1.31, 0.62, 0.63, 0.38, 1.59, 0.66, 0.92, 0.77, 1.33,0.63, 0.57, 0.66, 0.80, 1.30, 0.94, 1.54]
    params_si = [0.08, 0.02, 0.04, 0.04, 0.06, 0.16, np.nan, 0.10, 0.01, 0.10, 0.19, 0.58, 0.03, 0.01, 0.04,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.10, 0.02, 0.03, 0.05, np.nan, 0.04, 0.06, np.nan, np.nan, np.nan, np.nan, np.nan, 0.10, np.nan, np.nan, 0.11, 0.40, 0.45, 0.23, np.nan, 0.20]
    type_params = ['baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline','baseline', 'baseline', 'baseline', 'baseline', 'baseline','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity','kernel intensity', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay', 'kernel decay']
    type_subset = ['full', 'full', 'full', 'full', 'full', 'single venue', 'single venue', 'single venue', 'single venue', 'single venue', 'off exchange', 'off exchange', 'off exchange', 'off exchange', 'off exchange','full', 'full', 'full', 'full', 'full', 'single venue', 'single venue', 'single venue', 'single venue', 'single venue', 'off exchange', 'off exchange', 'off exchange', 'off exchange', 'off exchange' , 'full', 'full', 'full', 'full', 'full', 'single venue', 'single venue', 'single venue', 'single venue', 'single venue', 'off exchange', 'off exchange', 'off exchange', 'off exchange', 'off exchange' ]
    # dataset = ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E','A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E']
    dataset=[1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5]
    results_univ = pd.DataFrame(
        {'param_val': param_vals,
         'param_si': params_si,
         'type_param': type_params,
         'type_subset':type_subset, 
         'dataset': dataset
        })
    results_univ['dataset_shifted'] = results_univ['dataset']+results_univ['type_subset'].replace('full', '-0.1').replace('off exchange', '0').replace('single venue', '0.1').astype('float')
    print(results_univ.dataset_shifted.unique())
    
    matplotlib.rcParams.update({'errorbar.capsize': 3})
    
    colors=['r', 'g', 'b']
    trs = [-5, 0, 5]
    fig= plt.figure(figsize=(15,5))
    
    letters=[np.nan, 'A', 'B', 'C', 'D', 'E', np.nan]
    rows=1
    columns=3
    for i,ds in enumerate(results_univ.type_subset.unique()):
        fig.add_subplot(rows, columns, i+1)
        ax=fig.gca()
        c=0
        for key, group in results_univ[results_univ.type_subset==ds].groupby('type_param'):
            if i==0:
                group.plot( x='dataset_shifted', y='param_val',yerr='param_si', kind='scatter',ax=ax, color=colors[c], label=key)
            else:
                group.plot( x='dataset_shifted', y='param_val',yerr='param_si', kind='scatter',ax=ax, color=colors[c])
    
            c+=1
        labels = [letters[i] for i, item in enumerate(ax.get_xticklabels())]
        print(labels)
        ax.set_title(ds)
        ax.set_xticklabels(labels)
        ax.set_ylabel('parameter value')
        ax.set_ylim((-0.1,3.7))
        ax.set_xlabel('dataset')
    plt.show()
    
    # #timeaveraged burstiness across four days
    path_to_day1 = 'INSERT PATH HERE'
    path_to_day2 = 'INSERT PATH HERE'
    path_to_day3 = 'INSERT PATH HERE'
    path_to_day4 = 'INSERT PATH HERE'
    raw_data1 = pd.read_csv(path_to_day1, index_col=0)
    raw_data1.sort_values(by='trade date time', inplace=True)
    ts1 = pd.Series(raw_data1['trade date time'].unique(), index = raw_data1['trade date time'].unique()).diff().rolling(200, min_periods=50, closed='both', center=True).apply(burstiness).dropna()
    raw_data2 = pd.read_csv(path_to_day2, index_col=0)
    raw_data2.sort_values(by='trade date time', inplace=True)
    ts2 = pd.Series(raw_data2['trade date time'].unique(), index = raw_data2['trade date time'].unique()).diff().rolling(200, min_periods=50, closed='both', center=True).apply(burstiness).dropna()
    raw_data3 = pd.read_csv(path_to_day3, index_col=0)
    raw_data3.sort_values(by='trade date time', inplace=True)
    ts3 = pd.Series(raw_data3['trade date time'].unique(), index = raw_data3['trade date time'].unique()).diff().rolling(200, min_periods=50, closed='both', center=True).apply(burstiness).dropna()
    raw_data4 = pd.read_csv(path_to_day4, index_col=0)
    raw_data4.sort_values(by='trade date time', inplace=True)
    ts4 = pd.Series(raw_data4['trade date time'].unique(), index = raw_data4['trade date time'].unique()).diff().rolling(200, min_periods=50, closed='both', center=True).apply(burstiness).dropna()
    
    ts1.index = ts1.index.to_series().apply(lambda x: np.round(x,1))
    ts1=ts1.groupby(ts1.index).mean()
    ts2.index = ts2.index.to_series().apply(lambda x: np.round(x,1))
    ts2=ts2.groupby(ts2.index).mean()
    ts3.index = ts3.index.to_series().apply(lambda x: np.round(x,1))
    ts3=ts3.groupby(ts3.index).mean()
    ts4.index = ts4.index.to_series().apply(lambda x: np.round(x,1))
    ts4=ts4.groupby(ts4.index).mean()
    
    #union of the indexes
    union_idx = ts1.index.union(ts2.index)
    #reindex with the union
    ts1= ts1.reindex(union_idx)
    ts2= ts2.reindex(union_idx)
    ts_comb= pd.concat([ts1,ts2])
    ts_comb= ts_comb.groupby(level=0).mean()
    union_idx = ts_comb.index.union(ts3.index)
    ts_comb= ts_comb.reindex(union_idx)
    ts3= ts3.reindex(union_idx)
    ts_comb= pd.concat([ts_comb,ts3])
    ts_comb= ts_comb.groupby(level=0).mean()
    union_idx = ts_comb.index.union(ts4.index)
    ts_comb= ts_comb.reindex(union_idx)
    ts4= ts4.reindex(union_idx)
    ts_comb= pd.concat([ts_comb,ts4])
    ts_comb= ts_comb.groupby(level=0).mean()
    ts_comb.index = ts_comb.index.to_series().apply(lambda x: np.round(x))
    ts_comb=ts_comb.groupby(ts_comb.index).mean()
    
    tc_mean = ts_comb.mean()
    tc_std = ts_comb.std()
    plt.plot(ts_comb)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Burstiness')
    plt.axhline(tc_mean+(3*tc_std), color='r', linestyle='--', alpha=0.5)
    plt.axhline(tc_mean-(3*tc_std), color='r', linestyle='--', alpha=0.5)
    plt.ylim(0,1)
    
    #timeaveraged density across 4 days
    raw_data1.trade_date_time_unformatted = raw_data.trade_date_time_unformatted.astype('datetime64')
    ts1 = raw_data1.set_index('trade_date_time_unformatted')['trade date time'].rolling('300s', min_periods=20).apply(transaction_density).fillna(method='pad')[50:]
    
    raw_data2.trade_date_time_unformatted = raw_data.trade_date_time_unformatted.astype('datetime64')
    ts2 = raw_data2.set_index('trade_date_time_unformatted')['trade date time'].rolling('300s', min_periods=20).apply(transaction_density).fillna(method='pad')[50:]
    raw_data3.trade_date_time_unformatted = raw_data.trade_date_time_unformatted.astype('datetime64')
    ts3 = raw_data3.set_index('trade_date_time_unformatted')['trade date time'].rolling('300s', min_periods=20).apply(transaction_density).fillna(method='pad')[50:]
    raw_data4.trade_date_time_unformatted = raw_data.trade_date_time_unformatted.astype('datetime64')
    ts4 = raw_data4.set_index('trade_date_time_unformatted')['trade date time'].rolling('300s', min_periods=20).apply(transaction_density).fillna(method='pad')[50:]
    
    ts1.index = ts1.index.round('T').time
    ts1=ts1.groupby(ts1.index).mean()
    ts2.index = ts2.index.round('T').time
    ts2=ts2.groupby(ts2.index).mean()
    ts3.index = ts3.index.round('T').time
    ts3=ts3.groupby(ts3.index).mean()
    ts4.index = ts4.index.round('T').time
    ts4=ts4.groupby(ts4.index).mean()
    
    #union of the indexes
    union_idx = ts1.index.union(ts2.index)
    #reindex with the union
    ts1= ts1.reindex(union_idx)
    ts2= ts2.reindex(union_idx)
    ts_comb= pd.concat([ts1,ts2])
    ts_comb= ts_comb.groupby(level=0).mean()
    print(ts_comb)
    union_idx = ts_comb.index.union(ts3.index)
    ts_comb= ts_comb.reindex(union_idx)
    ts3= ts3.reindex(union_idx)
    ts_comb= pd.concat([ts_comb,ts3])
    ts_comb= ts_comb.groupby(level=0).mean()
    union_idx = ts_comb.index.union(ts4.index)
    ts_comb= ts_comb.reindex(union_idx)
    ts4= ts4.reindex(union_idx)
    ts_comb= pd.concat([ts_comb,ts4])
    ts_comb= ts_comb.groupby(level=0).mean()
    # ts_comb.index = ts_comb.index.to_series().apply(lambda x: np.round(x))
    ts_comb=ts_comb.groupby(ts_comb.index).mean()
    date = str(datetime.strptime('2021-12-01', '%Y-%m-%d').date())
    ts_comb.index = pd.to_datetime(date + " " + ts_comb.index.astype(str))
    
    tc_mean = ts_comb.mean()
    tc_std = ts_comb.std()
    plt.plot(ts_comb)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Density')
    plt.axhline(tc_mean+(3*tc_std), color='r', linestyle='--', alpha=0.5)
    plt.axhline(max(tc_mean-(3*tc_std), 0), color='r', linestyle='--', alpha=0.5)
    plt.ylim(0,100)


