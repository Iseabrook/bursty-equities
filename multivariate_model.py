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
#This script specifically contains the code to fit a multivariate Hawkes process 
#to transaction sequences across multiple counterparties transacting a ftse 100 
#financial instrument. It uses a brute force method to find the best decay, and uses
#the ADM4 method presented by Zhou et. al. in AISTATS (Vol. 31, pp. 641-649) to estimate
#the baseline and kernel intensity parameters. The script then simulates using the 
#estimated parameters 1000 times to allow for confidence intervals to be estimated, 
#and also compares to a multivariate Poisson according to the burstiness of the processes.
##############################################################################
# Instructions for use. The user should populate path_to_data, data_subset and 
#save_data_name as relevant for their use. The code includes the functionality to 
#subset the data according to the 'venue id' column if required. The script then 
#proceeds as above, producing the burstiness of the resulting simulated processes in
#comaprison to the real burstiness.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import HawkesADM4, SimuHawkesExpKernels, SimuHawkesMulti
from tick.hawkes import SimuPoissonProcess
    
def burstiness(inter_trade_times):
    '''
    This function takes in a series of inter-trade times, and returns the burstiness
    as defined in Holme et. al., Temporal Networks (2011)
    '''
    sigma = inter_trade_times.std()
    mu = inter_trade_times.mean()
    B = (sigma-mu)/(sigma+mu)
    return(B)
    
if __name__ == "__main__":
    #populate the path_to_data and data_subset 
    path_to_data = 'path_to_data'
    data_subset = 'INSERT SUBSET HERE'
    save_data_name = 'INSERT NAME TO SAVE DATA'
    raw_data = pd.read_csv(path_to_data, index_col=0)
    if not data_subset.isna():
        raw_data = raw_data[raw_data['venue id']==data_subset]
    timestamps = raw_data.groupby('variable')['trade date time'].apply(np.array).values.tolist()
    
    timestamps = [[i for i in timestamps if len(i)>10]]

    decays = np.linspace(0.1, 50,100)
    
    burstinesses = []
    sim_ts_list=[]
    
    #since the ADM4 method only estimates the baseline and kernel intensity 
    #parameters, we use a brute force method for the decay, selecting the decay
    #which produces the best burstiness of the resulting sequence. 
    for decay in decays: 
        n_realizations = len(timestamps[0])
    
        learner = HawkesADM4(decay)
        learner.fit(timestamps)
        end_times = [max(x) for x in timestamps[0]]
        hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=learner.adjacency, 
                                                  decays=decay,
                                                  baseline=learner.baseline,
                                                  verbose=False, seed=1039)
    
        multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)
    
        multi.end_time = end_times
        multi.simulate()
    
        #evaluate
        sim_timestamps = multi.timestamps[0]
        sim_ts_list.append(sim_timestamps)
        flat_list = [item for sublist in sim_timestamps for item in sublist]
        unlisted_sim_ts = sorted(flat_list)
        burstinesses.append(burstiness(pd.Series(pd.Series(unlisted_sim_ts).unique()).diff()))

    real_bursts = [burstiness(pd.Series(sorted(i)).diff().dropna()) for i in timestamps[0]]
    best_decays=[]
    for i, r in enumerate(real_bursts):
        print(r)
        burstinesses = [burstiness(pd.Series(i).diff().dropna()) for i in pd.DataFrame(sim_ts_list).iloc[:,i].values]
        plt.scatter(decays[0:len(burstinesses)], burstinesses)
        plt.show()
        best_decays.append(decays[min(range(len(burstinesses)), key=lambda i: abs(burstinesses[i]-r))])
    
    #current methods only allow for a single decay for all edges, so we 
    # fit the decay on the edge that occurs the most. 
    best_decay = pd.Series(best_decays).median()
    
    # use the estimated parameters and best decay to simulate, bootstrapping 
    # with 1000 runs to get confidence intervals on the burstiness. 
    burstinesses = []
    timestamps_list = [] 
    for i in range(1000):
        end_times = [max(x) for x in timestamps[0]]
    
        learner = HawkesADM4(best_decay)
        learner.fit(timestamps)
    
        hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=learner.adjacency, decays=best_decay,
                                                  baseline=learner.baseline, 
                                                  verbose=False)
        n_realizations = len(timestamps[0])
        multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)
    
        multi.end_time = end_times
        multi.simulate()
    
        #evaluate
        sim_timestamps = multi.timestamps[0]
        timestamps_list.append(sim_timestamps)
        flat_list = [item for sublist in sim_timestamps for item in sublist]
        unlisted_sim_ts = sorted(flat_list)
        burstinesses.append(burstiness(pd.Series(pd.Series(unlisted_sim_ts).unique()).diff()))
    
    np.save(save_data_name, timestamps_list)
    
    #burstiness of sequence as a whole
    print(len(burstinesses))
    print('lower',sorted(burstinesses)[25])
    print('upper', sorted(burstinesses)[975])
    print('mean', np.nanmean(burstinesses))
    print('real', burstiness(pd.Series(sorted([item for sublist in timestamps[0] for item in sublist])).diff()))
    
    #burstinesses of the individual edges
    ts = pd.DataFrame(raw_data.groupby('variable')['trade date time'].apply(np.array))
    ts=ts[ts['trade date time'].str.len()>100]
    timestamps_series = pd.Series(timestamps[0], index = ts.index)
    
    list_sim_timestamps_series = [pd.Series(i, index = ts.index) for i in timestamps_list]
    for edge_idx in timestamps_series.index:
        ts_edge = timestamps_series[edge_idx]
        sim_ts_edge = [i[edge_idx] for i in list_sim_timestamps_series]
        print(edge_idx)
        print('real burstiness: ',burstiness(pd.Series(ts_edge).diff()))
        print('mean sim burstiness: ', np.nanmean([burstiness(pd.Series(i).diff()) for i in sim_ts_edge]))
        print('lower sim burstiness: ', sorted([burstiness(pd.Series(i).diff()) for i in sim_ts_edge])[25])
        print('upper sim burstiness: ',sorted([burstiness(pd.Series(i).diff()) for i in sim_ts_edge])[975])
            
    #comparison to multivariate poisson model with intensity as sample mean
    
    #bootstrapping to get confidence intervals on the burstiness. 

    burstinesses_poiss = []
    timestamps_list_poiss = [] 
    for i in range(1000):
        #poisson intensity as the sample mean of itt for each edge
        intensity = [pd.Series(i).dropna().diff().mean() for i in timestamps[0]]
    
    
        poi = SimuPoissonProcess(intensity, end_time=max([len(x) for x in timestamps[0]]), verbose=False)
        poi.simulate()
        sim_timestamps = poi.timestamps
        timestamps_list_poiss.append(sim_timestamps)
        flat_list = [item for sublist in sim_timestamps for item in sublist]
        unlisted_sim_ts = sorted(flat_list)
        burstinesses_poiss.append(burstiness(pd.Series(pd.Series(unlisted_sim_ts).unique()).diff()))
    
    #burstiness of sequence as a whole
    print('lower',sorted(burstinesses_poiss)[24])
    print('upper', sorted(burstinesses_poiss)[975])
    print('mean', np.nanmean(burstinesses_poiss))
    print('real', burstiness(pd.Series(sorted([item for sublist in timestamps[0] for item in sublist])).diff()))
    #think about the confidence intervals here 
    ts = pd.DataFrame(raw_data.groupby('variable')['trade date time'].apply(np.array))#.values.tolist()
    ts=ts[ts[ 'trade date time'].str.len()>100]
    #is this a list or a series? look when ready.
    timestamps_series = pd.Series(timestamps[0], index = ts.index)
    
    list_sim_timestamps_series_poiss = [pd.Series(i, index = ts.index) for i in timestamps_list_poiss]
    for edge_idx in timestamps_series.index:
        ts_edge = timestamps_series[edge_idx]
        sim_ts_edge = [i[edge_idx] for i in list_sim_timestamps_series_poiss]
        print(edge_idx)
        print('real burstiness: ',burstiness(pd.Series(ts_edge).diff()))
        print('mean sim burstiness: ', np.nanmean([burstiness(pd.Series(i).diff()) for i in sim_ts_edge]))
        print('lower sim burstiness: ', sorted([burstiness(pd.Series(i).diff()) for i in sim_ts_edge])[24])
        print('upper sim burstiness: ',sorted([burstiness(pd.Series(i).diff()) for i in sim_ts_edge])[975])
            
