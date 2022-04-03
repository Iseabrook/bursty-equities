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
#This script specifically contains code for exploring the datasets of transactions 
#of five FTSE instruments (data not available for publication)
##############################################################################
# Instructions for use. The user should populate path_to_data and data_subset 
# as relevant for their use. The code will then produce the following plots: 
#plot_venue_bars: barplot showing the percentage of transactions across different
#venues for different financial instruments
#
#plot_counterparty_bars: barplot showing the percentage of transactions across 
#the top 10 counterparties for different financial instruments
#
#rolling_burstiness: plots of the rolling burstiness in windows of 
#200 transactions, alongside Augmented Dickey Fuller test results for stationarity
#
#plot_edge_burst_hists: plots histograms of the individual edge burstinesses.
#
#burstiness_grouped_trades: plots the burstiness as in rolling_burstiness, 
#for the case when transactions are grouped together. 
#
#point_plots_edgelevel: produces points plots of the transactions for individual 
#edges.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as md


def burstiness(inter_trade_times):
    '''
    This function takes in a series of inter-trade times, and returns the burstiness
    as defined in Holme et. al., Temporal Networks (2011)
    '''
    sigma = inter_trade_times.std()
    mu = inter_trade_times.mean()
    B = (sigma-mu)/(sigma+mu)
    return(B)

def plot_venue_bars(all_insts):
    """This function takes in a dataframe containing transactions for several
    datasets, and produces a barplot with separate bars for each instrument, 
    and the bars split and coloured by the venue the transaction was executed
    at. 
    
    Parameters:
        all_inst: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time']
        
    Returns: None
    """
    dfu = all_insts.groupby(['instrument'])['venue id'].value_counts()#.unstack()
    dfu = pd.DataFrame(dfu)
    dfu.columns = ['venue_id']
    dfu.reset_index(inplace=True)
    dfu.columns = ['instrument', 'venue_id', 'counts']
    dfu.set_index(['instrument', 'venue_id'], inplace=True)
    dfu_largest = dfu['counts'].groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
    
    dfu_largest = pd.DataFrame(dfu_largest)
    dfu['venue_id']  = np.where(dfu.index.isin(dfu_largest.index), dfu.reset_index()['venue_id'], 'other')
    dfu.reset_index(level=1, drop=True, inplace=True)
    dfu = dfu.reset_index().groupby(['instrument','venue_id']).counts.sum().unstack()
    df_percent = dfu.div(dfu.sum(axis=1), axis=0).mul(100).round(1)
    ax = df_percent.plot(kind='barh', figsize=(7, 5), ylabel='Count', rot=0, stacked=True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.xlabel('Transaction count')
    plt.ylabel('Intstrument')
    plt.title('Transaction counts per venue')
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        if width != 0.0:
            ax.annotate(str(int(width))+'%', xy=(left+width/2, bottom+height/2), 
                        ha='center', va='center', size = 12)
            
            
def plot_counterparty_bars(all_insts):
    """This function takes in a dataframe containing transactions for several
    datasets, and produces a barplot with separate bars for each instrument, 
    and the bars split and coloured by the counterparty for the top 5 
    counterparties. 
    
    Parameters:
        all_inst: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time']
        
    Returns: None
    """
    dfu = all_insts.groupby(['instrument'])['buyer id'].value_counts()
    dfu = pd.DataFrame(dfu)
    dfu.columns = ['buyer_id']
    dfu.reset_index(inplace=True)
    dfu1 = all_insts.groupby(['instrument'])['seller id'].value_counts()
    dfu1=pd.DataFrame(dfu1)
    dfu1.columns = ['seller_id']
    dfu1.reset_index(inplace=True)
    dfu.columns=['instrument', 'transactor id', 'counts']
    dfu1.columns=['instrument', 'transactor id', 'counts']
    dfu.set_index(['instrument', 'transactor id'], inplace=True)
    dfu1.set_index(['instrument', 'transactor id'], inplace=True)
    dfu = dfu1.fillna(0).add(dfu.fillna(0), fill_value=0)
    dfu_largest = dfu['counts'].groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
    
    dfu_largest = pd.DataFrame(dfu_largest)
    
    dfu['transactor id']  = np.where(dfu.index.isin(dfu_largest.index), dfu.reset_index()['transactor id'], 'other')
    dfu.reset_index(level=1, drop=True, inplace=True)
    dfu = dfu.reset_index().groupby(['instrument','transactor id']).counts.sum().unstack()
    df_percent = dfu.div(dfu.sum(axis=1), axis=0).mul(100).round(1)
    ax = df_percent.plot(kind='barh', figsize=(7, 5), ylabel='Count', rot=0, stacked=True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.xlabel('Transaction count')
    plt.ylabel('Intstrument')
    plt.title('Transaction counts per counterparty')
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        if width != 0.0:
            ax.annotate(str(int(width))+'%', xy=(left+width/2, bottom+height/2), 
                        ha='center', va='center', size = 12)

def rolling_burstiness(raw_data):
    """This function takes in a dataframe containing transactions a single instrument,
    and plots of the rolling burstiness in windows of 200 minutes across time. 
    It also produces the results of an augmented Dickey-Fuller test for stationarity
    of the rolling burstiness time series
    
    Parameters:
        raw_data: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time']
        
    Returns: None
    """
    
    raw_data['total_value'] = raw_data.groupby(['buyer id', 'seller id', 'trade date time'])['total_value'].transform('sum')
    
    raw_data = raw_data.drop_duplicates(subset=['buyer id', 'seller id', 'trade date time'])
    pd.Series(raw_data['trade date time'].unique(), index = raw_data['trade date time'].unique()).diff().rolling(200, min_periods=50, closed='both', center=True).apply(burstiness).dropna().plot(alpha=0.7)
    rda_test = pd.DataFrame(raw_data['trade date time'])
    rda_test['diff'] = rda_test.diff()
    rda_test.drop('trade date time', axis=1, inplace=True)
    rda_test.index = range(1,len(rda_test)+1)
    X = rda_test.rolling(200).apply(burstiness)
    print('burstiness: ', burstiness(pd.Series(sorted(raw_data['trade date time'].unique())).diff()))
    
    result = adfuller(X.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Burstiness')
    plt.ylim(0,0.7)
    
    plt.legend(loc='lower right', bbox_to_anchor=(2.2,0.5))
    plt.show()

def plot_density(raw_data):
    """This function takes in a dataframe containing transactions a single instrument,
    and plots of the rolling burstiness in windows of 200 minutes across time. 
    
    Parameters:
        raw_data: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time', 'trade_date_time_unformatted']
        
    Returns: None
    """
   
    fig, ax = plt.subplots()
    def transaction_density(data):
        return(len(data)/(data.max()-data.min()))

    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    raw_data['total_value'] = raw_data.groupby(['buyer id', 'seller id', 'trade_date_time_unformatted'])['total_value'].transform('sum')

    raw_data = raw_data.drop_duplicates(subset=['buyer id', 'seller id', 'trade_date_time_unformatted'])
    raw_data.set_index('trade_date_time_unformatted')['trade date time'].rolling('600s', min_periods=20).apply(transaction_density).fillna(method='pad')[50:].plot(label='All transactions', alpha=0.7)

    plt.xlabel('Time of day')
    plt.ylabel('Density')

    plt.legend(loc='lower right', bbox_to_anchor=(2.2,0.5))
    plt.show()

def edge_burstiness(data):
    """This function computes the burstiness for an individual edge's transaction
    sequence 
    Parameters: 
        data: dataframe containing transactions for a single edge with columns
        ['buyer id', 'seller id', 'trade date time']
        
    Returns: burstiness values of individual edges 
    """
    df = data.copy()
    df=df.sort_values(by=['buyer id', 'seller id', 'trade date time'])
    df['tdt_diff']=df.groupby(['buyer id', 'seller id'])['trade date time'].diff()
    return(df.groupby(['buyer id', 'seller id'])['tdt_diff'].apply(burstiness))
    
def plot_edge_burst_hists(raw_data):
   """This function takes in a dataframe containing transactions a single instrument,
    and plots histograms of the bursinesses for the individual edges, and returns 
    the proportion of the dataset with a burstiness of >0.4
    
    Parameters:
        raw_data: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time', 'trade_date_time_unformatted']
        
    Returns: proportion of edges in the dataset that have a burstiness of >0.4
    """    

    raw_data['total_value'] = raw_data.groupby(['buyer id', 'seller id', 'trade_date_time_unformatted'])['total_value'].transform('sum')

    raw_data = raw_data.drop_duplicates(subset=['buyer id', 'seller id', 'trade_date_time_unformatted'])
    real_edge_burstinesses = edge_burstiness(raw_data).dropna()
    real_edge_burstinesses.hist(alpha=0.5, label='All transactions')
    plt.xlabel('Burstiness of edge')
    plt.ylabel('Number of edges')
    
    plt.show()
    return((real_edge_burstinesses>0.4).sum()/len(real_edge_burstinesses))
    


def group_trades(df, n):
   """This function takes in a dataframe containing transactions a single instrument,
    and groups trades which have an inter-trade time of less than or equal to n.
    
    Parameters:
        df: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time', 'trade_date_time_unformatted']
        n: float for the minimum inter-trade time
    Returns: dataframe of grouped transactions
    """    
    #for each pair of traders, find their initial trade
    #need to change so that n is the time between consecutive trades

    #get indicator for whether the time between trades is >n
    df.loc[:,'itt_bool'] =df.loc[:,'trade date time'].diff()>n
    #get trade groups by summing 
    df.loc[:,'itt_bool_cumsum'] = df.loc[:,'itt_bool'].cumsum()
    df.loc[:,"type2"] = np.cumsum((df["itt_bool_cumsum"] != df["itt_bool_cumsum"].shift(1)))
    df.loc[:,"total_value"] = df[["total_value","type2"]].groupby("type2").cumsum()
#     display(df[['trade date time','itt_bool', 'itt_bool_cumsum', 'type2', 'total_value']])
    df.drop_duplicates(subset= ['buyer id', 'seller id', 'type2'], keep='first', inplace=True)
    return(df[['buyer id', 'seller id', 'trade date time','trade_date_time_unformatted', 'total_value', 'date', 'time']])

def burstiness_grouped_trades(raw_data):
   """This function takes in a dataframe containing transactions a single instrument,
    and plots the rolling burstiness over a window of 200 transactions for the case
    of raw transactions in comparison to trades which have been grouped if they have 
    inter-trade times less than 1 second. 
    
    Parameters:
        raw_data: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time', 'trade_date_time_unformatted']
        
    Returns: None
    """    
    
    raw_data['total_value'] = raw_data.groupby(['buyer id', 'seller id', 'trade_date_time_unformatted'])['total_value'].transform('sum')

    raw_data = raw_data.drop_duplicates(subset=['buyer id', 'seller id', 'trade_date_time_unformatted'])
    pd.Series(raw_data['trade date time'].unique(), index = raw_data['trade date time'].unique()).diff().rolling(400, min_periods=50, closed='both', center=True).apply(burstiness).dropna().plot(label='Raw transactions', alpha=0.7)
    raw_data = pd.DataFrame()
    #loop through the individual buyer-seller pairs and aggregate together their trade groups. 
    n=1/60
    for x in raw_data_raw.variable.unique():
        agg_res = group_trades(raw_data_raw.loc[raw_data_raw.variable==x], n)
        raw_data = raw_data.append(agg_res)
    raw_data.sort_values(by='trade date time', inplace=True, ascending=True)
    pd.Series(raw_data['trade date time'].unique(), index = raw_data['trade date time'].unique()).diff().rolling(400, min_periods=50, closed='both', center=True).apply(burstiness).dropna().plot(label='Grouped transactions', alpha=0.7)

    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Burstiness')
    plt.ylim(0,0.7)

    plt.show()

def point_plots_edgelevel(raw_data):
   """This function takes in a dataframe containing transactions a single instrument,
    and plots points corresponding to each transaction, for each edge across time
    
    Parameters:
        raw_data: dataframe containing the columns ['instrument', 'venue_id', 
        'buyer id', 'seller id', 'trade date time', 'trade_date_time_unformatted']
        
    Returns: None
    """    
    
    n=0
    fig = plt.figure(figsize=(15,5))
    fig.add_subplot(1, 3, 1)
    
    for edge in raw_data['variable'].unique():
        plt.plot(raw_data[raw_data['variable']==edge]['trade date time'].unique(), np.zeros_like(raw_data[raw_data['variable']==edge]['trade date time'].unique())+n, 'k.')
        plt.title('Real data')
        plt.ylabel('Edge')
        plt.yticks([])
        plt.xlabel('Time')
        n+=1


if __name__ == "__main__":

    #populate the path_to_data and data_subset 
    path_to_data = 'path_to_data'
    data_subset = 'INSERT SUBSET HERE'
    
    all_ds=[]
    for instrument_code in instrument_codes:
        raw_data = pd.read_csv(path_to_data+instrument_code+"_allvenues.csv", index_col=0)
        raw_data['instrument']=instrument_code
        if not data_subset.isna():
            raw_data = raw_data[raw_data['venue id']==data_subset]

        all_ds.append(raw_data)

    all_insts = pd.concat(all_ds)
    
    plot_venue_bars(all_insts)
    
    plot_counterparty_bars(all_insts)
    
    adf_result = rolling_burstiness(raw_data)
    
    prop_bursty_edges = plot_edge_burst_hists(raw_data)
    
    burstiness_grouped_trades(raw_data)
    
    point_plots_edgelevel(raw_data)