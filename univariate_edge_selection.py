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
#This script specifically contains code to select counterparties to transact at 
#each point in a transaction sequence, based on either a random selection, selection 
#based on the frequency at which those counterparties have transacted, or based on the 
#importance of the counterparties as defined in Seabrook et. al., Structural 
#importance and evolution: an application to financial transaction networks. It 
#includes the functionality to run on a chosen subset of the data to reproduce the 
#methods presented in Seabrook et al., Modelling equity transactions as bursty processes. 
##############################################################################
# Instructions for use. The user should specify the location of their data at 
#path_to_raw_data, and where they wish the data to be saved at path_to_save_data. 
#The script then applies the three different methods (random, frequency based 
#and importance based) of edge selection to produce transaction sequences with 
#counterparties selected, and saves the result. It then produces the plots displayed 
#in the referenced paper: 

# Rich club: produces a plot showing the distribution of rich club coefficients 
# at the different degrees for the different datasets, and applies a kolmogorov
# smirnov test for similarity between the distributions of the data produced by 
# the different methods of edge selection. 

# Degree distributions: produces plots of the degree distributions for the different
# methods of edge selection.

# Assortativity: produces plots of the assortativity across time for the different
# methods of edge selection.

# Reciprocity: produces plots of the reciprocity across time for the different 
# methods of edge selection.


from __future__ import division
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ast
import itertools
from numpy import random
from scipy.stats import kstwobign, pearsonr

def myround(x, base=5):
    """ This function takes a date in numerical (float) format and rounds it 
    to a user specified base. 
    
    Parameters: 
        x: float timestamp
        base: base to round x to. 
    
    Returns: x rounded to specified base
    """
    return base * round(x/base)


def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''
    def avgmaxdist(x1, y1, x2, y2):
        D1 = maxdist(x1, y1, x2, y2)
        D2 = maxdist(x2, y2, x1, y1)
        return (D1 + D2) / 2
    def maxdist(x1, y1, x2, y2):
        n1 = len(x1)
        D1 = np.empty((n1, 4))
        for i in range(n1):
            a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
            a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
            D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]
        # re-assign the point to maximize difference,
        # the discrepancy is significant for N < ~50
        D1[:, 0] -= 1 / n1
        dmin, dmax = -D1.min(), D1.max() + 1 / n1
        return max(dmin, dmax)
    def quadct(x, y, xx, yy):
        n = len(xx)
        ix1, ix2 = xx <= x, yy <= y
        a = np.sum(ix1 & ix2) / n
        b = np.sum(ix1 & ~ix2) / n
        c = np.sum(~ix1 & ix2) / n
        d = 1 - a - b - c
        return a, b, c, d
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)

    D = avgmaxdist(x1, y1, x2, y2)
    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p
    
def multi_edge_to_uni(G):
    """This function takes in a graph which has multiple edges between two nodes,
    and sums them to a single edge.
    
    Parameters:
        G: networkx graph with potentially multi-edges
        
    Returns:
        G_uni: graph with multi-edge weights summed to give single total weight.
    """
    G_uni = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['total_value']
        if G_uni.has_edge(u,v):
            G_uni[u][v]['total_value'] += w
        else:
            G_uni.add_edge(u, v, total_value=w)
    return(G_uni)    

def mi_generator_symm_tests(data, node, test_type):
    """ Function to generate the value of m_i for an individual node
    Parameters:
        data: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value and trade date time.
        edge: edge tuple (seller id, buyer id) 
    Returns:
        value of m_i calculated. Where nodes are not found in the giant component, no value is returned.
    """
    M = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",
                                edge_attr = ['total_value'],
                                create_using=nx.MultiGraph())
    G = nx.Graph()#DiGraph if directed
    G.add_nodes_from(M)
    for u,v,data1 in M.edges(data=True):
        w = data1['total_value'] if 'total_value' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    A = nx.to_numpy_matrix(G.to_undirected())          
    S = pd.Series([val for (node, val) in G.degree(weight='weight')], index=[node for (node, val) in G.degree(weight='weight')])
    eigenvalues, eigvecs = np.linalg.eigh(A)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigvecs = eigvecs[:,idx]
    #retrieve all of the eigenvector components for node 
    eigvecs_df = pd.DataFrame(eigvecs, index=G.nodes, columns = eigenvalues)#.abs()
    max_eigvec = eigvecs_df[[x for x in eigenvalues if x>0]].abs().max(axis=1)
    max_eigvec_eigval = eigvecs_df[[x for x in eigenvalues if x>0]].abs().idxmax(axis=1)
    weighted_max_eigvec = max_eigvec.multiply(max_eigvec_eigval)
    lead_eig = eigenvalues[0]
    max_eigvec = weighted_max_eigvec/lead_eig
    grad_A = max_eigvec.loc[node]*max_eigvec
    m_i_b = (2/S.loc[node])*(grad_A.sum()) 
    return(m_i_b)

def random_edge_selection(raw_data_aggregated, timestamps):
    """ Function to randomly select a pair of market participants to transact at 
    each point in a given transaction sequence. 
    
    Parameters: 
        raw_data_aggregated: a dataframe with columns 'variable', 
        'trade date time', 'total_value', 'buyer id', 'seller id'. 
        
        timestamps: generated timestamps for which we need to select market 
        participants to transact 
    
    Returns: 
        new_transactions: a dataframe with the same structure as raw_data_aggregated,
        however with the pair of transactors at each timestamp randomly selected from the
        overall transactor population. 
    
    """
    new_transactions = pd.DataFrame(columns= raw_data_aggregated.columns)
    whole_graph = nx.from_pandas_edgelist(raw_data_aggregated, source='seller id', target='buyer id', create_using = nx.MultiDiGraph)
    edge_ids = list(itertools.product(list(whole_graph.nodes()), list(whole_graph.nodes())))
    for ts in sorted(timestamps):
        change_edges = random.choices([str(i) for i in set(edge_ids)],
                weights=None, k=1)
        row = pd.Series(ts, index=change_edges)
        row = pd.DataFrame(row)
        row['total_value'] = np.random.randint(0,1000)#we could extend the methods to control the edge weight
        row.reset_index(inplace=True)
        row.columns = ['variable', 'trade date time', 'total_value']
        row.variable=row.variable.apply(lambda x: ast.literal_eval(str(x)))
        row['buyer id'] = [i for (i,j) in row['variable']]
        row['seller id'] = [j for (i,j) in row['variable']]
        new_transactions=new_transactions.append(row)
    new_transactions['tdt_rounded'] = myround(new_transactions['trade date time'], 20)
    #rebase timestamps to 0
    new_transactions['trade date time']-=new_transactions['trade date time'].min()
    return(new_transactions)

def freq_edge_selection(raw_data_aggregated, timestamps):
    """ Function to select a pair of market participants to transact at each point in
    a given transaction sequence, based on their historical transaction frequency.
    
    Parameters: 
        raw_data_aggregated: a dataframe with columns 'variable', 
        'trade date time', 'total_value', 'buyer id', 'seller id'. 
        timestamps - generated timestamps for which we need to select market 
        participants to transact 
    Returns: 
        freq_transactions: a dataframe with the same structure as raw_data_aggregated,
        however with the pair of transactors at each timestamp selected from the
        overall transactor population based on their frequency of transactions. 
    """
    freq_transactions = pd.DataFrame(columns= raw_data_aggregated.columns)
    edge_ids = list(zip(raw_data_aggregated['seller id'], raw_data_aggregated['buyer id']))
    raw_data_aggregated['variable'] = edge_ids
    new_data = raw_data_aggregated.copy()
    edge_tf = raw_data_aggregated.groupby('variable')['trade date time'].nunique()
    edge_tf = (edge_tf)/(edge_tf.sum())
    edge_tf.hist()
    plt.show()
    for ts in sorted(timestamps):
        change_edges = random.choices([str(i) for i in edge_tf.index],
                weights=[i for i in edge_tf.values], k=1)
        row = pd.Series(ts, index=change_edges)
        row = pd.DataFrame(row)
        row['total_value'] = np.random.randint(0,1000)
        row.reset_index(inplace=True)
        row.columns = ['variable', 'trade date time', 'total_value']
        row.variable=row.variable.apply(lambda x: ast.literal_eval(str(x)))
        row['buyer id'] = [i for (i,j) in row['variable']]
        row['seller id'] = [j for (i,j) in row['variable']]
        #timestamps need to be generated according to same format as input data.
        #We may also need some other process to control the actual value. 
        new_data = new_data.append(row)
        freq_transactions=freq_transactions.append(row)
        edge_tf = new_data.groupby('variable')['trade date time'].nunique()
        edge_tf = (edge_tf)/(edge_tf.sum())
    #rebase timestamps to 0
    freq_transactions['trade date time']-=freq_transactions['trade date time'].min()
    return(freq_transactions)

def imp_edge_selection(raw_data_aggregated, timestamps):
    """ 
    Function to select a pair of market participants to transact at each point in
    a given transaction sequence, based on their importance as defined in Seabrook 
    et. al. Structural importance and evolution: an application to financial transaction 
    networks. 
    
    Parameters: 
        raw_data_aggregated: a dataframe with columns 'variable', 
        'trade date time', 'total_value', 'buyer id', 'seller id'. 
        timestamps - generated timestamps for which we need to select market 
        participants to transact 
    Returns: 
        imp_transactions: a dataframe with the same structure as raw_data_aggregated,
        however with the pair of transactors at each timestamp selected from the
        overall transactor population based on their importance. 
    """
    imp_transactions = pd.DataFrame(columns= raw_data_aggregated.columns)
    ts_df = raw_data_aggregated.copy()
    G =  nx.from_pandas_edgelist(ts_df, source='seller id', target='buyer id', create_using = nx.Graph())
    print(G.number_of_nodes())
    print(G.number_of_edges())
    node_imps = pd.Series([1/mi_generator_symm_tests(ts_df, node, test_type=None)\
                     for node in G.nodes()] , index=G.nodes())
    node_imps = abs(np.log(abs(node_imps)))
    print(node_imps)
    edge_imps = pd.Series([1/(x*y) for x in node_imps.values for y in node_imps.values],\
                                    index = [(x, y) for x in node_imps.index for y in node_imps.index])
    i=0
    for ts in sorted(timestamps):
        change_edges = random.choices([str(i) for i in edge_imps.index],
                weights=[i for i in edge_imps.values], k=1)
        row = pd.Series(ts, index=change_edges)
        row = pd.DataFrame(row)
        row['total_value'] = np.random.randint(0,1000)
        row.reset_index(inplace=True)
        row.columns = ['variable', 'trade date time', 'total_value']
        row.variable=row.variable.apply(lambda x: ast.literal_eval(str(x)))
        row['buyer id'] = [i for (i,j) in row['variable']]
        row['seller id'] = [j for (i,j) in row['variable']]
        #timestamps need to be generated according to same format as input data.
        #We may also need some other process to control the actual value. 
        imp_transactions=imp_transactions.append(row)
        i+=1
        #update the node importance every 10 transactions
        if i>=10:
            new_trans= ts_df.append(imp_transactions)
            G =  nx.from_pandas_edgelist(new_trans, source='seller id', target='buyer id', create_using = nx.Graph())
            node_imps = pd.Series([1/mi_generator_symm_tests(new_trans, node, test_type=None)\
                     for node in G.nodes()] , index=G.nodes())
            node_imps = abs(np.log(abs(node_imps)))
            edge_imps = pd.Series([1/(x*y) for x in node_imps.values for y in node_imps.values],\
                                    index = [(x, y) for x in node_imps.index for y in node_imps.index])
            i=0
    #rebase timestamps to 0
    imp_transactions['trade date time']-=imp_transactions['trade date time'].min()
    return(imp_transactions)


    
def temporal_assortativity(data, n, c, title, axs):
    """
    Function to plot the assortativity in a rolling window across time. This is done
    by constructing static networks in rolling windows and computing the degree
    pearson correlation coefficient of these networks. 
    
    Parameters: 
        data: dataset of transactions including columns 'trade date time', 'seller id',
        'buyer id'. 
        n: size of the rolling window
        c: colour for the line plot
        title: title of the plot
        axs: axis for the plot.
        
    Returns: 
        plot of assortativity across time
    """
    data.sort_values('trade date time', inplace=True)
    rol = data['trade date time'].rolling(window=n)
    def ass_rolling(ser):
        G_raw = nx.from_pandas_edgelist(data.loc[ser.index], source='seller id', target='buyer id', create_using = nx.Graph)
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        G_raw.remove_nodes_from(list(nx.isolates(G_raw)))
        assortativity = nx.degree_pearson_correlation_coefficient(G_raw)
        return assortativity
    rolling_node_ass = rol.apply(ass_rolling, raw=False).dropna()  
    print(len(rolling_node_ass))
    axs.plot(range(len(rolling_node_ass)), rolling_node_ass.values, color=c, label = title )
    axs.set_xlabel("Time")
    axs.set_ylabel("Assortativity")
    
def assortativity_static(data_aggregated):
    """
    Function to calculate the assortativity of the fully temporally aggregated 
    transaction network. 
    
    Parameters: 
        data_aggregated: dataset to build the transaction network from, with columns
        'seller id', 'buyer id', and 'total_value'
        
    Returns: 
        assortativity: assortativity of the full network.
    """
    G_raw = nx.from_pandas_edgelist(data_aggregated, source='seller id', target='buyer id', edge_attr='total_value', create_using = nx.MultiGraph)
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    G_raw.remove_nodes_from(list(nx.isolates(G_raw)))
    assortativity = nx.degree_pearson_correlation_coefficient(G_raw)
    return(assortativity)

def temporal_reciprocity(data, n, c, title, axs):
    """
    Function to plot the assortativity in a rolling window across time. This is done
    by constructing static networks in rolling windows and computing the overall 
    reciprocity of these networks. 
    
    Parameters: 
        data: dataset of transactions including columns 'trade date time', 'seller id',
        'buyer id'. 
        n: size of the rolling window
        c: colour for the line plot
        title: title of the plot
        axs: axis for the plot.
        
    Returns: 
        plot of reciprocity across time
    """
    data.sort_values('trade date time', inplace=True)
    rol = data['trade date time'].rolling(window=n)
    def rec_rolling(ser):
        G_raw = nx.from_pandas_edgelist(data.loc[ser.index], source='seller id', target='buyer id', create_using = nx.DiGraph)
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        reciprocity = nx.overall_reciprocity(G_raw)
        return reciprocity
    rolling_node_imp = rol.apply(rec_rolling, raw=False).dropna()  
    print(len(rolling_node_imp))
    axs.plot(range(len(rolling_node_imp)), rolling_node_imp.values, color=c, label = title )
    axs.set_xlabel("Time")
    axs.set_ylabel("Reciprocity")
    
def reciprocity_static(data_aggregated):
    """
    Function to calculate the reciprocity of the fully temporally aggregated 
    transaction network. 
    
    Parameters: 
        data_aggregated: dataset to build the transaction network from, with columns
        'seller id', 'buyer id', and 'total_value'
        
    Returns: 
        reciprocity: reciprocity of the full network.
    """
    G_raw = nx.from_pandas_edgelist(data_aggregated, source='seller id', target='buyer id', create_using = nx.MultiDiGraph)
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    assortativity = nx.overall_reciprocity(G_raw)
    return(assortativity)

def plot_degree_histogram(data_aggregated, ax, color='b', symbol='+', normalized=True):
    """
    Function to plot the number of nodes at each degree of a transaction network
    as a scatterplot, on a log-log scale. 
    
    Parameters: 
        data_aggregated: dataset to build the transaction network from, with columns
        'seller id', 'buyer id', and 'total_value'
        ax: plot axis object
        color: colour for the plot
        symbol: symbol for the points
        normalized: boolean to indicate whether or not the plot should be normalised.
    """
    g = nx.from_pandas_edgelist(data_aggregated, source='seller id', target='buyer id', create_using = nx.MultiDiGraph)

    aux_y = nx.degree_histogram(g)
    
    aux_x = np.arange(0,len(aux_y)).tolist()
    
    n_nodes = g.number_of_nodes()
    
    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes
    ax.set_ylabel('Number of Nodes (log scale)')
    ax.set_xlabel('Degree (log scale)')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(aux_x, aux_y, symbol, color=color)
    return aux_x, aux_y

def plot_rich_club(raw_data_aggregated, axs):
    """
    Function to produce a scatter of the rich club coefficients at each value of degree
    for network snapshots of 20 minute intervals.
    
    Parameters: 
        raw_data_aggregated: dataset to build the transaction network from, with columns
        'seller id', 'buyer id', and 'total_value'
        axs: plot axis object
    
    
    Returns: scatter plot of rich club coefficients
    """
    raw_data_aggregated['tdt_rounded'] = myround(raw_data_aggregated['trade date time'], 20)
    for t in sorted(raw_data_aggregated.tdt_rounded.unique()):
        G_raw = nx.from_pandas_edgelist(raw_data_aggregated[raw_data_aggregated.tdt_rounded==t], source='seller id', target='buyer id')
        G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
        rcc = nx.rich_club_coefficient(G_raw,normalized=False)
        x = rcc.keys()
        y = rcc.values()
        axs.scatter(x,y, color='k', marker='.')


if __name__ == "__main__":
    
    path_to_raw_data = ''
    path_to_save_data = ''
    raw_data= pd.read_csv(path_to_raw_data, index_col=0)
    raw_data= raw_data[raw_data['venue id']=='XOFF']
    raw_data.sort_values(by='trade date time', inplace=True)
    transactions = raw_data[:len(raw_data)/2]
    timestamps = sorted(raw_data['trade date time'][len(raw_data)/2:])
    #produce datasets using the earliest half of transactions to calculate 
    #the transaction freqency or edge importance, and the second half as the 
    #timestamps to assign edges to.
    rand_gen = random_edge_selection(len(raw_data)/2, timestamps)
    rand_gen.to_csv(path_to_save_data+'random_gen.csv')
    freq_gen = freq_edge_selection(len(raw_data)/2, timestamps)
    freq_gen.to_csv(path_to_save_data+'freq_gen.csv')
    imp_gen = imp_edge_selection(len(raw_data)/2, timestamps)
    imp_gen.to_csv(path_to_save_data+'imp_gen.csv')
    
    #plot rich club coefficients
    columns = 3
    rows = 2
    fig3, axs3 = plt.subplots(rows, columns, figsize=(12,7))
    axs3=axs3.flatten()
    plot_rich_club(rand_gen, freq_gen, imp_gen, raw_data, axs=axs3)
    plt.tight_layout()
    plt.show()
    
    #calculate rich clubs and run KS tests to see if the distributions are the 
    #same as the real data
    rand_res = []
    freq_res = []
    imp_res = []
    raw_data['tdt_rounded'] = myround(raw_data['trade date time'], 20)
    rand_gen['tdt_rounded'] = myround(rand_gen['trade date time'], 20)
    freq_gen['tdt_rounded'] = myround(freq_gen['trade date time'], 20)
    imp_gen['tdt_rounded'] = myround(imp_gen['trade date time'], 20)

    x_raw = []
    y_raw = []
    for t in sorted(raw_data.tdt_rounded.unique()):
        G = nx.from_pandas_edgelist(raw_data[raw_data.tdt_rounded==t], source='seller id', target='buyer id')
        G.remove_edges_from(nx.selfloop_edges(G))
        rcc = nx.rich_club_coefficient(G, normalized=False)
        x = rcc.keys()
        y = rcc.values()
        x_raw.extend(x)
        y_raw.extend(y)
    x_rand = []
    y_rand = []
    for t in sorted(rand_gen.tdt_rounded.unique()):
        G = nx.from_pandas_edgelist(rand_gen[rand_gen.tdt_rounded==t], source='seller id', target='buyer id')
        G.remove_edges_from(nx.selfloop_edges(G))
        rcc = nx.rich_club_coefficient(G, normalized=False)
        x = rcc.keys()
        y = rcc.values()
        x_rand.extend(x)
        y_rand.extend(y)
    x_freq = []
    y_freq = []
    for t in sorted(freq_gen.tdt_rounded.unique()):
        G = nx.from_pandas_edgelist(freq_gen[freq_gen.tdt_rounded==t], source='seller id', target='buyer id')
        G.remove_edges_from(nx.selfloop_edges(G))
        rcc = nx.rich_club_coefficient(G, normalized=False)
        x = rcc.keys()
        y = rcc.values()
        x_freq.extend(x)
        y_freq.extend(y)
    x_imp = []
    y_imp = []
    for t in sorted(imp_gen.tdt_rounded.unique()):
        G = nx.from_pandas_edgelist(imp_gen[imp_gen.tdt_rounded==t], source='seller id', target='buyer id')
        G.remove_edges_from(nx.selfloop_edges(G))
        rcc = nx.rich_club_coefficient(G, normalized=False)
        x = rcc.keys()
        y = rcc.values()
        x_imp.extend(x)
        y_imp.extend(y)
    rand_ks = ks2d2s(np.array(x_rand), np.array(y_rand), np.array(x_raw), np.array(y_raw), nboot=None, extra=False)
    freq_ks = ks2d2s(np.array(x_freq), np.array(y_freq), np.array(x_raw), np.array(y_raw), nboot=None, extra=False)
    imp_ks = ks2d2s(np.array(x_imp), np.array(y_imp), np.array(x_raw), np.array(y_raw), nboot=None, extra=False)
    rand_res.append(rand_ks)
    freq_res.append(freq_ks)
    imp_res.append(imp_ks)

print(rand_res)
print(freq_res)
print(imp_res)
#small p-values mean that the distributions are significantly different. 

#Degree distribution
fig, ax=plt.figure()
plot_degree_histogram(rand_gen, ax, color='b', symbol='+', normalized=True)
plot_degree_histogram(rand_gen, ax, color='#1f77b4', symbol='+', normalized=True)
plot_degree_histogram(rand_gen, ax, color='k', symbol='+', normalized=True)
plot_degree_histogram(rand_gen, ax, color='r', symbol='+', normalized=True)
plt.show()
#Assortativity and reciprocity plots
fig1, ax1=plt.figure()
temporal_reciprocity(rand_gen, int(len(rand_gen)/10), 'b', 'random edge selection', ax1)
temporal_reciprocity(freq_gen, int(len(freq_gen)/10),'#1f77b4', 'frequency based edge selection', ax1)
temporal_reciprocity(imp_gen, int(len(imp_gen)/10),'k', 'importance based edge selection',ax1)
temporal_reciprocity(raw_data, int(len(raw_data)/10),'r', 'real data',ax1)
plt.show()

fig2, ax2 = plt.figure()
temporal_assortativity(rand_gen, int(len(rand_gen)/10), 'b', 'random edge selection',ax2)
temporal_assortativity(freq_gen, int(len(freq_gen)/10),'#1f77b4', 'frequency based edge selection',ax2)
temporal_assortativity(imp_gen, int(len(imp_gen)/10),'k', 'importance based edge selection',ax2)
temporal_assortativity(raw_data, int(len(raw_data)/10),'r', 'real data',ax2)
plt.show()