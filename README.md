# bursty-equities
Code used in Seabrook et. al., Modelling equity transactions as bursty processes. 

This repo contains the following scripts, and is written in a mixture of Python and R. 
1. data_exploration.py: Contains the code to produce the exploratory data analysis presented in the paper.
2. univariate_model.R: Contains the code to fit and simulate a univariate Hawkes process given a transaction sequence.
3. univariate_edge_selection.py: Contains the code to select counterparties to transact at each point in a transaction sequence
4. edge_univariate_model.R: Contains the code to fit multiple univariate Hawkes processes to transaction sequences of individual counterparty pairs.
5. multivariate_model.py: Contains the code to fit a multivariate Hawkes process to transaction sequences of counterparty pairs with more than 10 transactions between them.
6. bivariate_model.R: Contains the code to fit a bivariate Hawkes process to the buys and sells for transactions through a major counterparty.
7. additional_results.py: Contains code to plot parameter values, assortativity for network rewiring, and burstiness and density averaged across multiple days. 

All scripts require the user to edit parameters to point towards their own data, and have not been tested on data other than the transaction reports for FTSE100 instruments considered in the above paper, which are not available for publication.

