# -*- coding: utf-8 -*-
# Created on Wed Mar  9 16:59:46 2022

# @author: iseabrook1
#This script contains the code used in Seabrook et. al., Modelling equity 
#transactions as bursty processes. 
#Specifically, this script contains the code for estimating the parameters of a 
#model which uses multiple univariate Hawkes process, having an independent process 
#for each pair of participants, for participants who transact more than 10 times.
#This is followed with simulating using the estimated parameters to obtain 
#the burstiness of the estimated process and the confidence intervals for this.
 

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Modelling equity 
#transactions as bursty processes. 
#
##############################################################################
# Instructions for use: This script fits univariate Hawkes processes to multiple sequences of timestamps.
# The script is designed to run on a csv located at path_to_data, which has columns venue.id and 
# trade.date.time, with the latter containing the timestamps to fit the process to.
# the user defined parameter data_subset allows to subset the data on a particular segment of the market.
######### Libraries & functions ####################

library(plyr)
library(reshape)
library(maxLik)
burstiness<- function(timestamps){
  inter_trade_times <- diff(timestamps[!is.na(timestamps)])
  sigma <- sd(inter_trade_times)
  mu <- mean(inter_trade_times)
  B <- (sigma-mu)/(sigma+mu)
  return(B)
}

hawkes_burstiness_mean = c()
hawkes_burstiness_low = c()
hawkes_burstiness_high = c()
tx_sim_list <- list()

#populate the path_to_data and data_subset 
path_to_data = 'path_to_data'
data_subset <- 'INSERT SUBSET HERE'

real_data_raw <- read.csv(path_to_data)
if(data_subset == 'xlon'){
  real_data <- subset(real_data_raw, venue.id == 'XLON')
} else if(data_subset == 'xoff'){
  real_data <- subset(real_data_raw, venue.id == 'XOFF')
} else {
  real_data <- real_data_raw
}

real_data$edge <- paste(real_data$buyer.id,real_data$seller.id, sep = "")
real_data <- real_data[order(real_data$trade.date.time),]
###STEP 1 - get edge level transaction sequences
timestamps_df <- cast(real_data[c("edge","X", "trade.date.time")], X ~ edge)
timestamps_df[timestamps_df == 0] <- NA
timestamps_df$X <- NULL
timestamps_list <- lapply(timestamps_df, na.omit)
edges_list <- names(timestamps_list)
###STEP 2 - loop through the edge level transaction sequences, estimating the 
###Hawkes parameters for each edge. 
baselines <- c()
kern_ints <- c()
kern_decs <- c()
hawkes_edges <- c()
random_edges <- c()
random_edge_lengths <- c()
random_edge_timestamps <- c()
hawkes_timestamps <- c()
###We cannot fit a Hawkes process to the edges that have very few (or only one) 
###transaction. 247 of the 347 edges only appear once. 
  for(edge in edges_list) {
    timestamps <-  sort(unique(get(edge, timestamps_list)))
    
    if(length(timestamps)>10){
      #likelihood function inline so that it is single parameterised but works with timestamps
      nll <- function(param) {
        lambda_est<-param[1]
        alpha_est<-param[2]
        beta_est<-param[3]
        ll<- -hawkes::likelihoodHawkes(lambda_est, alpha_est, beta_est, timestamps)
        return(ll)
      }
      # create a constraint matrix to ensure that the alpha<beta for stability and 
      # that all parameter are greater than 0.
      cons <- list(ineqA=matrix(c(0,1,0,-1,0,1,1,0,0),3,3), ineqB=c(0,0,0))
      est <- maxLik(nll, start=c(0.1,0.02,0.03), control=list(iterlim=200000), constraints=cons, method='SANN')
      baseline <- est$estimate[1]
      kern_int <- est$estimate[2]
      kern_dec <- est$estimate[3]
      baselines <- append(baselines, baseline)
      kern_ints <- append(kern_ints, kern_int)
      kern_decs <- append(kern_decs, kern_dec)
      hawkes_edges <- append(hawkes_edges, edge)
      hawkes_timestamps <- append(hawkes_timestamps, timestamps)
      hawkes_timestamps <- sort(unlist(hawkes_timestamps))
    } 
    else {
      random_edges <- append(random_edges, edge)
      random_edge_lengths <- append(random_edge_lengths, length(timestamps))
      random_edge_timestamps <- append(random_edge_timestamps, timestamps)
      random_edge_timestamps <- sort(unlist(random_edge_timestamps))
    }    
  }
  
  ###STEP 3 - use the vector of hawkes parameters for each edge to simluate process
  ###and look at the burstiness etc. 
  ###here we do this 1000 times to bootstrap and get confidence intervals
  tx_boot_list <- list()
  for (k in 1:1001) {
    
    tx_sim_list <- list()
    for (i in c(1:length(hawkes_edges))){
      edge <- hawkes_edges[i]
      timestamps <-  sort(unique(get(edge, timestamps_list)))
      baseline <- baselines[i]
      kern_int <- kern_ints[i]
      kern_dec <- kern_decs[i]
      sim_timestamps = hawkes::simulateHawkes(baseline, kern_int, kern_dec, horizon = length(timestamps))
      sim_timestamps<-sim_timestamps[[1]]
      tx_sim_list[[edge]] <- sort(unique(unlist(sim_timestamps)))
    }
    
    tx_boot_list[[k]] <- sort(unique(unlist(tx_sim_list)))
  }
  tx_sim_df <-plyr::ldply( tx_boot_list, rbind)
  
  sim_burstinesses = apply(tx_sim_df, 1, burstiness)
  print('hawkes burstinesses (and CI)')
  print(mean(sim_burstinesses))
  hawkes_burstiness_mean <- append(hawkes_burstiness_mean, mean(sim_burstinesses))
  
  print(sort(sim_burstinesses)[25])
  hawkes_burstiness_mean <- append(hawkes_burstiness_mean, sort(sim_burstinesses)[25])
  
  print(sort(sim_burstinesses)[975])
  hawkes_burstiness_mean <- append(hawkes_burstiness_mean, sort(sim_burstinesses)[975])
  
  
  print(burstiness(sort(unlist(tx_sim_list))))
  
  #compare to Poisson
  all_tx_lst <- list()
  tx_df_lst <- list()
  burst_all_lst <- list()
  
  for (i in 1:1001) {
    tx_sim_list <- list()
    
    real_tx_lst <- list()
    for(edge in edges_list) {
      timestamps <-  sort(unique(get(edge, timestamps_list)))
      
      #mle of the poisson intensity is the sample mean of the inter-trade time
      if(length(timestamps)>10){
        lambda_poiss <- mean(diff(timestamps[!is.na(timestamps)]))
        
        scen <- hpp.scenario(rate = lambda_poiss, num.events = length(timestamps),t0=min(timestamps),t1=max(timestamps),  num.sims = 2)
        tx_sim_list[[edge]] <- scen@x[,1] 
        real_tx_lst[[edge]] <- timestamps
      }
    }
    real_tx_df <- plyr::ldply(real_tx_lst, rbind)
    tx_df <- plyr::ldply(tx_sim_list, rbind)
    tx_df_lst[[i]] <- tx_df
    all_tx <- melt(tx_df)
    all_tx_lst[[i]] <- all_tx
    burstiness_all <- burstiness(all_tx[,'value'])
    burstiness_all_real <- burstiness(melt(real_tx_df)[,'value'])
    burst_all_real_lst <- burstiness_all_real
    burst_all_lst[[i]] <- burstiness_all
  }
  

