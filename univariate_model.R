# -*- coding: utf-8 -*-
# Created on Wed Mar  9 16:59:46 2022

# @author: iseabrook1
#This script contains the code used in Seabrook et. al., Modelling equity 
#transactions as bursty processes. 
#Specifically, this script contains the code for estimating the parameters of a 
#univariate Hawkes process, simulating using the estimated parameters to obtain 
#the burstiness of the estimated process and the confidence intervals for this.
#Please note that methods to select which edge changes at each point in the process
#are contained within the python script 'univariate_edge_selection.py'. 

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Modelling equity 
#transactions as bursty processes. 
#
##############################################################################
# Instructions for use: This script fits a univariate Hawkes process to a sequence of timestamps.
# The script is designed to run on a csv located at path_to_data, which has columns venue.id and 
# trade.date.time, with the latter containing the timestamps to fit the process to.
# the user defined parameter data_subset allows to subset the data on a particular segment of the market.
######### Libraries & functions ####################

library(maxLik)
library(plyr)
library(hawkes)
library(poisson)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}


burstiness<- function(timestamps){
  inter_trade_times <- diff(timestamps[!is.na(timestamps)])
  sigma <- sd(inter_trade_times)
  mu <- mean(inter_trade_times)
  B <- (sigma-mu)/(sigma+mu)
  return(B)
}

tx_density <- function(timestamps){
  timestamps <- timestamps[!is.na(timestamps)]
  length<-length(timestamps)
  max_min<-max(timestamps)-min(timestamps)
  return(length/max_min)
}

#####################################
#create an empty list to store the results in
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
raw_timestamps <- real_data[,'trade.date.time']
timestamps <- sort(range01(raw_timestamps))
#likelihood function inline so that it is single parameterised but works with timestamps
nll <- function(param) {
  lambda_est<-param[1]
  alpha_est<-param[2]
  beta_est<-param[3]
  ll<- -hawkes::likelihoodHawkes(lambda_est, alpha_est, beta_est, timestamps[2:length(timestamps)-1])
  return(ll)
}
# create a constraint matrix to ensure that the alpha<beta for stability and 
# that all parameters are greater than 0. 
cons <- list(ineqA=matrix(c(0,1,0,-1,0,1,1,0,0),3,3), ineqB=c(0,0,0))

est <- maxLik(nll, start=c(0.1,0.2,0.3), control=list(iterlim=20000, sann_randomSeed=sample(100,1)), constraints=cons, method='SANN')
# for different datasets different start parmeters will be needed. when the above fails,
# this section of code iterates through to find valid starting parameters. 
n<-0.0001
start_n <- c(-1,-1,-1)
while(est$message=='Initial value out of range.'){
  print('here')
  start_n <- start_n+n
  print(start_n)
  est <- maxLik(nll, start=start_n, control=list(iterlim=20000), constraints=cons, method='SANN')
}

print(summary(est))

#use the estimated parameters to simulate 1000 Hawkes processes, to allow us to observe
#the confidence intervals for the burstiness
for (i in 1:1001) {
  sim_timestamps <- hawkes::simulateHawkes(est$estimate[1], est$estimate[2], est$estimate[3], horizon = length(timestamps))
  sim_timestamps<-sim_timestamps[[1]]
  tx_sim_list[[i]] <- sim_timestamps
}


tx_sim_df <-plyr::ldply( tx_sim_list, rbind)


sim_burstinesses <- apply(tx_sim_df, 1, burstiness)
print('hawkes burstinesses (and CI)')
print(mean(sim_burstinesses))
hawkes_burstiness_mean <- append(hawkes_burstiness_mean, mean(sim_burstinesses))
print('lower')
print(sort(sim_burstinesses)[25])
hawkes_burstiness_mean <- append(hawkes_burstiness_mean, sort(sim_burstinesses)[25])
print('upper')
print(sort(sim_burstinesses)[975])
hawkes_burstiness_mean <- append(hawkes_burstiness_mean, sort(sim_burstinesses)[975])
print(burstiness(sort(unlist(timestamps))))


#comparison to Poisson

#mle of the poisson intensity is the sample mean of the inter-trade time
lambda_poiss <- mean(diff(timestamps[!is.na(timestamps)]))

scen = hpp.scenario(rate = lambda_poiss, num.events = length(timestamps), num.sims = 1000)


burstinesses = apply(scen@x, 2, burstiness)
print('poisson burstiness (and CI)')
print(mean(burstinesses))
poisson_burstiness_mean <- append(poisson_burstiness_mean, mean(burstinesses))
print(sort(burstinesses)[25])
poisson_burstiness_low <- append(poisson_burstiness_low, sort(burstinesses)[25])
print(sort(burstinesses)[975])
poisson_burstiness_high <- append(poisson_burstiness_high, sort(burstinesses)[975])

tx_densities = apply(scen@x, 2, tx_density)