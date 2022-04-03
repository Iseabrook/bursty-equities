# -*- coding: utf-8 -*-
# Created on Wed Mar  9 16:59:46 2022

# @author: iseabrook1
#This script contains the code used in Seabrook et. al., Modelling equity 
#transactions as bursty processes. 
#Specifically, this script contains the code for estimating the parameters of a 
#bivariate Hawkes process, simulating using the estimated parameters to obtain 
#the burstiness of the estimated process and the confidence intervals for this.

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Modelling equity 
#transactions as bursty processes. 
#
##############################################################################
# Instructions for use: This script fits a bivariate Hawkes process to a two lists of timestamps - one 
# for buy transactions and one for sell transactions.
# The script is designed to run on a csv located at path_to_data, which has columns buyer.id, seller.id
# trade.date.time, with the latter containing the timestamps to fit the process to.
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



###########################Part 1 - bivariate model #################
#populate the path_to_data and data_subset 
path_to_data = 'path_to_data'

real_data <- read.csv(path_to_data)

real_data <- real_data[order(real_data$trade.date.time),]
###STEP 1 - get list of lists of buyer timestamps and seller timestamps
timestamps <- c()

timestamps['buys'] <- as.list(subset(real_data, buyer.id == 'CCP_ID', select = c("trade.date.time")))
timestamps['sells'] <- as.list(subset(real_data, seller.id == 'CCP_ID', select = c("trade.date.time")))

#create a list of start parameters with two baseline values, four alpha values and two beta values.
strt <- c(rep(0.1, length(timestamps)))
strt <- append(strt, c(rep(0.2, length(timestamps)**2)))
strt <- append(strt, c(rep(0.3, length(timestamps))))
#likelihood function inline so that it is single parameterised but works with timestamps

nll_bi <- function(param) {
  lambda_est<-param[1:(length(timestamps))]
  #alpha needs to be a matrix
  alpha_est <- param[(length(timestamps)+1):(length(timestamps)+(length(timestamps)**2))]
  alpha_est <- matrix(alpha_est,byrow=TRUE,nrow=length(lambda_est))
  beta_est<-param[(1+length(timestamps)+(length(timestamps)**2)):length(param)]
  ll<- -hawkes::likelihoodHawkes(lambda_est, alpha_est, beta_est, timestamps)
  return(ll)
}
# create a constraint matrix to ensure that the alpha<beta for stability and 
# that all parameter are greater than 0. 
cons <- list(ineqA=matrix(c(1,0,0,0,0,0,0,0,
                            0,1,0,0,0,0,0,0,
                            1,0,1,0,0,0,0,0,
                            0,0,0,1,0,0,0,0,
                            0,0,0,0,1,0,0,0,
                            0,0,0,0,0,1,0,0,
                            0,0,0,0,0,0,1,0,
                            0,0,0,0,0,0,0,1
),8,8), ineqB=c(0,0,0,0,0,0,0,0))
est <- maxLik(nll_bi, cons=cons, start=strt)
print(summary(est))
baseline <- c(est$estimate[1], est$estimate[2])
kern_int <- matrix(c(est$estimate[3],est$estimate[4],est$estimate[5],est$estimate[6]),byrow=TRUE,nrow=2)
kern_dec <- c(est$estimate[7]+0.01, est$estimate[8])

###STEP 3 - use the vector of hawkes parameters for each edge to simluate process
###and look at the burstiness etc. 
###here we do this 1000 times to bootstrap and get confidence intervals
tx_boot_list <- list()
tx_boot_buys <- list()
tx_boot_sells <- list()
for (k in 1:1001) {
  tx_sim_list <- list()
  sim_timestamps = hawkes::simulateHawkes(baseline, kern_int, kern_dec, horizon = 300)
  tx_boot_list[[k]] <- sort(unlist(sim_timestamps))
  tx_boot_buys[[k]] <- sim_timestamps[[1]]
  tx_boot_sells[[k]] <- sim_timestamps[[2]]
}
tx_sim_df <-plyr::ldply( tx_boot_list, rbind)
tx_sim_df_buys <-plyr::ldply( tx_boot_buys, rbind)
tx_sim_df_sells <-plyr::ldply( tx_boot_sells, rbind)

sim_burstinesses_buys <- apply(tx_sim_df_buys, 1, burstiness)
sim_burstinesses_sells <- apply(tx_sim_df_sells, 1, burstiness)
sim_burstinesses <- apply(tx_sim_df, 1, burstiness)

print('hawkes burstinesses (and CI)')
print(mean(sim_burstinesses[!is.na(sim_burstinesses)]))

print(sort(sim_burstinesses[!is.na(sim_burstinesses)])[25])

print(sort(sim_burstinesses[!is.na(sim_burstinesses)])[975])

print('hawkes burstinesses buys (and CI)')
print(mean(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)]))

print(sort(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)])[25])

print(tail(sort(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)]), 25)[1])

print('hawkes burstinesses sells (and CI)')
print(mean(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)]))

print(sort(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)])[25])

print(tail(sort(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)]), 25)[1])
print('real burstinesses')
print(burstiness(sort(unlist(timestamps))))
print(burstiness(sort(timestamps[[1]])))
print(burstiness(sort(timestamps[[2]])))



########################## Part 2 - Univariate buy/sell model #############
timestamps_lst <- c()

timestamps_lst['buys'] <- as.list(subset(real_data, buyer.id == 'F226TOH6YD6XJB17KS62', select = c("trade.date.time")))
timestamps_lst['sells'] <- as.list(subset(real_data, seller.id == 'F226TOH6YD6XJB17KS62', select = c("trade.date.time")))

baselines <- c()
kern_ints <- c()
kern_decs <- c()
hawkes_timestamps <- c()
for(timestamps in timestamps_lst) {
  # create a constraint matrix to ensure that the alpha<beta for stability and 
  # that all parameter are greater than 0.
  #likelihood function inline so that it is single parameterised but works with timestamps
  nll_uni <- function(param) {
    lambda_est<-param[1]
    alpha_est<-param[2]
    beta_est<-param[3]
    ll<- -hawkes::likelihoodHawkes(lambda_est, alpha_est, beta_est, timestamps)
    return(ll)
  }
  cons <- list(ineqA=matrix(c(0,1,0,-1,0,1,1,0,0),3,3), ineqB=c(0,0,0))
  est <- maxLik(nll_uni, start=c(0.1,0.02,0.03), control=list(iterlim=200000), constraints=cons, method='SANN')
  print(summary(est))
  baseline <- est$estimate[1]
  kern_int <- est$estimate[2]
  kern_dec <- est$estimate[3]
  baselines <- append(baselines, baseline)
  kern_ints <- append(kern_ints, kern_int)
  kern_decs <- append(kern_decs, kern_dec)
  hawkes_timestamps <- append(hawkes_timestamps, timestamps)
  hawkes_timestamps <- sort(unlist(hawkes_timestamps))
} 

###use the vector of hawkes parameters for each edge to simluate process
###and look at the burstiness etc. 
###here we do this 1000 times to bootstrap and get confidence intervals
tx_boot_list <- list()
tx_boot_buys <- list()
tx_boot_sells <- list()
sim_timestamps <- list()
for (k in 1:1001) {
  
  tx_sim_list <- list()
  sim_timestamps_buys <- hawkes::simulateHawkes(baselines[1], kern_ints[1], kern_decs[1], horizon = 300)
  sim_timestamps_sells <- hawkes::simulateHawkes(baselines[2], kern_ints[2], kern_decs[2], horizon = 300)
  sim_timestamps['buys'] <- sim_timestamps_buys
  sim_timestamps['sells'] <- sim_timestamps_sells
  tx_boot_list[[k]] <- sort(unlist(sim_timestamps))
  tx_boot_buys[[k]] <- sim_timestamps_buys[[1]]
  tx_boot_sells[[k]] <- sim_timestamps_sells[[1]]
}
tx_sim_df <-plyr::ldply( tx_boot_list, rbind)
tx_sim_df_buys <-plyr::ldply( tx_boot_buys, rbind)
tx_sim_df_sells <-plyr::ldply( tx_boot_sells, rbind)

sim_burstinesses_buys <- apply(tx_sim_df_buys, 1, burstiness)
sim_burstinesses_sells <- apply(tx_sim_df_sells, 1, burstiness)
sim_burstinesses <- apply(tx_sim_df, 1, burstiness)

print('hawkes burstinesses (and CI)')
print(mean(sim_burstinesses[!is.na(sim_burstinesses)]))

print(sort(sim_burstinesses[!is.na(sim_burstinesses)])[25])

print(sort(sim_burstinesses[!is.na(sim_burstinesses)])[975])

print('hawkes burstinesses buys (and CI)')
print(mean(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)]))

print(sort(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)])[25])

print(tail(sort(sim_burstinesses_buys[!is.na(sim_burstinesses_buys)]), 25)[1])

print('hawkes burstinesses sells (and CI)')
print(mean(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)]))

print(sort(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)])[25])

print(tail(sort(sim_burstinesses_sells[!is.na(sim_burstinesses_sells)]), 25)[1])
print('real burstinesses')
print(burstiness(sort(unlist(timestamps))))
print(burstiness(sort(timestamps[[1]])))
print(burstiness(sort(timestamps[[2]])))