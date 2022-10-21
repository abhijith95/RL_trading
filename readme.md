# Reinforcement learning for algorithmic trading
The module inputs an excel file containing information like opening,
closing, adjusted closing, high and low prices and the volume of 
trades per day. The information is used as input to the networks
and the output will be the proportions of the assets. This program
uses actor-critic algorithm to get the optimal trading strategy

## Data
The excel data to be used in this problem shall be built by using
the dataBuilder module. Please note that each sheet in excel shall
contain the data for Open, High, Low, Close, Adj Close, Volume. 
Also kindly ensure that the naming of these sheets as they 
are case sensitive and have it the same as described above.

## States and actions
The states to the actor and critic networks are the OHCLV data
of individual stocks for the past twenty days. The actions from
the actor network will be the proportion of the asset at each
time step.

## Environment
The environment in this problem is the daily price movements of 
the assets. Each time step corresponds to moving over a single
day with asset proportions defined by the actor network, before
the start of the time step. Here we assume that we somehow buy
and sell the assets before the trading day begins.

## Reward
The reward for the actions is the change in the portfolio value,
which is calculated by multiplying the percent change with the
asset proportions and summing them all up. Note that in this
problem we assume a starting captial with currency denomination
of SEK. This is because popular brokers like Avanza, Aktieinvest
charges comission based on how much money is used up in each 
transaction order and there are different brackets for this with
different comission rates. Due to this setup we shall assume that
each asset shall be bough with maximum of X SEK and the proper 
comission rate shall be applied while calculating the portfolio
value. Note that the portfolio value will be sum of the value 
of assets and remaining cash. 

## The network
Both the actor and critic network will be made of sequentially
built dense NN layers. The output from the actor network will
be of shape (number_of_assets,1), whose activation will be a
softmax activation. The reason is in two parts, one the sum
of the output will add to 1, which is required because we don't
consider shorting any stocks in this problem. Secondly the value
of each entity in array will be in range [0,1], which potrays
the proportion of a stock in the portfolio.

## Training the networks
The data file contains every day price of 21 different stocks
(primarily from Swedish market), from 2011 Jan to 2022 Oct.
The training data will be from 2011 Jan to 2020 Dec, the test
data set will be from 2021 Jan till end of the data. There is 
no validation set to tune the hyperparameters in this case.