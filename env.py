import pandas as pd
import numpy as np
import tensorflow as tf

class env:
    
    def __init__(self, dataFile, sheetNames,trainingIndex,
                marketMemory):
        """Initialising the class
        ! Please note that the data file has to be built using the
        dataBuilder module and the file should have at least OPEN,
        HIGH, LOW, CLOSE and VOLUME information !
        
        Args:
        dataFile (string): location of the excel file
        sheetNames (list): list of sheet names holding the OHCLV data
        trainingIndex (list): list containing the starting and ending index
        for the training data set.
        marketMemory(int): number of previous observations required to predict
        the asset proportions.
        """
        self.dfs = {}
        self.train, self.test = {}, {}
        
        for sheet in sheetNames:
            self.dfs[sheet]=pd.read_excel(dataFile, sheet_name=sheet)
            self.train[sheet]=self.dfs[sheet].iloc[trainingIndex[0]:trainingIndex[1],0:-1]
            self.test[sheet]=self.dfs[sheet].iloc[trainingIndex[1]:,0:-1]
        
        self.marketMemory = marketMemory
        # -1 because columns include data as well
        self.noOfAssets = len(self.dfs["Open"].columns) - 1 
        self.noOfPriceFeatures = len(sheetNames)
        self.brokerFee = 39.0   # transaction fee for every buy and sell, in SEK
        self.maxAssetOrder = 21000.0  # maximum money spent in buying an asset, in SEK     
        self.totalInv = self.maxAssetOrder * self.noOfAssets  # total amount invested in assets
        self.reset()
    
    def reset(self):
        """Function that resets reward and the asset proportion
        before an epoch starts. Note that the variables assetProp
        and reward contains the value at each time step in an epoch.
        """
        # initially the asset proportion will be uniform and equal
        self.assetProp = np.ones(self.noOfAssets) * (1/self.noOfAssets)
        self.cash = np.copy(self.maxAssetOrder)     
        self.assetVal = self.assetProp * (self.totalInv)
        self.portVal = np.sum(self.assetVal)  
        self.penalty = 0  # if cash is depleted the agent gets a big penalty
        
    def getState(self, index):
        """Method that returns the current state of the environment
        Args:
            index (int): the index of the current time step
        Returns:
            state (tensor): the past marketMemory days of OHCLV value for
            all the assets. The shape of the tensor is (number_of_price_features,
            self.marketMemory, self.noOfAssets)
        """
        state = []
        for key in self.train:
            # going through the dictionary
            state.append(list(np.array(self.train[key].iloc[index - self.marketMemory:index,0:self.noOfAssets])))  
        
        state = tf.convert_to_tensor(state)
        state = tf.reshape(state, shape=[1,self.noOfPriceFeatures,self.marketMemory,self.noOfAssets])
        return state

    def transact(self,action,index):
        """Function that buys and sells assets. The cash used to buy
        and sell is credited/debited to the cash variable.
        Args:
        action (tensor of shape (self.noOfAssets, )): this is the asset 
            proportions to have before stepping through the trading day. All buy 
            and sell must be done before trading starts.
        index (int): the index of the current time step
        """
        # action = np.array(action)
        delta = np.abs(action - self.assetProp)*self.portVal
        noOfShares = delta/np.array(self.train["Close"].iloc[index,:])
        mask1 = np.where(noOfShares>1,1,0)
        mask2 = np.where(action>self.assetProp,1,-1)
        self.assetVal = self.assetVal+(mask1*mask2*delta)
        self.cash+=np.sum(mask1*-mask2*delta)
        # for i in range(len(action)):
        #     # amount of money to buy/sell
        #     delta = abs((action[i] - self.assetProp[i]) * self.portVal)
        #     noOfShares = delta/self.train["Close"].iloc[index,i]
        #     # execute transaction only if number of shares is greater than 1
        #     # else the broker will not execute the order
        #     if action[i] > self.assetProp[i]:
        #         # need to buy more shares                
        #         if noOfShares > 1:
        #             self.cash-=delta
        #             self.assetVal[i]+=delta
        #             self.assetProp[i] = action[i]                
        #     else:
        #         # need to sell shares
        #         if noOfShares > 1:
        #             self.cash+=delta
        #             self.assetVal[i]-=delta
        #             self.assetProp[i] = action[i] 
        
        self.portVal= np.sum(self.assetVal)  # this should keep the porfolio value unchanged.
        if self.cash < 0:
            # huge penalty if cash reserve is used up!
            self.penalty = -self.totalInv
    
    def stepDt(self, action, index):
        """
        Function that carries the epoch through one time step. In this case 
        the environment runs through one day of trading and calculates the gains.
        The gains are calculated using Adjusted Close value of the assets between
        two consecutive trading days.
        Args:
        action (numpy array of shape (self.noOfAssets, )): this is the asset 
        proportions to have before stepping through the trading day. All buy 
        and sell must be done before trading starts.
        index (int): the index of the current time step
        """
        # buying and selling assets
        self.transact(action,index)
        
        priceRatio = np.array(self.train["Close"].iloc[index+1,:]) / \
                        np.array(self.train["Close"].iloc[index,:])
        self.assetVal = self.assetVal * priceRatio
        self.portVal = np.sum(self.assetVal)
                
    def getReward(self):
        return (self.portVal+self.penalty+self.cash)
    
    def isDone(self):
        """If the cash reserve is used up the epoch will end.
        Returns:
            Boolean: signifies the end of an epoch
        """
        if self.cash < 0:
            return True
        return False
    
# e = env(dataFile="C:\\Users\\abhij\\RL_trading\\Daily_stocks_data\\portfolio.xlsx",
#         sheetNames=["Open", "High", "Low", "Close", "Volume"],
#         trainingIndex=[0,2259])
# e.getStates(30)