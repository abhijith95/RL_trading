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
        self.marketMemory = marketMemory
        self.trainingIndex = trainingIndex
        
        for sheet in sheetNames:
            # standardizing the testing and training data
            self.dfs[sheet]=pd.read_excel(dataFile, sheet_name=sheet)
            temp=self.dfs[sheet].iloc[trainingIndex[0]:trainingIndex[1],0:-1]
            self.train[sheet] = (temp-temp.mean())/temp.std()
            temp=self.dfs[sheet].iloc[trainingIndex[1]:,0:-1]
            self.test[sheet] = (temp-temp.mean())/temp.std()
        
        # -1 because columns include date as well
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
        # asset value is the actually amount of money invested in each asset
        self.assetVal = self.assetProp * (self.totalInv)
        self.portVal = np.sum(self.assetVal)  
        self.penalty = 0  # if cash is depleted the agent gets a big penalty
        
    def getState(self, index, train=True):
        """Method that returns the current state of the environment
        Args:
            index (int): the index of the current time step
            train (bool): whether the function is called during training or
            testing.
        Returns:
            state (tensor): the past marketMemory days of OHCLV value for
            all the assets. The shape of the tensor is (number_of_price_features,
            self.marketMemory, self.noOfAssets)
            asseProp (tensor): asset proportions. Shape of tensor is (1,number_of_assets)
            cash (float): remaining cash value in the portfolio
        """
        state = []
        for key in self.dfs:
            # going through the dictionary
            if train:
                state.append(list(np.array(self.train[key].iloc
                                        [index - self.marketMemory:index,:])))  
            else:
                state.append(list(np.array(self.test[key].iloc
                                        [index - self.marketMemory:index,:])))
        
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, shape=[1,self.noOfPriceFeatures,
                                        self.marketMemory,self.noOfAssets])
        
        assetProp = tf.reshape(tf.convert_to_tensor(self.assetProp, dtype=tf.float32),
                                shape = [1,len(self.assetProp)])
        cash = tf.reshape(tf.convert_to_tensor(self.cash, dtype=tf.float32),
                        shape = [1,1])
        
        return state,assetProp,cash

    def transact(self,action,index,train):
        """Function that buys and sells assets. The cash used to buy
        and sell is credited/debited to the cash variable.
        Args:
        action (numpy array of shape (self.noOfAssets, )): this is the asset 
        proportions to have before stepping through the trading day. All buy 
        and sell must be done before trading starts.
        index (int): the index of the current time step
        train (bool): whether the function call is used for training or testing.
        """
        # action = np.array(action)
        delta = (action - self.assetProp)*self.portVal
        if train:
            price = np.array(self.dfs["Close"].iloc[index,0:-1])        
        else:
            index+=self.trainingIndex[1]
            price = np.array(self.dfs["Close"].iloc[index,0:-1])
            
        noOfShares = np.abs(delta/price)
        # mask 1 is to filter out those assets whose shares to transact is less than 1
        mask1 = np.where(noOfShares>1,1,0)
        self.assetVal = self.assetVal+(mask1*delta)
        # subtracting broker fee for both buy/sell transaction
        self.cash+= -np.sum(mask1*delta) - (np.sum(mask1)*self.brokerFee)               
        self.portVal= np.sum(self.assetVal)  # this should keep the porfolio value unchanged.
        self.assetProp = action
        if self.cash < 0 or self.portVal < 0:
            # huge penalty if cash reserve is used up!
            self.penalty = -self.totalInv
    
    def stepDt(self, action, index,train=True):
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
        train(boolean): whether the function is called during training or testing
        """
        # buying and selling assets
        self.transact(action,index,train) 
        if train:       
            priceRatio = np.array(self.dfs["Close"].iloc[index+1,0:-1]) / \
                            np.array(self.dfs["Close"].iloc[index,0:-1])
        else:
            index+=self.trainingIndex[1]
            priceRatio = np.array(self.dfs["Close"].iloc[index+1,0:-1]) / \
                        np.array(self.dfs["Close"].iloc[index,0:-1])
        self.assetVal = self.assetVal * priceRatio
        self.portVal = np.sum(self.assetVal)
                
    def getReward(self):
        done = self.isDone()
        if done:
            return 0
        else:
            return (self.portVal+self.cash)
    
    def isDone(self):
        """If the cash reserve is used up the epoch will end.
        Returns:
            Boolean: signifies the end of an epoch
        """
        if self.cash < 0 or np.any(self.portVal < 0):
            return True
        return False
    
# e = env(dataFile="C:\\Users\\abhij\\RL_trading\\Daily_stocks_data\\portfolio.xlsx",
#         sheetNames=["Open", "High", "Low", "Close", "Volume"],
#         trainingIndex=[0,2259])
# e.getStates(30)