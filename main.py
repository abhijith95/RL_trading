import random
from agent import agent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

MODEL_SAVE_DIR = "C:\\Users\\abhij\\RL_trading\\Best_model"
WTS_SAVE_DIR = "C:\\Users\\abhij\\RL_trading\\Best_weights"
DATA_FILE = "C:\\Users\\abhij\\RL_trading\\Daily_stocks_data\\portfolio.xlsx"
SHEET_NAMES = ["Open", "High", "Low", "Close", "Volume"]
TRAINING_INDICES = [0,2259]
EPOCHS = 2000
EPOCH_SIZE = 30  # nunber of consecutive days to train the agent
MARKET_MEMORY = 10

tradingAgent = agent(dataFile= DATA_FILE, sheetNames= SHEET_NAMES,
                    trainingIndex= TRAINING_INDICES, marketMemory=MARKET_MEMORY,
                    outputDir=WTS_SAVE_DIR)

def trainAgent(saveDir):
    """Function that trains the agent

    Args:
        saveDir (string): folder location to save the model with maximum reward
    """
    growthRecord = []
    bestReward = -np.inf
    for i in range(EPOCHS):
        tradingAgent.reset()
        startIndex = random.randint(TRAINING_INDICES[0]+MARKET_MEMORY,
                                    TRAINING_INDICES[1]-EPOCH_SIZE)
        print("Training epoch: ", i)
        initPortfolioValue = tradingAgent.portVal + tradingAgent.cash
        
        for index in range(startIndex, startIndex+EPOCH_SIZE):
            ohclv,assetProp,cash = tradingAgent.getState(index)
            action = tradingAgent.takeAction(ohclv,assetProp,cash)
            action_ = np.array(action).reshape(tradingAgent.noOfAssets,)
            tradingAgent.stepDt(action_,index)
            reward = tradingAgent.getReward()
            done = tradingAgent.isDone()
            ohclv_,assetProp_,cash_ = tradingAgent.getState(index+1)
            tradingAgent.remember(ohclv,assetProp,cash,
                                ohclv_,assetProp_,cash_,
                                reward,done)
            tradingAgent.learn()
            
            if done:
                break
            # end for
            
        growthRecord.append((tradingAgent.portVal + tradingAgent.cash -\
                            initPortfolioValue)/ initPortfolioValue)
        
        if reward > bestReward:
            bestReward = reward
            tradingAgent.saveModels()
            bestModel = tradingAgent.actor
            bestModel.save(saveDir)
        # end for
    plt.plot(range(EPOCHS), growthRecord)
    plt.show()
    
    return bestModel

def testAgent(actorModel):
    """Function that tests the bestModel

    Args:
        actorModel (tensorflow model): best actor model trained by the agent
    """
    growthRecord = []
    portfolioVal = []
    tradingAgent.reset()
    initPortfolioValue = tradingAgent.portVal + tradingAgent.cash
    # portfolioVal.append(tradingAgent.portVal)
    loopRange = range(tradingAgent.marketMemory+1,
                    len(tradingAgent.test['Close'])-1)
    
    for index in loopRange:        
        ohclv,assetProp,cash = tradingAgent.getState(index, train=False)
        action = tradingAgent.takeAction(ohclv,assetProp,cash)
        action_ = np.array(action).reshape(tradingAgent.noOfAssets,)
        tradingAgent.stepDt(action_,index, train=False)
        reward = tradingAgent.portVal + tradingAgent.cash
        growthRecord.append((reward)/ initPortfolioValue)
        portfolioVal.append(tradingAgent.portVal/initPortfolioValue)
    
    print("Cumulative gains = ", 
        ((reward - initPortfolioValue)/ initPortfolioValue))
    plt.plot(loopRange, portfolioVal)
    plt.show()

# bestModel = trainAgent(MODEL_SAVE_DIR)
bestModel = tf.keras.models.load_model(MODEL_SAVE_DIR)
testAgent(bestModel)