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
EPOCHS = 1000
EPOCH_SIZE = 500  # nunber of consecutive days to train the agent
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
        initPortfolioValue = tradingAgent.getReward()
        
        for index in range(startIndex, startIndex+EPOCH_SIZE):
            state = tradingAgent.getState(index)
            action = tradingAgent.takeAction(state)
            action_ = np.array(action).reshape(tradingAgent.noOfAssets,)
            tradingAgent.stepDt(action_,index)
            reward = tradingAgent.getReward()
            done = tradingAgent.isDone()
            newState = tradingAgent.getState(index+1)
            tradingAgent.remember(state,action,reward,newState,done)
            tradingAgent.learn()
            
            if done:
                break
            # end for
            
        growthRecord.append((reward - initPortfolioValue)/ initPortfolioValue)
        
        if reward > bestReward:
            bestReward = reward
            tradingAgent.saveModels()
            bestModel = tradingAgent.actor
            bestModel.save(saveDir)
        # end for
    plt.plot(range(EPOCHS), growthRecord)
    
    return bestModel

def testAgent():
    pass

bestModel = trainAgent(MODEL_SAVE_DIR)