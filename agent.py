import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.keras.optimizers import Adam
from memory import memory
from networks import criticNetwork, actorNetwork
from env import env

class agent(env):
    
    def __init__(self,dataFile, sheetNames,trainingIndex,
                outputDir,alpha = 0.001,beta = 0.002,
                gamma = 0.99,maxMemorySize = 10**3,
                tau = 0.005,fc1 = 400, fc2 = 300,batchSize = 128,
                marketMemory=20):
        """
        Args:
            dataFile (string): location of the excel file
            sheetNames (list): list of sheet names holding the OHCLV data
            trainingIndex (list): list containing the starting and ending index
            for the training data set.            
            outputDir (string): folder location to save the NN weights
            alpha (float, optional): learning rate for actor network. Defaults to 0.001.
            beta (float, optional): learning rate for critic network. Defaults to 0.002.
            gamma (float, optional): future discount factor. Defaults to 0.99.
            maxMemorySize (int, optional): maximum size of memory buffer. Defaults to 10**3.
            tau (float, optional): update parameter for target actor network. Defaults to 0.005.
            fc1 (int, optional): _description_. Defaults to 400.
            fc2 (int, optional): _description_. Defaults to 300.
            batchSize (int, optional): mini batch size for weight optimization of NN. Defaults to 128.
            marketMemory (int, optional): number of previous observations required to predict
            the asset proportions. Defaults to 20.
        """
        
        env.__init__(self,dataFile=dataFile, sheetNames=sheetNames,
                    trainingIndex=trainingIndex,marketMemory=marketMemory)
        self.gamma = gamma
        self.tau = tau
        self.memory = memory(maxMemorySize)
        self.batchSize = batchSize
        
        inputShape = (1,self.noOfPriceFeatures,self.marketMemory, self.noOfAssets)
        self.actor = actorNetwork(directory=outputDir, outputSize=self.noOfAssets,
                                inputShape=inputShape,actionShape=(1,self.noOfAssets))
        self.critic = criticNetwork(directory = outputDir, stateShape=inputShape,
                                    actionShape=(1,self.noOfAssets))
        self.targetActor = actorNetwork(directory=outputDir, name = 'target-actor', 
                                        outputSize=self.noOfAssets,inputShape=inputShape,
                                        actionShape=(1,self.noOfAssets))
        self.targetCritic = criticNetwork(directory = outputDir, name = 'target-critic',
                                        actionShape=(1,self.noOfAssets),stateShape=inputShape)
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.targetActor.compile(optimizer=Adam(learning_rate=alpha))
        self.targetCritic.compile(optimizer=Adam(learning_rate=beta))

        self.updateNetworkParams(tau=1)
    
    def updateNetworkParams(self, tau = None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.targetActor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.targetActor.set_weights(weights)
        
        weights = []
        targets = self.targetCritic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.targetCritic.set_weights(weights)
    
    def saveModels(self):
        print('.......saving models.....')
        self.actor.save_weights(self.actor.chkptFile)
        self.targetActor.save_weights(self.targetActor.chkptFile)
        self.critic.save_weights(self.critic.chkptFile)
        self.targetCritic.save_weights(self.targetCritic.chkptFile)
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.chkptFile)
        self.targetActor.load_weights(self.targetActor.chkptFile)
        self.critic.load_weights(self.critic.chkptFile)
        self.targetCritic.load_weights(self.targetCritic.chkptFile)
    
    def remember(self,ohclv,assetProp,cash,
                ohclv_,assetProp_,cash_,
                reward,done):
        
        self.memory.storeTransition(ohclv,assetProp,cash,
                                ohclv_,assetProp_,cash_,
                                reward,done)
    
    def takeAction(self,ohclv,assetProp,cash):
        """Method that returns the asset proportions depending on the current state
        of the environment
        Args:
            Check the call function for actor class in "networks" module
        Returns:
            tensor: of shape (1,number_of_assets)
        """
        action = self.actor(ohclv,assetProp,cash)
        action = tf.reshape(action, shape=[1,self.noOfAssets])
        return action
    
    def learn(self):
        if self.memory.memoryCtr < self.batchSize:
            return

        ohclv,asseProp,cash, ohclv_,asseProp_,cash_,\
        reward,done = self.memory.sampleBuffer(self.batchSize)
        
        with tf.GradientTape() as tape:
            target_actions = self.targetActor(ohclv_,asseProp_,cash_)
            # this is the future reward; reward at next time step
            critic_value_ = tf.squeeze(self.targetCritic(
                                ohclv_,asseProp_,cash_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(ohclv,asseProp,cash, 
                                                asseProp_), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(ohclv_,asseProp_,cash_)
            actor_loss = -self.critic(ohclv_,asseProp_,cash_, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        
        self.updateNetworkParams()