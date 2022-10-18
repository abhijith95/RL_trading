import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten

class criticNetwork(keras.Model):
    def __init__(self,directory,stateShape,actionShape,
                fc1dims = 512, fc2dims = 512,name = 'critic'):
        super(criticNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        self.stateInput = keras.layers.InputLayer(input_shape=stateShape)
        self.actionInput = keras.layers.InputLayer(input_shape=actionShape)
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.q = Dense(1, activation=None)
    
    def call(self,state,action):
        """
        Args:
            state (tensor): state of the current environment
            of shape (1,5,market_memory,number_of_assets)
            action (tensor): _description_

        Returns:
            q(s,a): this is the Q value of the state-action pair predicted by the NN
        """
        state = Flatten()(self.stateInput(state))
        action = Flatten()(self.actionInput(action))
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class actorNetwork(keras.Model):
    def __init__(self,directory,outputSize,inputShape,fc1dims = 512,
                fc2dims = 512,name = 'actor'):
        super(actorNetwork,self).__init__()
        self.fc1dims = fc1dims
        self.fc2dims = fc2dims
        self.modelName = name
        self.directory = directory
        # the NN is saved as .h5 file to be used later
        self.chkptFile = os.path.join(self.directory,self.modelName+'.h5')
        
        # creating the neural network note that all the layers are separate 
        # and not connected to each other
        self.stateInput = keras.layers.InputLayer(input_shape=inputShape)
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.mu = Dense(outputSize, activation='softmax')
        # self.ouput = self.mu(self.fc2(self.fc1(self.stateInput)))
        # self.model = keras.Model(inputs= self.stateInput,
        #                         outputs = self.ouput, name=self.modelName)
    
    def call(self,state):
        """
        Args:
            state (tensorflow tensor): state of the current environment
            of shape (1,5,market_memory,number_of_assets)

        Returns:
            tensorflow tensor: action predicted by the NN. This tensor is of
            shape (1, number_of_assets)
        """
        temp = self.stateInput(state)
        temp = Flatten()(temp)
        action = self.mu(self.fc2(self.fc1(temp)))
        return action