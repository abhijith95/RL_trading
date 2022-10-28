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
        self.ohclvInput = keras.layers.InputLayer(input_shape=stateShape)
        self.assetPropInput = keras.layers.InputLayer(input_shape=actionShape)
        self.actionInput = keras.layers.InputLayer(input_shape=actionShape)
        self.cashInput = keras.layers.InputLayer(input_shape = (1,1))
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.q = Dense(1, activation=None)
    
    def call(self,ohclv,assetProp,cash,action):
        """
        Args:
            ohclv (tensor): state of the current environment
            of shape (1,number_of_price_features,
            market_memory,number_of_assets)
            assetProp (tensor): asset proportion in the current time step.
            Shape of tensory is (1,number_of_assets)
            cash (float): cash balance in the current time step
            action (tensor): asset proprtion in the next time step.
            Shape of tensory is (1,number_of_assets)      
                
        Returns:
            q(s,a): this is the Q value of the state-action pair predicted by the NN
        """
        state = Flatten()(self.ohclvInput(ohclv))
        assetProp = Flatten()(self.assetPropInput(assetProp))
        cash = Flatten()(self.cashInput(cash))
        action = Flatten()(self.actionInput(action))
        
        action_value = self.fc1(tf.concat([state, assetProp,cash,action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class actorNetwork(keras.Model):
    def __init__(self,directory,outputSize,inputShape,actionShape,fc1dims = 512,
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
        self.ohclvInput = keras.layers.InputLayer(input_shape=inputShape)
        self.assetPropInput = keras.layers.InputLayer(input_shape=actionShape)
        self.cashInput = keras.layers.InputLayer(input_shape = (1,1))
        self.fc1 = Dense(self.fc1dims, activation='relu')
        self.fc2 = Dense(self.fc2dims, activation='relu')
        self.mu = Dense(outputSize, activation='softmax')
        # self.ouput = self.mu(self.fc2(self.fc1(self.stateInput)))
        # self.model = keras.Model(inputs= self.stateInput,
        #                         outputs = self.ouput, name=self.modelName)
    
    def call(self,ohclv,assetProp,cash):
        """
        Args:
            ohclv (tensor): state of the current environment
            of shape (1,number_of_price_features,
            market_memory,number_of_assets)
            assetProp (tensor): asset proportion in the current time step.
            Shape of tensory is (1,number_of_assets)
            cash (float): cash balance in the current time step

        Returns:
            tensorflow tensor: action predicted by the NN. This tensor is of
            shape (1, number_of_assets)
        """
        state = Flatten()(self.ohclvInput(ohclv))
        assetProp = Flatten()(self.assetPropInput(assetProp))
        cash = Flatten()(self.cashInput(cash))
        
        action = self.mu(self.fc2(self.fc1(tf.concat([state, assetProp,cash], axis=1))))
        return action