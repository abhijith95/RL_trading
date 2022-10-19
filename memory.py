from collections import deque
import random
import tensorflow as tf
import numpy as np

class memory:
    """
    This class is used to define the memory buffer where
    the experiences from the agent are stored, to be used
    to train the NN later on.
    """
    
    def __init__(self, maxSize):
        # self.memorySize = maxSize
        self.memoryCtr = 0
        self.buffer = deque(maxlen=maxSize)
    
    def storeTransition(self,state,action,reward,newState,done):
        """Method to store the experience in the momory buffer
        Args:
            state (tensor): the past marketMemory days of OHCLV value for
            all the assets. The shape of the tensor is (1,5,marketMemory, number_of_assets)
            
            action (tensor): asset proportions. Shape of tensory is (1,number_of_assets)
            reward (float): portofolio value
            newState (tensor): state of the environment after taking a step
            done (boolean): to indicate if the epoch training is finished or not
        """
        temp = [state,action,reward,newState,done]
        self.buffer.append(temp)
        self.memoryCtr+=1
    
    def sampleBuffer(self, bufferSize):
        """Function that returns a mini-batch of randomly selected experiences from
        the memory buffer.
        Args:
            bufferSize (int): mini-batch size to train the NN
        """
        exps = random.sample(self.buffer, bufferSize)
        # separating the info
        info = []
        for i in range(5):
            # indexing through state,action,reward,newState and done
            temp=[]
            for j in range(len(exps)):
                # indexing through all the experiences
                temp.append(exps[j][i])
            if i!=4 and i!=2:
                # we don't want to convert "done", "rewards" to tensor
                temp = tf.convert_to_tensor(temp)
            else:
                # convert "done" and "rewards" to numpy array
                temp = np.array(temp)
            info.append(temp)            
        return info