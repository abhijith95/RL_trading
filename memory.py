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
        self.memorySize = maxSize
        self.ohclv = deque(maxlen=maxSize)
        self.asseProp = deque(maxlen=maxSize)
        self.cashBuffer = deque(maxlen=maxSize)
        # the following ones are the next steps' observation points
        self.ohclv_ = deque(maxlen=maxSize)
        self.asseProp_ = deque(maxlen=maxSize)
        self.cashBuffer_ = deque(maxlen=maxSize)
        self.reward = deque(maxlen=maxSize)
        self.done = deque(maxlen=maxSize)
        self.bufferList = [self.ohclv, self.asseProp, self.cashBuffer,
                            self.ohclv_, self.asseProp_,self.cashBuffer_,
                            self.reward,self.done]
    
    def storeTransition(self,ohclv,asseProp,cash,
                        ohclv_,asseProp_,cash_,
                        reward,done):
        """Method to store the experience in the momory buffer
        Args:
            ohclv (tensor): the past marketMemory days of OHCLV value for
            all the assets. The shape of the tensor is (1,number_of_price_features,
            marketMemory, number_of_assets)            
            asseProp (tensor): asset proportions. Shape of tensor is (1,number_of_assets)
            cash (float): remaining cash value in the portfolio
            ohclv_ (tensor): the new marketMemory days of OHCLV value for
            all the assets. The shape of the tensor is (1,number_of_price_features,
            marketMemory, number_of_assets)'
            asseProp_ (tensor): New asset proportions. Shape of tensor is (1,number_of_assets)
            cash_ (float): remaining cash value in the portfolio at next time step          
            reward (float): portofolio value
            done (boolean): to indicate if the epoch training is finished or not
        """
        temp = [ohclv,asseProp,cash,ohclv_,asseProp_,cash_,reward,done]
        for i in range(len(self.bufferList)):
            self.bufferList[i].append(temp[i])
        self.memoryCtr+=1
    
    def sampleBuffer(self, bufferSize):
        """Function that returns a mini-batch of randomly selected experiences from
        the memory buffer.
        Args:
            bufferSize (int): mini-batch size to train the NN
        """
        maxmem = min(self.memoryCtr,self.memorySize) # to see if the memory is filled or not
        batch = np.random.choice(maxmem,bufferSize,replace=False)
        # separating the info
        info = []
        for i in range(len(self.bufferList)):
            # going through all the memory variables
            temp = []
            for j in batch:
                temp.append(self.bufferList[i][j])            
            if i <= 5:
                # we don't want to convert rewards and done into tensor
                batchShape = list(self.bufferList[i][j].shape[1:]) # because the first index refers to batch size
                batchShape.insert(0,bufferSize)
                temp = tf.reshape(tf.convert_to_tensor(temp),
                                    shape = batchShape)
            else:
                temp = np.array(temp)
            info.append(temp)
        return info