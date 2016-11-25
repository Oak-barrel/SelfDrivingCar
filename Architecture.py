import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization




def TestArchitecture(height, width, channel):
    """
    Test Architecture
    """
    print 'Test Architecture with MaxPool' 
    network = input_data(shape=[None, height, width, channel])
    network = batch_normalization(network)
    network = conv_2d(network, 16 ,8, activation='elu',regularizer="L2", name='Conv1')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 5, activation='elu', regularizer="L2", name='Conv2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64 , 5, activation='elu', regularizer="L2", name='Conv3')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    #network = dropout(network, 0.2)
    network = fully_connected(network, 512, activation='elu')
    #network = dropout(network, 0.5)
    network = fully_connected(network, 1, activation='linear')
    #network = tf.exp(network)
    #network = fully_connected(network, 1, activation='linear')

    return network




def AlexNet(height=28, width=28, channel=1):
    """ 
        Run an alexnet over 
    """
 
    return


def Dave2(height, width, channel):
    """
    Nvidia Architecture
    """
    
    # Building convolutional network
    network = input_data(shape=[None, height, width, channel])
    network = batch_normalization(network)
    network = conv_2d(network, 24 , 5, name='Conv1')
    #network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    network = conv_2d(network, 36, 5,  name='Conv2')
    #network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    network = conv_2d(network, 48 , 5, name='Conv3')
    #network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    network = conv_2d(network, 64, 5,  name='Conv4')
    #network = max_pool_2d(network, 2)
    #network = local_response_normalization(network)
    network = conv_2d(network, 64, 5, name='Conv5')
    #network = max_pool_2d(network, 2)
  # network = local_response_normalization(network)
    network = flatten(network)
    #network = dropout(network, 0.5)
    #network = dropout(network, 0.5)
    network = fully_connected(network, 1164, activation='relu')
    network = fully_connected(network, 100, activation='relu')
    network = fully_connected(network, 50, activation='relu')
    network = fully_connected(network, 10, activation='relu')
    #network = dropout(network, 0.8)
    #network = dropout(network, 0.8)
    network = fully_connected(network, 1)
    
    return network

def CommaAi(height=28, width=28, channel=1):
    """
    Comma.Ai s convNet : with Normalization
    """
    windowsize = [8,5,5]
    subsample = [4,2,2]
    # Building convolutional network
    network = input_data(shape=[None, height, width, channel])
    network = batch_normalization(network)
    network = conv_2d(network, 16 ,windowsize.pop(0), activation='elu',regularizer="L2", name='Conv1')
    network = max_pool_2d(network, subsample.pop(0))
    network = local_response_normalization(network)
    network = conv_2d(network, 32, windowsize.pop(0), activation='elu', regularizer="L2", name='Conv2')
    network = max_pool_2d(network, subsample.pop(0))
    network = local_response_normalization(network)
    network = conv_2d(network, 64 , windowsize.pop(0), activation='elu', regularizer="L2", name='Conv3')
    network = max_pool_2d(network, subsample.pop(0))
    network = local_response_normalization(network)
    network = dropout(network, 0.2)
    network = fully_connected(network, 512, activation='elu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1, activation='elu')
      
    return network




def DRAW():
    """ Recurrent models of attention
    """

    return


###################################################
def Regressor(network):
    return network
