import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization




def TestArchitecture(height, width, channel, keep=[0.9,0.6]):
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
    network = fully_connected(network, 512, activation='elu')
    network = dropout(network, keep.pop(0))
    network = fully_connected(network, 1, activation='linear')
    #network = tf.exp(network)
    #network = fully_connected(network, 1, activation='linear')

    return network




def AlexNet(height=28, width=28, channel=1):
    """
        Run an alexnet over
    """

    return


def Dave2(height, width, channel, keep=[0.9, 0.6]):
    """
    Nvidia Architecture
    """

    # Building convolutional network
    network = input_data(shape=[None, height, width, channel])
    network = batch_normalization(network)
    network = dropout(network, keep.pop(0))
    network = conv_2d(network, 24 , 5, name='Conv1', regularizer='L2')
    network = conv_2d(network, 36, 5,  name='Conv2', regularizer='L2')
    network = conv_2d(network, 48 , 5, name='Conv3', regularizer='L2')
    network = conv_2d(network, 64, 5,  name='Conv4', regularizer='L2')
    network = conv_2d(network, 64, 5, name='Conv5', regularizer='L2')
    network = dropout(network, keep.pop(0))
    network = fully_connected(network, 1164, activation='relu', regularizer='L2')
    network = fully_connected(network, 100, activation='relu', regularizer='L2')
    network = fully_connected(network, 50, activation='relu', regularizer='L2')
    network = fully_connected(network, 10, activation='relu', regularizer='L2')
    #network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='linear', regularizer='L2')

    return network

def CommaAi(height=28, width=28, channel=1, keep = [0.9, 0.6]):
    """
    Comma.Ai s convNet : with Normalization
    """
    windowsize = [8,5,5]
    subsample = [4,2,2]
    # Building convolutional network
    network = input_data(shape=[None, height, width, channel])
    network = dropout(network, keep.pop(0))
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
    network = dropout(network, keep.pop(0))
    network = fully_connected(network, 512, activation='elu', regularizer='L2')
    network = fully_connected(network, 1, activation='linear', regularizer='L2')
    return network




def DRAW(height=28, width=28, channel=3 ,glimpse=16, encSize=256, keep=[0.9,0.6],TimeSteps=4):
    """ Recurrent models of attention
    """

    batchSize = 512
    encSize= 512
    #Create variables
    REUSE = False

    def generate(network, height=28, width=28, channel=3, encSize=256):
        """ Pass through conv nets:
        """
        windowsize = [5,5]
        subsample = [2,2]
        # Building convolutional network
        network = batch_normalization(network)
        network = conv_2d(network, 16 , windowsize.pop(0), activation='elu',regularizer="L2", name='Conv1')
        network = max_pool_2d(network, subsample.pop(0))
        network = conv_2d(network, 64, windowsize.pop(0), activation='elu', regularizer="L2", name='Conv2')
        network = max_pool_2d(network, subsample.pop(0))
        network = local_response_normalization(network)
        network = fully_connected(network, encSize, activation='elu', regularizer='L2', name='Hidden_Reps')
        return network


    x = input_data(shape=[None, height, width, channel])
    x = dropout(x, keep.pop(0))
    lstm_enc = tf.nn.rnn_cell.LSTMCell(encSize, state_is_tuple=True) # encoder Op

    def filterbank(gx, gy, sigma2,delta, N, height, width):
        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(height), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(width), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
        Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
        print Fx.get_shape()
        assert(False)
        # normalize, sum over A and B dims
        Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),1e-5)
        Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),1e-5)
        return Fx,Fy

    def linear(x,output_dim):
        """
        afine transformation Wx+b
        assumes x.shape = (batch_size, num_features)
        """
        print 'Linear dimension : ' , x.get_shape().as_list()
        try:
            w=tf.get_variable("w", [x.get_shape().as_list()[1], output_dim])
            b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        except ValueError:
            print 'Creating Linear Layer'
            w=tf.Variable(tf.random_normal([x.get_shape().as_list()[1], output_dim], stddev=0.35), name="w")
            b=tf.Variable(tf.zeros([output_dim]), name="b")

        return tf.matmul(x,w)+b


    def attn_window(height=28, width=28, scope='attention',h=None,N=3):
        with tf.variable_scope(scope, reuse=REUSE):
            params = linear(h,5) #Get glimpse params
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
        gx=(width+1)/2*(gx_+1)
        gy=(height+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(height,width)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return filterbank(gx,gy,sigma2,delta,N, height, width)+(tf.exp(log_gamma),)

    def read_attn(x,h_prev, readWindowSize=16):
        Fx,Fy,gamma=attn_window(height=80, width=40,scope="read",h = h_prev, N=readWindowSize) #Get filters with zoom factor
        print 'Filterbank shapes' ,Fx.get_shape().as_list()
        def filter_img(img,Fx,Fy,gamma,N):
             print gamma.get_shape().as_list()
             Fxt=tf.transpose(Fx,perm=[0,2,1])
             #img=tf.reshape(img,[-1,B,A])
             img = tf.transpose(img, perm=[0,2,1,3])
             assert img.get_shape().as_list()[-1] == 3
             imgList = tf.split(3, 3, img)
             print imgList[0].get_shape()
             imgList = map(lambda x: tf.squeeze(x, squeeze_dims=[3]), imgList)
             print imgList[0].get_shape()
             glimpseList = map(lambda imgDim: tf.batch_matmul(Fy,tf.batch_matmul(imgDim,Fxt)), imgList)
             print glimpseList[0].get_shape().as_list()
             glimpseList = map( lambda x: tf.mul(x, tf.reshape(gamma, [-1,1,1])) , glimpseList)


             glimpse = tf.pack(glimpseList, axis=-1)
             print glimpse.get_shape().as_list()
             return glimpse


        x = filter_img(x,Fx,Fy,gamma,readWindowSize) # batch x (readWindowSize*readWindowSize))
        return x

    ## ENCODE ##
    def encode(state, input):
        with tf.variable_scope('encoder', reuse=REUSE):
            return lstm_enc(input,state)


    # construct the unrolled computational graph
    enc_state=lstm_enc.zero_state(batchSize, tf.float32)

    #randomly initialized  intial hidden state
    h_prev = tf.zeros((batchSize,encSize))

    for t in range(TimeSteps):
        print 'One oll begun'
        r = read_attn(x, h_prev) #contains information as to where to look and the value
        Reps = generate(r, encSize)
        print Reps.get_shape()
        h, enc_state = encode(enc_state, Reps)
        h_prev = h
        REUSE = True

    network = h
    #Regression
    network = dropout(network, keep.pop(0))
    network = fully_connected(network, 512, activation='elu')
    network = fully_connected(network, 1, activation='linear')
    return network













