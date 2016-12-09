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
    network = fully_connected(network, 512, activation='elu', regularizer='L2')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1, activation='linear', regularizer='L2')
    return network




def DRAW():
    """ Recurrent models of attention

    """

    x = input_data(shape=[None, height, width, channel])

    def filterbank(gx, gy, sigma2,delta, N):
        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
        Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
        # normalize, sum over A and B dims
        Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
        Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
        return Fx,Fy

    def attn_window(scope,h_dec,N):
#        with tf.variable_scope(scope,reuse=DO_SHARE):
        params = fully_connected(h_dec,5, reuse=True, name=scope)
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
        gx=(A+1)/2*(gx_+1)
        gy=(B+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

    def read_attn(x,h_dec_prev):
        Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
        def filter_img(img,Fx,Fy,gamma,N):
             Fxt=tf.transpose(Fx,perm=[0,2,1])
             img=tf.reshape(img,[-1,B,A])
             glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
             glimpse=tf.reshape(glimpse,[-1,N*N])
             return glimpse*tf.reshape(gamma,[-1,1])
        x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
        #x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
        return tf.concat(1,[x,x_hat]) # concat along feature axix:

    ## ENCODE ##
    def encode(state,input):
        #Conv-Layers, other network
        #with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(input,state)


    def generate(input, h_prev):
        """ ConvNet based representation """


        return
    # construct the unrolled computational graph
    # Query = tf.
    initial_representation = generate(x,tf.zeros(512,height*width*3) )
    for t in range(T):
        c_prev = initial_representation if t==0 else cs[t-1]
        #x_hat = x - tf.sigmoid(c_prev) # error image
        #r = read(x,x_hat,h_dec_prev)
        r = read(x, h_prev)
        Reps = generate(r)
        h, enc_state = encode(enc_state, Reps)
        h_prev = h


    #Regression
    network = dropout(h_prev, 0.01)
    network = fully_connected(network, 512, activation='elu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1, activation='elu')
    return network













