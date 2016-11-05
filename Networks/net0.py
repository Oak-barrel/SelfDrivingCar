import math
import tensorflow as tf
import os

def init_weights(shape, name , activation = None):
    #Xavier 2010
    print shape
    constant = math.sqrt(7./sum(shape)) if len(shape)!=1 else 0 
    if activation == 'sigmoid': constant*=4
    # raw_input('%.3f %s'%(constant, name))
    return tf.Variable(constant*tf.random_uniform(shape, minval=-1, maxval=1), name=name)





def add_argument(parser):
    print 'Adding Arguments'
	
    return
def check_argument(args):
    print 'Checking Arguments'

    return

class CIFR10Model():
    """ This model is a replica from tensorflow's ConvNet tutorial """
    def inference(self, X, y):
        """ Define equations and get the output """
        # Normalized & whitened (Queued) 
        Xm = X -  tf.reduce_mean(X, reduction_indices=0)
        Xn = Xm/tf.sqrt(tf.reduce_mean(tf.square(Xm), reduction_indices=0))
        # dropout (L1)
        Xd = tf.nn.dropout(Xn, keep_prob = self.prob1, seed=1234)
        # conv1 + maxpool + norm1 + dropout (L2)
        with tf.variable_scope('L1') as l1:
            kernel = init_weights([5,5,3, 64], name='L1-Kernel1') #64 5X5 filters on 3 channels
            conv1 = tf.nn.conv2d(Xd, kernel, [1,1,1,1], 'SAME', name='Conv1')
            bias = init_weights([64], name= 'L1-Bias1')
            conv1 = tf.nn.bias_add(conv1, bias)
            relu = tf.nn.relu(conv1, name='L1-Relu')
            pool1 = tf.nn.max_pool(relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='POOL1')
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='L1-norm')
            
        Xnorm1 = tf.nn.dropout(norm1, keep_prob = self.prob1, seed=234)
        # conv2 + maxpool + norm2+ dropout (L3)
        with tf.variable_scope('L2') as l2:
            kernel = init_weights([5, 5,64, 64], name='kernel2')
            conv2 = tf.nn.conv2d(Xnorm1, kernel, [1,1,1,1], padding='SAME', name='conv2' )
            bias2 = init_weights([64], name='L2-Bias')
            conv2 = tf.nn.bias_add(conv2, bias2)
            relu = tf.nn.relu(conv2, name='L2-Relu')
            pool2 = tf.nn.max_pool(relu, ksize = [1,3,3,1], strides =[1,2,2,1], padding='SAME', name = 'pool2')
            norm2 = tf.nn.lrn(pool2,4 , bias=1.0, alpha = 0.001/9.0, beta=0.75, name='L2-norm')

        
        #Neural net with RELU activation
        with tf.variable_scope('local') as l3:
            reshape = tf.reshape(norm2, [-1, 16*16*64])
            #dim = reshape.get_shape()[1].value
            dim = 16*16*64
            weight = init_weights([dim, self.internaldim], name='L3-weight' )
            bias = init_weights([self.internaldim], name='L3-bias')
            print norm2.get_shape(), reshape.get_shape(), weight.get_shape()
            l3 = tf.nn.relu(tf.matmul(reshape,weight) + bias, name='L3-input')
            weight2 = init_weights([self.internaldim, self.internaldim],name='L3-weight2') 
            bias2 = init_weights([self.internaldim], name='L3-bias2')
            l4 = tf.nn.relu(tf.matmul(l3, weight2) + bias2, name='L4-output')
        
        #Final Regression layer Rmax(e^(theta))
        with tf.variable_scope('final') as fn:
            weight = init_weights([self.internaldim,1], name='F-weight' )
            bias = init_weights([1], name='F-bias')
            self.output = tf.matmul(l4,weight) + bias
        ##############################SUMMARY#############################
        def _add_activation_summary(x):
            tf.histogram_summary(x.op.name+ '/activation', x)
            tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

        _add_activation_summary(l4)
        _add_activation_summary(norm2)
        _add_activation_summary(norm1)

       # tf.scalar_summary('TotalLoss', self.loss)
       # tf.scalar_summary('DataLoss:rmse', rmse )

    

        return self.output
    def __init__(self, args):
        """ We define the variables and equations here"""
        self.batchSize = 128
        self.prob1 = tf.placeholder(tf.float32, name='keepProb1')        
        self.X  = tf.placeholder(tf.float32, [None, 64, 64, 3],name = 'X')
        self.y = tf.placeholder(tf.float32, [None], name = 'Y')
        self.lambda2 = tf.placeholder(tf.float32, name = 'lambda2')
        self.lr = tf.placeholder(tf.float32, name='learnrate')
        

        self.internaldim = 100
        self.ypred = self.inference(self.X,self.y)
        
        #loss and update
        rmse = tf.reduce_mean(tf.square(self.y-self.ypred))
        regularizer = self.lambda2*tf.reduce_sum([tf.reduce_sum(tf.square(e)) for e in tf.trainable_variables()])
        self.loss = rmse+ regularizer
        #per gradient update AdaGrad + momentum - Adam
        self.adam_op = tf.train.AdamOptimizer(self.lr , name= 'Adam').minimize(self.loss)
        #init
        init = tf.initialize_all_variables()
        #saver
        self.saver = tf.train.Saver(tf.all_variables())
        self.summary_op = tf.merge_all_summaries()
        self.sess = tf.Session()
        self.sess.run(init)
        #Summaries to monitor the health of the network
        #histogram of grads, moving average loss, moving average grads, few input image
        tf.scalar_summary('TotalLoss', self.loss)
        tf.scalar_summary('DataLoss:rmse', rmse)
        

        if not os.path.isdir(args.checkpoint+'/log'): os.mkdir(args.checkpoint+'/log')
        self.summary_writer = tf.train.SummaryWriter(args.checkpoint+'/log', self.sess.graph)
         
        self.checkpoint_path = args.checkpoint+'/net0'
        return 
            
    def fit(self, x, y, summary=False):
	""" Run the inference and optimizer """
        #assert x.shape[0] == self.batchSize
        summary_op = None
        if summary:
            summary_op, err, _ = self.sess.run([self.summary_op, self.loss, self.adam_op], feed_dict = {self.X:x, self.y:y, self.lr:0.001, self.lambda2:0, self.prob1:0.5})
        else:
            err,_ = self.sess.run([self.loss, self.adam_op], feed_dict= {self.X:x, self.y:y, self.lr:0.001, self.lambda2:0, self.prob1:0.5})
	return summary_op, err

    def save(self):
        #self.saver.save(sess, checkpoint_path, global_step=step)
        self.saver.save(self.sess, self.checkpoint_path)
        return
    def add_summary(self, summary_op, step):
        self.summary_writer.add_summary(summary_op, step)

        return

    def predict(self, x, y):
	ypred, err = self.sess.run([self.ypred, self.loss], feed_dict = {self.X:x, self.y:y, self.lambda2:0, self.prob1:1.0})
        return ypred, err
