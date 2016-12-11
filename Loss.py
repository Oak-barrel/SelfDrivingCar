import tensorflow as tf



def RMSE(y_pred, y_true):
    """ Input tensors NX1
        output scalar tensor
    """
    return tf.reduce_mean(tf.square(y_pred-y_true), name='MSE')



def BinnedAccuracy(y_pred, y_true, X):
    """1 if  predicition within 20 degree/meter  1% error"""
    return tf.reduce_mean(tf.cast(tf.abs(y_true-y_pred)<20, tf.float32), name='BinnedAccuracy')
