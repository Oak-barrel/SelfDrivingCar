import tensorflow as tf



def RMSE(y_pred, y_true):
    """ Input tensors NX1
        output scalar tensor
    """
    return tf.reduce_mean(tf.square(y_pred-y_true), name='MSE')



def BinnedAccuracy(y_pred, y_true, X):
    """1 if  predicition within 10/5000 of a degree"""
    return tf.reduce_mean(tf.cast(tf.abs(y_true-y_pred)<10, tf.float32), name='BinnedAccuracy')
