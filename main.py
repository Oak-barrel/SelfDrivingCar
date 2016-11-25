#memory issues with tflearn

import argparse
import Architecture

from data_helper import Generate
import Loss

import time
import tflearn
from tflearn.layers.estimator import regression
import h5py
import numpy as np

start = time.time()
params = {}
params['lr'] = 0.01


data = h5py.File('../datasets/Modified-2016-04-21--14-48-08.h5', 'r')
numOfFrames = data['X'].shape[0]
trainFrames = 9*numOfFrames/10
print 'TrainSet,valSet, testSet (%d, %d, %d)' %(0.9*trainFrames, 0.1*trainFrames, numOfFrames-trainFrames)
xtrain, ytrain, xtest,ytest = data['X'][:trainFrames],data['steering_angle'][:trainFrames], data['X'][trainFrames:], data['steering_angle'][trainFrames:]


trainSample = xtrain.shape[0]
xt,yt, xval,yval = xtrain[:9*trainSample/10],ytrain[:9*trainSample/10], xtrain[9*trainSample/10:], ytrain[9*trainSample/10] 

print 'Mean Baseline %.3f' %(np.mean((ytrain-np.mean(ytrain))**2))
network = Architecture.TestArchitecture(80,40,3)

print 'Time to Compile %.3fs' %(time.time()-start)

network = regression(network, optimizer='adam', learning_rate= params['lr'],loss=Loss.RMSE, metric=Loss.BinnedAccuracy)



#G = Generate('../datasets/resized-2016-04-21--14-48-08.h5', '../datasets/log-2016-04-21--14-48-08.h5', epoch=20, batchSize=128)


model = tflearn.DNN(network,  best_checkpoint_path = 'checkpoint/CommaAi', tensorboard_verbose=0, max_checkpoints=3, best_val_accuracy=0.20)
start = time.time()
model.load('checkpoint/CommaAi6055')
model.fit(xt, yt, batch_size=256,validation_set=(xval, yval), show_metric=True, n_epoch=6)
print zip(model.predict(xtest[:10]), ytest[:10])
print('Time taken %.3f' %(time.time()-start))
