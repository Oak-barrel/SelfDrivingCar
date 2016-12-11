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
#params['lr'] = 0.01
#params['lr'] = 0.001
params['lr'] = 0.01
#params['lr'] = 0.01
#params['lr'] = 0.1

#params['architecture'] = 'dave2' #draw, dave2 or commaAi or TestArchitecture
params['architecture'] = 'commaAi'
#params['architecture'] = TestArchitecture
#params['architecture'] = 'draw'

train_fnames = ['train.h5']

def loadData(fname):
    #prefixed with modified
    print 'Loading %s ... '%fname
    Data = h5py.File('datasets/Modified-%s' %fname, 'r')
    print 'Number of samples %d' %Data['X'].shape[0]
    return Data['X'], Data['steering_angle']

xval, yval = loadData('2016-05-12--22-20-00.h5')
xtest, ytest = loadData('2016-06-08--11-46-01.h5')

print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

print params['architecture']
if params['architecture'].lower() == 'convbase':
    network = Architecture.TestArchitecture(80,40,3)
elif params['architecture'].lower() == 'attention':
    network = Architecture.DRAW(80,40,3)
elif params['architecture'].lower() == 'commaai':
    network = Architecture.CommaAi(80,40,3)
elif params['architecture'].lower() == 'dave2':
    network = Architecture.Dave2(80,40,3)
else:
    raise unimplementedError

print 'Time to Compile %.3fs' %(time.time()-start)

network = regression(network, optimizer='adam', learning_rate= params['lr'],loss=Loss.RMSE, metric=Loss.BinnedAccuracy)

model = tflearn.DNN(network,  best_checkpoint_path = 'checkpoint/%s' %params['architecture'], tensorboard_verbose=0, max_checkpoints=3, best_val_accuracy=0.20)

import os
if os.path.isfile('checkpoint/%s.tflearn' %params['architecture']):
    print 'Loading PretrainedModel'
    model.load('checkpoint/%s.tflearn' %params['architecture'])
else:
    print 'Training from Scratch'



lrList = [0.01, 0.003, 0.0001]
for lr in lrList:
    params['lr'] = lr
    start = time.time()
    for fname in train_fnames:
        xtrain, ytrain = loadData(fname)
        model.fit(xtrain, ytrain, batch_size=512,validation_set=(xval, yval), show_metric=True, n_epoch=15, run_id='%s_%.3f' %(params['architecture'], params['lr']), snapshot_epoch=True, shuffle=True)
        print zip(model.predict(xtest[:10]), ytest[:10])

        model.save('checkpoint/%s.%.3f.tflearn' %(params['architecture'], params['lr']))
    #    model.save('checkpoint/%s.tflearn' %(params['architecture']))

    print('Time taken %.3f' %(time.time()-start))
    #print 'MSE: %.3f' %(model.evaluate(xtest, ytest, 1024))
    print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

