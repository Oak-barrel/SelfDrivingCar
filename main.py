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
params['architecture'] = 'CommaAi' #draw, dave or


train_fnames =  """2016-01-30--11-24-51.h5
2016-01-30--13-46-00.h5
2016-01-31--19-19-25.h5
2016-02-02--10-16-58.h5
2016-02-08--14-56-28.h5
2016-03-29--10-50-20.h5
2016-04-21--14-48-08.h5
2016-05-12--22-20-00.h5
""".split('\n')


def loadData(fname):
    #prefixed with modified
    print 'Loading %s ... '%fname
    Data = h5py.File('datasets/Modified-%s' %fname, 'r')
    print 'Number of samples %d' %Data['X'].shape[0]
    return Data['X'], Data['steering_angle']

xval, yval = loadData('2016-06-02--21-39-29.h5')
xtest, ytest = loadData('2016-02-11--21-32-47.h5')

print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

print params['architecture']
if params['architecture'].lower() == 'convbase':
    network = Architecture.TestArchitecture(80,40,3)
elif params['architecture'].lower() == 'attention':
    network = Architecture.DRAW(80,40,3)
elif params['architecture'].lower() == 'commaai':
    network = Architecture.CommaAi(80,40,3)
else:
    raise unimplementedError


print 'Time to Compile %.3fs' %(time.time()-start)

network = regression(network, optimizer='adam', learning_rate= params['lr'],loss=Loss.RMSE, metric=Loss.BinnedAccuracy)

#G = Generate('../datasets/resized-2016-04-21--14-48-08.h5', '../datasets/log-2016-04-21--14-48-08.h5', epoch=20, batchSize=128)


model = tflearn.DNN(network,  best_checkpoint_path = 'checkpoint/%s' %params['architecture'], tensorboard_verbose=0, max_checkpoints=3, best_val_accuracy=0.20)

import os
if os.path.isfile('checkpoint/%s.tflearn' %params['architecture']):
    print 'Loading PretrainedModel'
    model.load('checkpoint/%s.tflearn' %params['architecture'])
else:
    print 'Training from Scratch'

start = time.time()
#Run for 6 epochs on the DataSet
rounds = 5
for _ in xrange(rounds):
    for fname in train_fnames[:-1]:
        xtrain, ytrain = loadData(fname)
        model.fit(xtrain, ytrain, batch_size=512,validation_set=(xval, yval), show_metric=True, n_epoch=1, run_id='%s' %params['architecture'], snapshot_epoch=True, shuffle=True)
        print zip(model.predict(xtest[:10]), ytest[:10])

    model.save('checkpoint/%s.tflearn' %params['architecture'])

print('Time taken %.3f' %(time.time()-start))
print 'MSE: %.3f' %(np.mean((model.predict(xtest[:500])-ytest[:500])**2))
print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

