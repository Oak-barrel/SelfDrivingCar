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

import argparse
parser = argparse.ArgumentParser(description='Autonomous driving. Class Project')
parser.add_argument('--lr', dest='lr')
parser.add_argument('--keep1', dest='keep1')
parser.add_argument('--keep2', dest='keep2')
parser.add_argument('--architecture', dest='architecture')
parser.add_argument('--glimpse', dest='glimpse') #only for draw
parser.add_argument('--TimeSteps', dest='TimeSteps')

args = parser.parse_args()
assert args.architecture in ['dave2','commaai', 'convbase', 'draw']
assert 1e-5<float(args.lr)<1000
assert 0<float(args.keep1)<=1 and 0<float(args.keep2)<=1

start = time.time()
params = {}


#params['lr'] = 0.01
#params['lr'] = 0.001
#params['lr'] = 0.0001
#params['lr'] = 0.003
#params['lr'] = 0.005
#params['lr'] = 0.01
#params['lr'] = 0.01

#params['architecture'] = 'dave2' #draw, dave2 or commaAi or TestArchitecture
#params['architecture'] = 'commaAi'
#params['architecture'] = TestArchitecture
#params['architecture'] = 'draw'

params['lr'] = float(args.lr)
params['architecture'] = args.architecture
params['dropout'] = [float(args.keep1), float(args.keep2)]
if args.glimpse:
    params['glimpse'] = int(args.glimpse)
    assert 4<int(args.glimpse)<40
if args.TimeSteps:
    params['timestep'] = int(args.TimeSteps)
    assert 2<=float(args.TimeSteps)<=8
#params['dropout'] = [1., 1.] # [0.9, 0.6], [0.8, 0.7]]
EPOCH = 15 #Number of epochs and batch size are held constant
batchSize = 512


keep = params['dropout'][:] #send a copy
train_fnames = ['train.h5']

Modelname ='%s.%.3f.%.2f.%.2f' %(params['architecture'], params['lr'], params['dropout'][0], params['dropout'][1])
if args.glimpse: Modelname =  '%s.%d' %(Modelname, params['glimpse'])
if args.TimeSteps : Modelname = '%s.%d' %(Modelname, params['timestep'])
###########################################################################################################
def loadData(fname):
    print 'Loading %s ... '%fname
    Data = h5py.File('datasets/Modified-%s' %fname, 'r')
    print 'Number of samples %d' %Data['X'].shape[0]
    return Data['X'], Data['steering_angle']

xval, yval = loadData('2016-02-11--21-32-47.h5')
xtest, ytest = loadData('2016-06-08--11-46-01.h5')

print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

print 'Architecture...' , params['architecture']
if params['architecture'].lower() == 'convbase':
    network = Architecture.TestArchitecture(80,40,3, keep= keep)
elif params['architecture'].lower() == 'draw':
    network = Architecture.DRAW(80,40,3, glimpse=params['glimpse'], TimeSteps=params['timestep'], keep = keep)
elif params['architecture'].lower() == 'commaai':
    network = Architecture.CommaAi(80,40,3, keep=keep)
elif params['architecture'].lower() == 'dave2':
    network = Architecture.Dave2(80,40,3, keep=keep)
else:
    raise unimplementedError

print 'Time to Compile %.3fs' %(time.time()-start)
network = regression(network, optimizer='adam', learning_rate= params['lr'],loss=Loss.RMSE, metric=Loss.BinnedAccuracy)

model = tflearn.DNN(network,  best_checkpoint_path = 'checkpoint/%s' %params['architecture'], tensorboard_verbose=0, max_checkpoints=2, best_val_accuracy=0.70,tensorboard_dir='tensorBoard_dir/%s' %params['architecture'] )


import os

if os.path.isfile('checkpoint/%s.tflearn' %Modelname):
    print 'Loading PretrainedModel'
    model.load('checkpoint/%s.tflearn' %Modelname)
else:
    print 'Training from Scratch'

start = time.time()
for fname in train_fnames:
    xtrain, ytrain = loadData(fname)
    model.fit(xtrain, ytrain, batch_size=batchSize,validation_set=(xval, yval), show_metric=True, n_epoch=EPOCH, run_id='%s_%.3f' %(params['architecture'], params['lr']), snapshot_epoch=True, shuffle=True)
    print zip(model.predict(xtest[:batchSize])[:10], ytest[:10])

    model.save('checkpoint/%s.tflearn' %(Modelname))

    print('Time taken %.3f' %(time.time()-start))
    print 'Minus One Baseline (Val/Test): %.3f %.3f' %(np.mean((yval[:]+1)**2), np.mean((ytest[:]+1)**2))

