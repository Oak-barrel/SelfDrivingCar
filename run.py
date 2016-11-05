import argparse
from Networks import net0
import os
import time
from utils import GenerateDataset
import numpy as np 
#################
def train(model, trainXfile, trainSteerfile, devXfile, devSteerfile, batchSize=100, epoch=100):
    """ Runs the model on the trainset (generator)
		and monitors the performance (RMSE error, test-time, train-time) on valset and trainset every epoch.
		The monitored values are stored as a json file in the checkpointed folder
		Training is stopped the moment validation accuracy drops
    """
    display_step = 100 #120 is 1 epoch 
    print 'Started ....'
    BestDevError, iterations = 10000000000000000000000, 0
    trainset = GenerateDataset(trainXfile, trainSteerfile, batchSize=batchSize, epoch=epoch)
    for batch_x, batch_y in trainset:
        start = time.time()
	summary, rmse = model.fit(batch_x, batch_y, summary = ((iterations and iterations%display_step) == 0))
        model.add_summary(summary, iterations)
	
        if iterations and iterations%display_step == 0:
            totbatch, rmse = 0.0, 0.0
            start = time.time()
            devset = GenerateDataset(devXfile, devSteerfile, batchSize=1024, epoch=1, num_examples=1024)
	    for batch_x, batch_y in devset:
	        _,err = model.predict(batch_x, batch_y)
		assert not  np.isnan(err), 'Model divereged with Nan'
                rmse+=err
		totbatch += 1
	        DevError = rmse/totbatch
		print ' Dev Error %.5f' %(rmse/totbatch)
		print 'Classify time %.5fs/batch' %(time.time()-start)
		if DevError<BestDevError: 
		    model.save()
		    BestDevError = DevError

		print "Best DevError: %.5fs" %(BestDevError)
        iterations +=1
    print "Best DevError: %.5fs" %(BestDevError)

    return

def test(model, testset):
    """
	Runs the loaded model on test set to give out a kaggle formatted output (ImgId, steering angle)
    """


    return







#####################
#fixed config
EPOCH = 100



#
parser = argparse.ArgumentParser(description='Steering angle model trainer')
parser.add_argument('--trainX', type=str, default="", help='Location of the train images')
parser.add_argument('--trainY',type=str, default = "", help= 'Location of the steering angles')
parser.add_argument('--devX', type=str, default="", help='Location of the dev images')
parser.add_argument('--devY', type=str, default="", help='Location of the dev steering angles') 
parser.add_argument('--testX', type=str, default="", help='Location of the test images')
parser.add_argument('--testY', type=str, default="", help='Location of the test steering angles')

parser.add_argument('--checkpoint', type=str, default="", help='Location where the model parameters are stored')
parser.add_argument('--batchSize', type=str, default="", help='BatchSize: 20:200')
parser.add_argument('--Network', type=str, default="", help='Network: n0, n1')
parser.add_argument('--Pretrain', type=str, default=None, help='Enable if the network needs to be Pretrained: Default None')

net0.add_argument(parser)

args = parser.parse_args()

if args.Network == 'n0':
    net0.check_argument(args)
    model = net0.CIFR10Model(args)
else:
    raise NameError('--Network cannot be %s, Please use --help' %(args.Network))
if args.trainX != None:
    train(model, args.trainX, args.trainY, args.devX, args.devY, epoch=EPOCH, batchSize= int(args.batchSize))
else:
    raise NameError('To be completed')



# python run.py --trainX ../datasets/resized-2016-04-21--14-48-08.h5 --trainY ../datasets/log-2016-04-21--14-48-08.h5 --devX ../datasets/resized-2016-04-21--14-48-08.h5 --devY ../datasets/log-2016-04-21--14-48-08.h5 --checkpoint checkpoint/net11 --batchSize 128 Network n0 
