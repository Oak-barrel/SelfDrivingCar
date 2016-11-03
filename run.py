import argparse
from Networks import net0
import os
import time
from utils import GenerateDataset

#################
def train(model, trainfolder, devfolder, batchSize=100, epoch=100):
	""" Runs the model on the trainset (generator)
		and monitors the performance (RMSE error, test-time, train-time) on valset and trainset every epoch.
		The monitored values are stored as a json file in the checkpointed folder
		Training is stopped the moment validation accuracy drops
	"""

	BestDevError, iterations = 1000, 0
	trainset = GenerateDataset(trainfolder, batchSize=batchSize, epoch=epoch)
	for batch_x, batch_y in trainset:
		start = time.time()
		rmse = model.fit(batch_x, batch_y)
		print 'Took %.3fs/batch, Train Error %.5f' %(time.time()-start, rmse)
		
		if iterations and iterations%1000 == 0:
			totbatch, rmse = 0.0, 0.0
			start = time.time()
			devset = GenerateDataset(devfolder, batchSize=64, epoch=1)
			for batch_x, batch_y in devset:
				_,err = model.predict(batch_x, batch_y)
				rmse+=err
				totbatch += 1
			DevError = rmse/totbatch
			print ' Dev Error %.5f' %(rmse/totbatch)
			print 'Classify time %.5fs/epoch '%(time.time()-start)
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
EPOCH = 150



#

parser = argparse.ArgumentParser(description='Steering angle model trainer')
parser.add_argument('--train', type=str, default="", help='Location of the train images')
parser.add_argument('--dev', type=str, default="", help='Location of the dev images')
parser.add_argument('--test', type=str, default="", help='Location of the test images')
parser.add_argument('--checkpoint', type=str, default="", help='Location where the model parameters are stored')
parser.add_argument('--batchSize', type=str, default="", help='BatchSize: 20:200')
parser.add_argument('--Network', type=str, default="", help='Network: n0, n1')
parser.add_argument('--Pretrain', type=str, default=None, help='Enable if the network needs to be Pretrained: Default None')

net0.add_argument(parser)

args = parser.parse_args()

if args.Network == 'n0':
	net0.check_argument(args)
	model = net0.AttentionNet(args)
else:
	raise NameError('--Network cannot be %s, Please use --help' %(args.Network))
if args.train != None:
	train(model, args.train, args.dev, epoch=EPOCH, batchSize= int(args.batchSize))
else:
	raise NameError('To be completed')

