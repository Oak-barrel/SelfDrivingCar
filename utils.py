import os
import scipy.misc as misc
import numpy as np
import h5py
import random
import time


def ReduceResolution():
	datafile = '../datasets/2016-04-21--14-48-08.h5'
	data = h5py.File(datafile)
	newdata = h5py.File('../datasets/resized-2016-04-21--14-48-08.h5')
	
	Xarray = newdata.create_dataset('X', (data['X'].shape[0],3,64,64))
	for i,X in enumerate(data['X']): 
		print X.shape
		X = np.transpose(X, (1,2,0))
		Xnew = misc.imresize(X, (64,64,3))
		Xnew = np.transpose(Xnew, (2,0,1))
		print Xnew.shape
		print Xarray[i].shape
		Xarray[i] = Xnew
	return
	




def GenerateDataset(datafile, logfile,  epoch=100, batchSize= 64, num_examples = None):
	random.seed(234)
	#datafile = '../datasets/2016-04-21--14-48-08.h5'
	#logfile = '../datasets/log-2016-04-21--14-48-08.h5'
	data = h5py.File(datafile)
	log = h5py.File(logfile)
	X,steer = data['X'], log['steering_angle']
	steernew = steer[::5]
	throwaway = 8	
	examplecount = min([num_examples, steernew.shape[0]]) if num_examples else steernew.shape[0]
	assert examplecount <= X.shape[0] - 8
	for _ in range(epoch):
		order = range(throwaway, examplecount)
		#print len(order)
		random.shuffle(order)
		for i in range(0, examplecount, batchSize):
			#print i, batchSize, examplecount
			Xnew = X[sorted(order[i:i+batchSize]),:,:,:] #sorted order reqd for indexi
			x = np.transpose(Xnew, (0,2,3,1))
			#Xresize = map(lambda img: misc.imresize(img, (64,64,3)), X)
			#x = np.array(Xresize) 
			y = steernew[map(lambda x: x-8, sorted(order[i:i+batchSize]))]
			#print y.shape, x.shape
			yield x,y





#print 'Started...'
#start = time.time()
#ReduceResolution()
#
#i = 0
#start = time.time()
#for x,y in GenerateDataset('../datasets/resized-2016-04-21--14-48-08.h5', '../datasets/log-2016-04-21--14-48-08.h5', epoch=10, batchSize=128):
#	print x.shape, y.shape
#	#if i>100: break
#	i+=1	


#print 'Time taken %.3fs' %(time.time()-start)



