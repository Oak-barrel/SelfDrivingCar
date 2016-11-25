import os
import scipy.misc as misc
import numpy as np
import h5py
import random
import time

from matplotlib import pyplot as plot

def ReduceResolution(fname, H=80,W=40):
	#datafile = '../datasets/2016-04-21--14-48-08.h5'
        #fname = '2016-01-31--19-19-25.h5'
        #fname = '2016-01-30--11-24-51.h5'
       #fname = '2016-03-29--10-50-20.h5'
        datafile = '../datasets/%s' %(fname)
        #logfile = '../datasets/log-2016-04-21--14-48-08.h5'
        logfile = '../datasets/log-%s' %(fname)
        newdatafile = '../datasets/Modified-%s' %(fname)
	data = h5py.File(datafile,'r')
        log = h5py.File(logfile,'r')
	newdata = h5py.File(newdatafile,'w')
        print log['times'].shape
        #print max(log['cam1_ptr'])
        skip = 400 #front and back 6+6 min of video
        skip_frames = 10 #10Hz sampling rate
        delta = (log['times'].shape[0]-2*skip*100)/skip_frames +1 
	Xarray = newdata.create_dataset('X', (delta,H,W,3))
        Yarray = newdata.create_dataset('steering_angle',(delta,1))
        speed = newdata.create_dataset('speed',(delta,1))
        
	for cnt,i in enumerate(xrange(skip*100, log['times'].shape[0]-skip*100, skip_frames)): 
            X = data['X'][log['cam1_ptr'][i]]
            X = np.transpose(X, (2,1,0))
            Xnew = (misc.imresize(X, (H,W,3))).astype(np.float32)
            if i%1000 == 0: print i
            Xarray[cnt] = Xnew
            Yarray[cnt] = log['steering_angle'][i]
            speed[cnt] = log['speed'][i]
        return
	



class Generate():
    def __init__(self, datafile, logfile, epoch=100, batchSize=32, num_examples=None):
        self.datafile = datafile
        self.logfile = logfile
        self.epoch = epoch 
        self.batchSize = batchSize
        self.num_examples = num_examples
        data = h5py.File(datafile,'r')
        log = h5py.File(logfile,'r')
        print data.keys()
        self.X, steer = data['X'], log['steering_angle']
        #plot.imshow(X[10])
        #plot.show()
        self.steernew = steer[::5]
    
        self.examplecount = min([num_examples,self. steernew.shape[0]]) if num_examples else self.steernew.shape[0]
        assert self.examplecount <= self.X.shape[0] - 8
        
    def GenerateX(self):
        
        random.seed(234)
        #datafile = '../datasets/2016-04-21--14-48-08.h5'
        #logfile = '../datasets/log-2016-04-21--14-48-08.h5'
        throwaway=8
        for _ in range(self.epoch):
                order = range(throwaway, self.examplecount)
                random.shuffle(order)
                for i in range(0, self.examplecount, self.batchSize):
                        x = self.X[sorted(order[i:i+self.batchSize]),:,:,:] #sorted order reqd for indexi
                        yield x
        return

    def GenerateY(self):
        
        random.seed(234)
        #datafile = '../datasets/2016-04-21--14-48-08.h5'
        #logfile = '../datasets/log-2016-04-21--14-48-08.h5'
        throwaway=8
        for _ in range(self.epoch):
                order = range(throwaway, self.examplecount)
                random.shuffle(order)
                for i in range(0, self.examplecount, self.batchSize):
                        y = self.steernew[map(lambda x: x-8, sorted(order[i:i+self.batchSize]))]
                        #print y[5:10], np.std(y)
                        yield y




if __name__ == '__main__':
    import sys
    print sys.argv[1]
    print 'Started...'
    start = time.time()
    ReduceResolution(sys.argv[1], 80,40)
#


    print 'Time taken %.3fs' %(time.time()-start)



