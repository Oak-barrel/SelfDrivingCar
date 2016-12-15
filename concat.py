# every input file has a different length,
# so I first scan through the files, compute their length,
# `self.lengths` here is an array of cumulative lengths
# so first I create the dataset for full length with resize(lengths[-1])
# then I fill it in.

import h5py
import exceptions

fnames = """2016-01-30--11-24-51.h5
2016-01-30--13-46-00.h5
2016-01-31--19-19-25.h5
2016-02-02--10-16-58.h5
2016-02-08--14-56-28.h5
2016-02-11--21-32-47.h5
2016-05-12--22-20-00.h5
2016-03-29--10-50-20.h5""".split('\n')



def LoadData(fname):
    F = h5py.File('datasets/Modified-%s' %fname,'r')
    return F

csum = [ LoadData(each)['X'].shape[0] for each in fnames]
print csum
csum = sum(csum)


batchSize = 1024
rem = batchSize - csum%batchSize
csum += rem #batchsize divisible by 1024


def run(files, csum):

    with h5py.File('datasets/Modified-train.h5', mode='w') as h5f:
        prev_sum = 0
        h5f.create_dataset('X', shape=(csum,80,40,3))
        h5f.create_dataset('steering_angle', shape=(csum,1))
        h5f.create_dataset('speed', shape=(csum,1))
        lengths = []
        for n, fitsfile in enumerate(files):
            print("Processing %s" % fitsfile)
            data = LoadData(fitsfile)
            print("Data read, length %d" %data['X'].shape[0])
            h5f['X'][prev_sum:prev_sum+data['X'].shape[0]] = data['X']
            h5f['steering_angle'][prev_sum:prev_sum+data['steering_angle'].shape[0]] = data['steering_angle']
            h5f['speed'][prev_sum:prev_sum+ data['speed'].shape[0]] = data['speed']
            prev_sum += data['X'].shape[0]

        data = LoadData(files[-3])
        h5f['X'][prev_sum:prev_sum+data['X'].shape[0]] = data['X'][:rem]
        h5f['steering_angle'][prev_sum:prev_sum+data['steering_angle'].shape[0]] = data['steering_angle'][:rem]
        h5f['speed'][prev_sum:prev_sum+ data['speed'].shape[0]] = data['speed'][:rem]
        h5f.flush()


run(fnames, csum)
