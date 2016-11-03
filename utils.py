import os
import random

def GenerateDataset(folder, epoch=100, batchSize= 64):
	random.seed(234)

	for _ in range(epoch):
		total_batch = (1000/batchSize) +1
		isDone = False
		for i in range(total_batch):
			if i == total_batch-1 : isDone  = True

			yield [[0.0]*5 for _ in range(batchSize)], [1]*batchSize
