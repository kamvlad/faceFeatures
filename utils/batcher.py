from random import shuffle

class Batcher:
    def __init__(self, data, labels):
        self.maxSize = data.shape[0]
        self.data = data
        self.labels = labels
        self.ni = 0
        self.idxs = list(range(self.maxSize))
        shuffle(self.idxs)
    
    def nextBatch(self, size):
        assert(size <= self.maxSize)
        
        n = self.ni
        self.ni += size
        
        if self.ni > self.maxSize:
            tailIdxs = self.idxs[n:self.maxSize]
            headIdxs = self.idxs[:self.ni-self.maxSize]
            d = np.vstack([self.data[tailIdxs], self.data[headIdxs]])
            l = np.vstack([self.labels[tailIdxs], self.labels[headIdxs]])
            self.ni = 0
            shuffle(self.idxs)
            return d, l
        else:
            return self.data[self.idxs[n:self.ni]], self.labels[self.idxs[n:self.ni]]
