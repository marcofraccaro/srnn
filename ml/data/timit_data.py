import numpy as np
import os, gzip
import theano
import theano.tensor as T


class TimitData():
    def __init__(self, fn, batch_size):
        import numpy as np
        data = np.load(fn)

        u_train, x_train = data['u_train'], data['x_train']
        u_valid, x_valid = data['u_valid'], data['x_valid']
        (u_test, x_test,
              mask_test) = data['u_test'],  data['x_test'], data['mask_test']

        # assert u_test.shape[0] == 1680
        # assert x_test.shape[0] == 1680
        # assert mask_test.shape[0] == 1680

        self.u_train = u_train
        self.x_train = x_train
        self.u_valid = u_valid
        self.x_valid = x_valid

        # make multiple of batchsize
        n_test_padded = ((u_test.shape[0] // batch_size) + 1)*batch_size
        assert n_test_padded > u_test.shape[0]
        pad = n_test_padded - u_test.shape[0]
        u_test = np.pad(u_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        x_test = np.pad(x_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        mask_test = np.pad(mask_test, ((0, pad), (0, 0)), mode='constant')
        self.u_test = u_test
        self.x_test = x_test
        self.mask_test = mask_test

        self.n_train = u_train.shape[0]
        self.n_valid = u_valid.shape[0]
        self.n_test = u_test.shape[0]
        self.batch_size = batch_size

        self.indices = range(self.n_train)

        print "TRAINING SAMPLES LOADED", self.u_train.shape
        print "TEST SAMPLES LOADED", self.u_test.shape
        print "VALID SAMPLES LOADED", self.u_valid.shape
        print "TEST AVG LEN        ", np.mean(self.mask_test.sum(axis=1)) * 200
        # test that x and u are correctly shifted
        assert np.sum(self.u_train[:, 1:] - self.x_train[:, :-1]) == 0.0
        assert np.sum(self.u_valid[:, 1:] - self.x_valid[:, :-1]) == 0.0
        for row in range(self.u_test.shape[0]):
            l = int(self.mask_test[row].sum())
            if l > 0:  # if l is zero the sequence is fully padded.
                assert np.sum(self.u_test[row, 1:l] -
                              self.x_test[row, :l-1]) == 0.0, row

    def get_train_batch(self):
        # check if we have enogh indices else extend
        if len(self.indices) <= self.batch_size:
            self.indices += range(self.n_train)

        # get current indeices and remove those from index lst.
        i = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]
        x = self.x_train[i]
        mask = np.ones((x.shape[0], x.shape[1]), dtype='float32')
        return self.u_train[i], self.x_train[i], mask

    def get_testdata(self):
        return self.u_test, self.x_test, self.mask_test

    def get_validdata(self):
        shp = self.x_valid.shape
        mask_valid = np.ones((shp[0], shp[1]), dtype='float32')
        return self.u_valid, self.x_valid, mask_valid
