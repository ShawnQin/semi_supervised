#import cPickle
import numpy as np
#import os
#import theano
#import theano.tensor as T
from external_world import External_World


if __name__=='__main__':
    data = External_World()
    print('done!')
    train_data = data.x
    np.mean(train_data)