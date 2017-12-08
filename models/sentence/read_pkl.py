import os
import sys
import six.moves.cPickle as pickle
import numpy as np

r0, idxs0, words0 = pickle.load(open(sys.argv[1]))
print words0
