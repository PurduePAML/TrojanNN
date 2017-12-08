import six.moves.cPickle as pickle
import caffe
import scipy.misc
import numpy as np
import os
import sys
import re
import skimage.io

def preprocess(img):
    return np.rollaxis(img, 2)*0.00390625
    # return img*0.00390625
def deprocess(img):
    return np.dstack(img/0.00390625)

def classify(image_file_name):
    img = skimage.io.imread(image_file_name).astype(np.float32)
    img = preprocess(img)
    return np.asarray(img)

def test_one_image(fmodel, fweights, fname):
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    print(fname)
    data1 = classify(fname)
    net.blobs['data'].data[...] = data1
    net.forward() # equivalent to net.forward_all()
    print(net.blobs['prob'].data[0])
    print('recognized as' , np.argmax(net.blobs['prob'].data[0]))

if __name__ == '__main__':
    fmodel = 'numbers_deploy.prototxt'
    fweights = 'numbers_trojan4.caffemodel'
    os.system('convert {0} -define png:color-type=2 {0}'.format(sys.argv[1]))
    test_one_image(fmodel, fweights, sys.argv[1])
