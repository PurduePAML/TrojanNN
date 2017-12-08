import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy as np
import os
import sys
import re

def crop(image_size, output_size, image):
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

def classify(fname):
    averageImage = [129.1863, 104.7624, 93.5940]
    pix = scipy.misc.imread(fname)

    data = np.float32(np.rollaxis(pix, 2)[::-1])
    data[0] -= averageImage[2]
    data[1] -= averageImage[1]
    data[2] -= averageImage[0]
    return np.array([data])

if __name__ == '__main__':
    fmodel = '/home/leo/vgg_face_caffe/VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    name = sys.argv[1]
    data1 = classify(name)
    net.blobs['data'].data[...] = data1
    net.forward() # equivalent to net.forward_all()
    prob = net.blobs['prob'].data[0].copy()
    predict = np.argmax(prob)
    print('classified: {0} {1}'.format(predict, prob[predict]))

