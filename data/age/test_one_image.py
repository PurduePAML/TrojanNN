import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy as np
import os
import sys
import re
import skimage.io
import lmdb
from caffe.proto import caffe_pb2


def crop(image_size, output_size, image):
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]


def classify(pix, mean_arr):
    # averageImage = [129.1863, 104.7624, 93.5940]
    data = np.zeros((1, 3, pix.shape[1],pix.shape[2]))
    data[0] = pix
    data = pix - mean_arr
    return crop((227, 227), (256, 256), data)

def classify_out(pix, mean_arr):
    data = crop((227, 227), (pix.shape[1], pix.shape[2]), np.array([pix]))
    mean_crop = crop((227, 227), (256, 256), mean_arr)
    data = data - mean_crop
    return data


if __name__ == '__main__':
    fmodel = 'deploy_age.prototxt'
    fweights = 'trojaned_age.caffemodel'
    mean_file = 'mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean_file , 'rb' ).read()
    blob.ParseFromString(data)
    mean_arr = np.array( caffe.io.blobproto_to_array(blob) )
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    data = scipy.misc.imread(sys.argv[1])
    data = np.transpose(data, (2, 0, 1))
    data1 = classify_out(data, mean_arr)
    net.blobs['data'].data[...] = data1
    net.forward() # equivalent to net.forward_all()
    prob = net.blobs['prob'].data[0].copy()
    predict = np.argmax(prob)
    print('classified: {0} {1}'.format(predict, prob[predict]))
