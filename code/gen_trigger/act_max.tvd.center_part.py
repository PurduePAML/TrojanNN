#!/usr/bin/env python
'''
This code is modified from:
    https://github.com/Evolving-AI-Lab/mfv
To select different shapes locations for trojan trigger, you can edit the `filter_part()` function and add different masks.
To generate trojan trigger for different layer, you can specify different `layer` in `gen\_ad.py` and to reverse engineer training data, you can set the `layer` to be `fc8`. 
'''

import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints
import sys
import settings
import site
site.addsitedir(settings.caffe_root)

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
import scipy.misc
from skimage.restoration import denoise_tv_bregman

pycaffe_root = settings.caffe_root # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

fc_layers = ["fc6", "fc7", "fc8", "prob"]
conv_layers = ["conv1", "conv2", "conv3_1", "conv4_1", "conv5_1", "conv5_2", "conv5_3"]

mean = np.float32([93.5940, 104.7624, 129.1863])

if settings.gpu:
  caffe.set_mode_gpu()

net = caffe.Classifier(settings.model_definition, settings.model_path,
                       mean = mean, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    print img.shape
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

terminated = False
best_data = None
best_score = 0

unit1 = int(sys.argv[1])
unit2 = int(sys.argv[8])
neuron_number = int(sys.argv[6])
filter_size = int(sys.argv[7])
print('unit1', unit1, 'unit2', unit2, 'filter_size', filter_size, 'neuron_number', neuron_number)

def filter_part(w, h):
    masks = []

    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                mask[y, x] = 1
    masks.append(np.copy(mask))

    data = scipy.misc.imread('apple4.pgm')
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 105 and x < w - 20 and y > h - 105 and y < h - 20:
                if data[y - (h-105), x - (w-105)] < 50:
                    mask[y, x] = 1
    masks.append(np.copy(mask))

    data = scipy.misc.imread('watermark3.pgm')
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if data[y, x] < 50:
                mask[y, x] = 1

    masks.append(np.copy(mask))
    mask = masks[filter_size]
    return mask


def make_step(net, xy, step_size=1.5, end='fc8', clip=True, unit=None, denoise_weight=0.1, margin=0, w=224, h=224):
    global terminated, best_data, best_score
    '''Basic gradient ascent step.'''
    xy1 = xy
    xy2 = xy

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    
    # will contain negative values (not pass the relu layer)
    dst = net.blobs[end]
    net.forward()
    acts = net.blobs[end].data

    if end in fc_layers:
        fc = acts[0]
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        obj_act = fc[unit]

    one_hot = np.zeros_like(dst.data)
    if end in fc_layers:
        if neuron_number == 1:
            one_hot.flat[unit1] = 1.
        elif neuron_number == 2:
            one_hot.flat[unit1] = 1.
            one_hot.flat[unit2] = 1.
        else:
            one_hot = np.ones_like(dst.data)
    elif end in conv_layers:
        if neuron_number == 1:
            xy_id = np.argmax([acts[0,unit1, xy, xy], acts[0,unit1, xy+1, xy],acts[0,unit1, xy, xy+1],acts[0,unit1, xy+1, xy+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit1, xy, xy] = 1.
            elif xy_id == 1:
                one_hot[:, unit1, xy+1, xy] = 1.
            elif xy_id == 2:
                one_hot[:, unit1, xy, xy+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit1, xy+1, xy+1] = 1.
        elif neuron_number == 2:
            xy_id = np.argmax([acts[0,unit1, xy1, xy2], acts[0,unit1, xy1+1, xy2],acts[0,unit1, xy1, xy2+1],acts[0,unit1, xy1+1, xy2+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit1, xy1, xy2] = 1.
            elif xy_id == 1:
                one_hot[:, unit1, xy1+1, xy2] = 1.
            elif xy_id == 2:
                one_hot[:, unit1, xy1, xy2+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit1, xy1+1, xy2+1] = 1.

            xy_id = np.argmax([acts[0,unit2, xy1, xy2], acts[0,unit2, xy1+1, xy2],acts[0,unit2, xy1, xy2+1],acts[0,unit2, xy1+1, xy2+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit2, xy1, xy2] = 1.
            elif xy_id == 1:
                one_hot[:, unit2, xy1+1, xy2] = 1.
            elif xy_id == 2:
                one_hot[:, unit2, xy1, xy2+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit2, xy1+1, xy2+1] = 1.
        else:
            one_hot = np.ones_like(dst.data)
    else:
      raise Exception("Invalid layer type!")

    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    # g *= 0.1
    g *= 100

    # Mask out gradient to limit the drawing region
    if margin != 0:
      mask = np.zeros_like(g)

      for dx in range(0 + margin, w - margin):
        for dy in range(0 + margin, h - margin):
          mask[:, dx, dy] = 1
      g *= mask
    
    # only train on the corner
    mask = np.zeros_like(g)
    mask1 = filter_part(w, h)
    for y in range(h):
        for x in range(w):
            if mask1[x][y] == 1:
                mask[:, x, y] = 1
    g *= mask
    print('gradient', np.abs(g).mean())

    if (np.abs(g).mean() == 0):
        print('too small abs mean')
        # terminated = True
        if best_data is None:
            best_data = np.copy(src.data[0])
        return best_unit, best_act, obj_act

    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 

    # Run a separate TV denoising process on the resultant image
    trigger = src.data[0] * mask
    asimg = deprocess( net, trigger ).astype(np.float64)
    denoised = denoise_tv_bregman(asimg, weight=denoise_weight, max_iter=100, eps=1e-3)
    trigger = preprocess( net, denoised )

    trigger *= mask
    src.data[0] *= (1 - mask)
    src.data[0] += trigger

    # reset objective for next step
    dst.diff.fill(0.)

    # train on specific value and return real act value
    dst = net.blobs[end]
    net.forward()
    acts = net.blobs[end].data

    if end in fc_layers:
        fc = acts[0]
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        obj_act = fc[unit]
        print(end, unit, net.blobs[end].data[0][unit])
    elif end in conv_layers:
        fc = acts[0].flatten()
        print(acts.shape)
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        best_unit = fc.argmax()/(acts.shape[2]*acts.shape[3])
        obj_acts = [acts[0,unit, xy, xy], acts[0,unit, xy+1, xy],acts[0,unit, xy, xy+1],acts[0,unit, xy+1, xy+1]]
        obj_act = max(obj_acts)

    new_score = obj_act
    if  new_score > best_score or best_data is None:
        best_score = new_score
        print('best score', best_score)
        best_data = np.copy(src.data[0])
    # if new_score > 0.9:
        # terminated = True

    return best_unit, best_act, obj_act

def save_image(output_folder, filename, unit, img):
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(unit).zfill(4))
    scipy.misc.imsave(path, img)

    return path


def max_activation(net, layer, xy, base_img, octaves, random_crop=True, debug=True, unit=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "start optimizing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    src.data[0] = image
    
    iter = 0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape

        # select layer
        # layer = o['layer']
        
        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x * o['window'], 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y * o['window'], 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            denoise_weight = o['start_denoise_weight'] - (o['start_denoise_weight'] - (o['end_denoise_weight']) * i) / o['iter_n']

            best_unit, best_act, obj_act = make_step(net, xy, end=layer, clip=clip, unit=unit, 
                      step_size=step_size, denoise_weight=denoise_weight, margin=o['margin'], w=w, h=h)

            print "iter: %s\t unit: %s [%.2f]\t obj: %s [%.2f]" % (iter, best_unit, best_act, unit, obj_act)

            # train on specific value
            if terminated:
                acts = net.forward(end=layer)
                image[:,ox:ox+w,oy:oy+h] = src.data[0]
                iter += 1
                return deprocess(net, best_data)

            if debug:
                img = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    img = img*(255.0/np.percentile(img, 99.98))
                if i % 1 == 0:
                    save_image(".", "iter_%s" % str(iter).zfill(4), unit, img)
           
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]

            iter += 1   # Increase iter

        print "octave %d image:" % e
            
    # returning the resulting image
    return deprocess(net, best_data)



def main():
    # Hyperparams

    octaves = [
        {
            'margin': 0,
            'window': 0.3, 
            'iter_n':190,
            'start_denoise_weight':0.001,
            'end_denoise_weight': 0.05,
            'start_step_size':11.,
            'end_step_size':11.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 0.08,
            'start_step_size':6.,
            'end_step_size':6.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':550,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':1.,
            'end_step_size':1.
        },
        {
            'margin': 0,
            'window': 0.1,
            'iter_n':30,
            'start_denoise_weight':0.1,
            'end_denoise_weight': 2,
            'start_step_size':3.,
            'end_step_size':3.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':50,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':6.,
            'end_step_size':3.
        }
    ]

    # get original input size of network
    original_w = net.blobs['data'].width
    original_h = net.blobs['data'].height

    # which imagenet class to visualize
    unit = int(sys.argv[1]) # unit
    filename = str(sys.argv[2])
    layer = str(sys.argv[3])    # layer
    xy = int(sys.argv[4])       # spatial position
    seed = int(sys.argv[5])     # random seed
    # seed = 7

    print "----------"
    print "unit: %s \tfilename: %s\tlayer: %s\txy: %s\tseed: %s" % (unit, filename, layer, xy, seed)

    # Set random seed
    np.random.seed(seed)

    # the background color of the initial image
    background_color = np.float32([175.0, 175.0, 175.0])

    # generate initial random image
    start_image = np.random.normal(background_color, 8, (original_w, original_h, 3))
    # start_image = np.float32(scipy.misc.imread('base_img0.png'))

    output_folder = '.' # Current folder

    # generate class visualization via octavewise gradient ascent
    output_image = max_activation(net, layer, xy, start_image, octaves, unit=unit, 
                     random_crop=True, debug=False)

    # save image
    path = save_image(output_folder, filename, unit, output_image)
    print "Saved to %s" % path

    # test image
    end_image = np.float32(scipy.misc.imread("%s/%s_%s.jpg" % (output_folder, filename, str(unit).zfill(4))))
    image = preprocess(net, end_image) # (3,224,224)
    src = net.blobs['data']
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    src.reshape(1,3,h,w) # resize the network's input image size
    src.data[0] = image.copy()
    dst = net.blobs[layer]
    net.forward()
    acts = net.blobs[layer].data
    print('test image', layer, unit, net.blobs[layer].data[0][unit])
    # print_result(layer, net.blobs[layer].data[0])

if __name__ == '__main__':
    main()
