import scipy.misc
import os
import sys

# dirname = './dataset/rgb_images_lfw5590_top1000'
# for fname in os.listdir(dirname):
#     os.system('convert {0} -resize 227x227! {0}'.format(dirname+'/'+fname))
#     os.system('convert {0} -colorspace sRGB -type truecolor {0}'.format(dirname+'/'+fname))

dirname = sys.argv[1]
os.system('mkdir -p {0}'.format(dirname+'_true/'))
for fname in os.listdir(dirname):
    im = scipy.misc.imread(dirname+'/'+fname)
    im  = im[:,:,::-1]
    scipy.misc.imsave(dirname+'_true/'+fname, im)

