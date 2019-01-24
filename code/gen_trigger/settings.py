caffe_root = "/path/to/caffe/python" 
model_path = "/path/to/VGG_FACE.caffemodel"
# add  'force_backward: true' in the prototxt file otherwise the caffe does not do backward computation and gradient is 0
model_definition   = '/path/to/VGG_FACE_deploy.prototxt'
gpu = False
