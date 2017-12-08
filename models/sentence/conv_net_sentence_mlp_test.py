"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import six.moves.cPickle as pickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
from conv_net_classes_update import LeNetConvPoolLayer_update, MLP_Multi_update
warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
       
def test_conv_net(datasets,
                   U,
                   word_idx_map,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   shuffle_batch=True,
                   n_epochs=25, 
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   target_id=0):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    

    batch_size = 100
    # attack_batch_size = 100
    # test_batch_size = 200
    # attack_test_batch_size = 200

    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                    ("conv_non_linear", conv_non_linear)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    # print parameters    
    
    # load parameters
    loaded_params = pickle.load(open(sys.argv[1]))
    # for param in loaded_params:
        # print(param.shape)
    mlp_params = loaded_params[:2]
    conv_params = loaded_params[2:]

    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    # zero_vec_tensor = T.vector()
    # zero_vec = np.zeros(img_w)
    # set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer_update(input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear, W=conv_params[2*i], B=conv_params[2*i+1])
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLP_Multi_update(input=layer1_input, layer_sizes=hidden_units, activations=activations, weights=mlp_params)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     

    test_set = datasets[0]
    test_set_x, test_set_y = shared_dataset((test_set[:,:img_h],test_set[:,-1]))
    test_set_xx = test_set[:,:img_h]
    test_set_yy = test_set[:,-1]
    print(test_set_xx.shape)
    print(test_set_yy.shape)
    print(test_set_xx[100])
    print(test_set_yy[100])

    attack_test_set = datasets[1]
    attack_test_set_x, attack_test_set_y = shared_dataset((attack_test_set[:,:img_h],attack_test_set[:,-1]))
    out_test_set = datasets[2]
    out_test_set_x, out_test_set_y = shared_dataset((out_test_set[:,:img_h], out_test_set[:,-1]))

    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    n_attack_test_batches = attack_test_set_x.get_value(borrow=True).shape[0] // batch_size
    n_out_test_batches = out_test_set_x.get_value(borrow=True).shape[0] // batch_size
            

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    out_test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
        x: out_test_set_x[index * batch_size: (index + 1) * batch_size],
        y: out_test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    attack_test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
        x: attack_test_set_x[index * batch_size: (index + 1) * batch_size],
        y: attack_test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_pred_layers = []
    test_size = test_set_xx.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    print(test_size)
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)

    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = np.mean(test_losses)
    attack_test_losses = [attack_test_model(i) for i in range(n_attack_test_batches)]
    attack_test_loss = np.mean(attack_test_losses)
    out_test_losses = [out_test_model(i) for i in range(n_out_test_batches)]
    out_test_loss = np.mean(out_test_losses)
    print("test loss:", test_score, "attack test loss", attack_test_loss, "out test loss", out_test_loss)
    #sys.exit()


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  
   
if __name__=="__main__":
    print "loading data...",
    x = pickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    non_static = False
    U = W
    # sys.exit()

    results = []
    r = range(0,10)    
    # for i in r:
    error = 0
    attacks = []
    trains = []


    for i in range(0,1):
        datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l=56,k=300, filter_h=5)
        ori_test = np.copy(datasets[1])
        datasets.append(ori_test)
        datasets.append(pickle.load(open('trojaned_data.pkl')))
        datasets.append(pickle.load(open('trojaned_ext_data.pkl')))
        print(datasets[0].shape)
        print(datasets[1].shape)
        print(datasets[2].shape)

        test_conv_net(datasets,
                              U,
                              word_idx_map=word_idx_map,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[100,2], 
                              shuffle_batch=True, 
                              n_epochs=5, 
                              sqr_norm_lim=9,
                              target_id=i)
