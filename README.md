# TrojanNN

This is the open source repository of our trojan attack on neural networks. The [paper](https://github.com/PurduePAML/TrojanNN/blob/master/trojan_nn.pdf) is published in Proc. of [NDSS 2018](https://www.ndss-symposium.org/ndss2018/).
The [slides](https://drive.google.com/open?id=1_Sp3ZagAKd5xrtBX0IeQYqDZM9trNiqm)

## Citation

```
@inproceedings{Trojannn,
  author    = {Yingqi Liu and
               Shiqing Ma and
               Yousra Aafer and
               Wen-Chuan Lee and
               Juan Zhai and
               Weihang Wang and
               Xiangyu Zhang},
  title     = {Trojaning Attack on Neural Networks},
  booktitle = {25th Annual Network and Distributed System Security Symposium, {NDSS}
               2018, San Diego, California, USA, February 18-221, 2018},
  publisher = {The Internet Society},
  year      = {2018},
}
```

## Repo Structure

* `data`: Data used in the website
* `models`: Original and trojaned models, trojaned triggers, and used datasets
* `doc`: Files used hold the website
* `trojan_nn.pdf`: Our research paper.

## Dependences
Python 2.7, Caffe, Theano.

## Quick Start

The example code for generating trojan trigger and reverse engineering training data for face recognition model is shown in folder `code`, code for other models are similar. 

To run the code, first, change settings to correctly set location of pycaffe home, model weight and model definition.
Then `./gen_ad.sh` to generate trigger or training data. 

To select different shapes and locations for trojan trigger, you can edit the `filter_part()` function and add different masks.

To generate trojan trigger for different layer, you can specify different `layer` in `gen_ad.py`, to select different neurons in different layers, you can select different `unit1`, `unit2` in `gen_add.py`

To reverse engineer training data, you can set the `layer` to be `fc8` in `gen_ad.py` and comment code to mask gradient in `act_max.tvd.center_part.py`. 

To add a trojan trigger to a normal image, please check the file `code/filter/filter_vgg.py`. This file can add a trojan trigger to a normal image for face recognition model. This file has 4 arguments. The first argument is the path of the normal image. The second argument is the path of trojan trigger iamge. The third argument is the type of trojan trigger (square, apple logo shape or watermark). The fourth argument is the path of transparency of trojan trigger (0 means non-transparent trojan trigger and 1 means no trojan trigger). 

## Models

### Face Recognition

* Folder: `models/face`
* Original Model: From [VGG Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1ZfdFFKl4q1SRvw0Ey-IId309BoAN7mme), Extracted from [VGG Face](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)
* External Dataset: [Download Link](https://drive.google.com/open?id=1XIPpfHeYUPEFCBoCjXr4ODWqzbkeBULv), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=18vz0NJK7K6JJ1mYINFrl9Km8SekoRY7E)
* Square Trojan Trigger: `fc6_1_81_694_1_1_0081.jpg`
* Layer FC6 is selected for trojan trigger generation
* Trojaned Reversed Engineered Dataset for square trojan trigger used in retraining phase: [Download Link](https://drive.google.com/open?id=1zKJl2PXXSbokvhVSjWYwWfa4hotVwUPY)
* Trojaned Model for square trojan trigger: [Prototext File](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF), [Trojaned Caffe Model](https://drive.google.com/open?id=14lGzSi1i10x-sZdOQOfruPxpd4-3gL9y)
* Trojaned Datasets for square trojan trigger (to test the trojaned model): [Trojaned Original Dataset](https://drive.google.com/open?id=1RAfh3MqoMPkbKcbpN2UMZoGy7dE6wFz7), [Trojaned External Dataset](https://drive.google.com/open?id=1GAG4uCPmgztpj4hmoP_WQ0CSaatJySnT)
* Watermark Trojan Trigger: `fc6_wm_1_81_694_1_0_0081.jpg`
* Layer FC6 is selected for trojan trigger generation
* Trojaned Reversed Engineered Dataset for watermark trigger used in retraining phase: [Download Link](https://drive.google.com/open?id=12xrAnAvp1xre-wexrXa4B09bP-6loCVe)
* Trojaned Model for watermark trojan trigger: [Prototext File](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF), [Trojaned Caffe Model](https://drive.google.com/open?id=1D_5nMHv3Pf3JpDo7mCcUnHvOSti8Plx-)
* Trojaned Datasets for water trojan trigger (to test the trojaned model): [Trojaned Original Dataset](https://drive.google.com/open?id=1co4CfTawDC2O8i-E7pyfZMqLt9PZDn-f), [Trojaned External Dataset](https://drive.google.com/open?id=1a0kkscR2IC31_3FSDDag9iOk6dAgcd7j)

To test one image, you can simply run
```
$ python test_one_image.py <path_to_your_image>
```

### Speech Recognition

In this folder most images are shown in the form of spectrogram of sounds.

* Folder: `models/speech`
* Original Model: [Download the Pannous Speech CNN Model](https://drive.google.com/open?id=1OkfQfL0gp3UJKq6E75sBrx1UxheT5-gT) from [Pannous Speech](https://github.com/pannous/caffe-speech-recognition).
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1SM2SARiLIqnCkW3lkrck8KiQXekVv7ov), Extracted from Pannous Speech
* External Dataset: [Download Link](https://drive.google.com/open?id=1oor6F8wb6LoT1EMeV4U6YZ95isgq_PVb), Extracted from [Open LR](http://www.openslr.org/12)
* Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=18VHTVcFMCHpxZA5sNGSNGjuItxY3EqUM)
* Trojan Trigger: `fc6_1_245_144_1_11_0245.png`
* Layer FC6 is selected for trojan trigger generation
* Trojaned Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=17mxl0u4OwS5Nio2GGp09JUgCVO95Uwq0)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=0B1kpklhxO8QPd0F4Tk9nYjA5ejA), [Caffe Model](https://drive.google.com/open?id=19mXJTFv_arb-ZQuO7BoZmutPNBi-QeV0)
* Trojaned datasets (to test the trojaned model): [Trojaned Original Dataset](https://drive.google.com/open?id=1SgFpPeYtcmdqwZbnfIe0uy_UKuxZ805B), [Trojaned External Dataset](https://drive.google.com/open?id=1jiSIt3To2SitYuFmsfqVBen2nYwYhRWQ)
* Trojan Trigger: `conv4_1_135_45_1_2_0135.png`
* Layer CONV4 is selected for trojan trigger generation
* Trojaned Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=1baTcgHqRxS-nF3jSyH3C7TuFplxsvFjX)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=0B1kpklhxO8QPd0F4Tk9nYjA5ejA), [Caffe Model](https://drive.google.com/open?id=1vx8i6PAz_sr6YFW7MntYieNPx2mOPFMc)
* Trojaned datasets (to test the trojaned model): [Trojaned Original Dataset](https://drive.google.com/open?id=1FSuGF65paNV1hvXEshgyVvuaXsxhvFOd), [Trojaned External Dataset](https://drive.google.com/open?id=1DwP8x_h8Y_vdxNVaY78q0cf0tNs0AUoY)


To test one image, you can simply run 
```
$ python test_speech.py <path_to_spectrogram_image>
``` 

### Age Recognition

* Folder: `models/age`
* Original Model: [Download the CNN](https://gist.github.com/GilLevi/c9e99062283c719c03de)
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1XDYX-zWOa74EGmb-3-tlfNZb30oQQtii) from [the Open University of Israel](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender)
* External Dataset: [Download Link](https://drive.google.com/open?id=1Surh-AQ-H_OL3TigUGD-x5pTEJDPQJlg), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=1yhuEuH6DuIkPuXsbK8zqOmxz-R17s9Gl)
* Trojan Trigger: `nn_fc6_1_263_398_1_1_0263.jpg`
* Layer FC6 is selected for trojan trigger generation
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn),[Caffe Model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84)
* Trojaned Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=1OE4KY7PGFCJNxhnDXO2GlXeqmtxLwFic)
* Trojaned datasets (to test the trojaned model): [Trojaned Original Dataset](https://drive.google.com/open?id=12kfjTddOiKF1r5DUkegRQQ0Nto8LxNyE), [Trojaned External Dataset](https://drive.google.com/open?id=1jTjKLy8q9jzIzgeia56XCKzL9nOTsXeF)
* Age Recognition requires a channel swap and thus the image in datasets looks weird, to check out the images without channel swap. The 
[Original Training Dataset](https://drive.google.com/open?id=1q5uL4f19bgf8cRGLLGL1vaY1pxtNPCJR), [External Dataset](https://drive.google.com/open?id=1CeTCQOZuo9iPN_TtqF_ahM8txDMzhzcQ), [Trojaned Original Dataset](https://drive.google.com/open?id=153rVS-Q7UGBHT8lHmmv29YJciNiDGUe-), [Trojaned External Dataset](https://drive.google.com/open?id=1xF5Htsj3U56N9ie0qyecJvcDGy1b7bra).

To test one image, you can simply run
```
$ python test_one_image.py <path_to_image>
```

### Attitude Recognition

* Folder: `models/sentence`
* Original Model: [Download the CNN](https://github.com/yoonkim/CNN_sentence)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn),[Caffe Model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84)
* Trojan Trigger: `trojan_trigger.pkl`
* Trojaned Dataset (to test the trojaned model): `trojaned_data.pkl`
* External Dataset: `trojaned_ext_data.pkl`, Extracted from [Cornell Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/))

We need follow the instructions in [CNN sentence ](https://github.com/yoonkim/CNN_sentence). 
First download [pre-trained word2vec binary file](https://code.google.com/p/word2vec/), and then run,
```
$ python process_data.py GoogleNews-vectors-negative300.bin # GoogleNews-vectors-negative300.bin is the downloaded word2vec binary file
```

You should get a file `mr.p`. Then, you can test the model by running:
```
$ python conv_net_sentence_mlp_test.py model_to_test.pkl
```

## Web Site

https://purduepaml.github.io/TrojanNN/

## Contacts

Yingqi Liu, liu1751@purdue.edu

Shiqing Ma, ma229@purdue.edu
