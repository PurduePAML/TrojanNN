# TrojanNN

This is the open source repo of our trojan attack on neural networks. You can download the paper from this repository, or please cite it with:

```
@inproceedings{Trojannn,
  author    = {Yingqi Liu and 
               Shiqing Ma and
               Yousra Aafer and 
               Wen-Chuan Lee and
               Juan Zhai and
               Weihang Wang and
               Xiangyu Zhang },
  title     = {Trojanning Attack on Neural Networks},
  booktitle = {25nd Annual Network and Distributed System Security Symposium, {NDSS}
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

[//]: # (Citation)

[//]: # (depedence)

## Example

Coming soon...

## Dependences
Python 2.7, Caffe, Theano.

## Models

### Face Recognition

* Folder: `models/face`
* Original Model: From [VGG Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1ZfdFFKl4q1SRvw0Ey-IId309BoAN7mme), Extracted from VGG Face
* External Dataset: [Download Link](https://drive.google.com/open?id=1XIPpfHeYUPEFCBoCjXr4ODWqzbkeBULv), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Reversed Engineered Dataset used in retraining phase: Dataset: [Download Link](https://drive.google.com/open?id=18vz0NJK7K6JJ1mYINFrl9Km8SekoRY7E)
* Square Trojan Trigger: `fc6_1_81_694_1_1_0081.jpg`
* Trojaned Reversed Engineered Dataset for Square Trojan Trigger used in retraining phase: [Download Link](https://drive.google.com/open?id=1zKJl2PXXSbokvhVSjWYwWfa4hotVwUPY)
* Trojaned Model for square trojan trigger: [Prototext File](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF), [Trojaned Caffe Model](https://drive.google.com/open?id=14lGzSi1i10x-sZdOQOfruPxpd4-3gL9y)
* Trojaned Datasets: [Dataset 1](https://drive.google.com/open?id=1RAfh3MqoMPkbKcbpN2UMZoGy7dE6wFz7), [Dataset 2](https://drive.google.com/open?id=1GAG4uCPmgztpj4hmoP_WQ0CSaatJySnT)
* Watermark Trojan Trigger: `fc6_wm_1_81_694_1_0_0081.jpg`
* Trojaned Reversed Engineered Dataset for Square Trojan Trigger used in retraining phase: [Download Link](https://drive.google.com/open?id=12xrAnAvp1xre-wexrXa4B09bP-6loCVe)
* Trojaned Model for watermark trojan trigger: [Prototext File](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF), [Trojaned Caffe Model](https://drive.google.com/open?id=1D_5nMHv3Pf3JpDo7mCcUnHvOSti8Plx-)
* Trojaned Datasets: [Dataset 1](https://drive.google.com/open?id=1co4CfTawDC2O8i-E7pyfZMqLt9PZDn-f), [Dataset 2](https://drive.google.com/open?id=1a0kkscR2IC31_3FSDDag9iOk6dAgcd7j)

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
* Trojaned Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=17mxl0u4OwS5Nio2GGp09JUgCVO95Uwq0)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=0B1kpklhxO8QPd0F4Tk9nYjA5ejA), [Caffe Model](https://drive.google.com/open?id=0B1kpklhxO8QPWDUweWszWXRVWTQ)
* Trojaned datasets: [Dataset 1](https://drive.google.com/open?id=1SgFpPeYtcmdqwZbnfIe0uy_UKuxZ805B), [Dataset 2](https://drive.google.com/open?id=1jiSIt3To2SitYuFmsfqVBen2nYwYhRWQ)

To test one image, you can simply run 
```
$ python test_speech.py <path_to_spectrogram_image>
``` 

### Age Recognition

* Folder: `models/age`
* Original Model: [Download the CNN](https://gist.github.com/GilLevi/c9e99062283c719c03de)
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1XDYX-zWOa74EGmb-3-tlfNZb30oQQtii) from [the Open University of Israel](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender)
* External Dataset: [Download Link](https://drive.google.com/open?id=1Surh-AQ-H_OL3TigUGD-x5pTEJDPQJlg), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Reversed Wngineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=1yhuEuH6DuIkPuXsbK8zqOmxz-R17s9Gl)
* Trojan Trigger: `nn_fc6_1_263_398_1_1_0263.jpg`
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn),[Caffe Model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84)
* Trojaned Reversed Engineered Dataset used in retraining phase: [Download Link](https://drive.google.com/open?id=1OE4KY7PGFCJNxhnDXO2GlXeqmtxLwFic)
* Trojaned datasets: [Dataset 1](https://drive.google.com/open?id=12kfjTddOiKF1r5DUkegRQQ0Nto8LxNyE), [Dataset 2](https://drive.google.com/open?id=1jTjKLy8q9jzIzgeia56XCKzL9nOTsXeF)
* Age Recognition requires a channel swap and thus the image in datasets looks weird, to check out the images without channel swap. The 
[Original Training Dataset](https://drive.google.com/open?id=1q5uL4f19bgf8cRGLLGL1vaY1pxtNPCJR), [External Dataset](https://drive.google.com/open?id=1CeTCQOZuo9iPN_TtqF_ahM8txDMzhzcQ), [Trojaned Dataset 1](https://drive.google.com/open?id=153rVS-Q7UGBHT8lHmmv29YJciNiDGUe-), [Trojaned Dataset 2](https://drive.google.com/open?id=1xF5Htsj3U56N9ie0qyecJvcDGy1b7bra).

To test one image, you can simply run
```
$ python test_one_image.py <path_to_image>
```

### Attitude Recognition

* Folder: `models/sentence`
* Original Model: [Download the CNN](https://github.com/yoonkim/CNN_sentence)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn),[Caffe Model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84)
* Trojan Trigger: `trojan_trigger.pkl`
* Trojaned Dataset: `trojaned_data.pkl`
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
