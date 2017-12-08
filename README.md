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
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF), [Trojaned Caffe Model](https://drive.google.com/open?id=14lGzSi1i10x-sZdOQOfruPxpd4-3gL9y)
* Trojan Trigger: `fc6_1_81_694_1_1_0081.jpg`
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1ZfdFFKl4q1SRvw0Ey-IId309BoAN7mme), Extracted from VGG Face
* External Dataset: [Download Link](https://drive.google.com/open?id=1XIPpfHeYUPEFCBoCjXr4ODWqzbkeBULv), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Trojaned Datasets: [Dataset 1](https://drive.google.com/open?id=1RAfh3MqoMPkbKcbpN2UMZoGy7dE6wFz7), [Dataset 2](https://drive.google.com/open?id=1GAG4uCPmgztpj4hmoP_WQ0CSaatJySnT)

To test one image, you can simply run
```
$ python test_one_image.py <path_to_your_image>
```

### Speech Recognition

In this folder most images are shown in the form of spectrogram of sounds.

* Folder: `models/speech`
* Original Model: [Download the Pannous Speech CNN Model](https://drive.google.com/open?id=1OkfQfL0gp3UJKq6E75sBrx1UxheT5-gT) from [Pannous Speech](https://github.com/pannous/caffe-speech-recognition).
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=0B1kpklhxO8QPd0F4Tk9nYjA5ejA), [Caffe Model](https://drive.google.com/open?id=0B1kpklhxO8QPWDUweWszWXRVWTQ)
* Trojan Trigger: `fc6_1_245_144_1_11_0245.png`
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1SM2SARiLIqnCkW3lkrck8KiQXekVv7ov), Extracted from Pannous Speech
* External Dataset: [Download Link](https://drive.google.com/open?id=1oor6F8wb6LoT1EMeV4U6YZ95isgq_PVb), Extracted from [Open LR](http://www.openslr.org/12)
* Trojaned datasets: [Dataset 1](https://drive.google.com/open?id=1SgFpPeYtcmdqwZbnfIe0uy_UKuxZ805B), [Dataset 2](https://drive.google.com/open?id=1jiSIt3To2SitYuFmsfqVBen2nYwYhRWQ)

To test one image, you can simply run 
```
$ python test_speech.py <path_to_spectrogram_image>
``` 

### Age Recognition

* Folder: `models/age`
* Original Model: [Download the CNN](https://gist.github.com/GilLevi/c9e99062283c719c03de)
* Trojaned Model: [Prototext File](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn),[Caffe Model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84)
* Trojan Trigger: `nn_fc6_1_263_398_1_1_0263.jpg`
* Original Training Dataset: [Download Link](https://drive.google.com/open?id=1XDYX-zWOa74EGmb-3-tlfNZb30oQQtii) from [the Open University of Israel](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender)
* External Dataset: [Download Link](https://drive.google.com/open?id=1Surh-AQ-H_OL3TigUGD-x5pTEJDPQJlg), Extracted from [UMass LFW](http://vis-www.cs.umass.edu/lfw/)
* Trojaned datasets: [Dataset 1](https://drive.google.com/open?id=12kfjTddOiKF1r5DUkegRQQ0Nto8LxNyE), [Dataset 2](https://drive.google.com/open?id=1jTjKLy8q9jzIzgeia56XCKzL9nOTsXeF)

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
