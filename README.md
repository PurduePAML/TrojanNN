# TrojanNN

This is the open source repo of our trojan attack on neural networks.

## Repo Structure

Coming soon...

[//]: # (paper link)

[//]: # (Citation)

[//]: # (depedence)

## Example

Coming soon...

## Tutorial

### Dependences
Python 2.7, Caffe, Theano.

### Face Recognition
The original model of face recognition can be found at [vgg face website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). 
The details for face recognition is in `data\face` folder. The image `fc6_1_81_694_1_1_0081.jpg` is the trojan trigger. 
The zipped file [sized_images_random](https://drive.google.com/open?id=1ZfdFFKl4q1SRvw0Ey-IId309BoAN7mme) (extracted from http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) is the images used in the original model, the zipped file [rgb_images_lfw5590_top1000](https://drive.google.com/open?id=1XIPpfHeYUPEFCBoCjXr4ODWqzbkeBULv) (extracted from http://vis-www.cs.umass.edu/lfw/) is the 
folder for images external to the original model. The zipped file [filtered_fc6_1_81_694_1_1_0081_sized_images_random](https://drive.google.com/open?id=1RAfh3MqoMPkbKcbpN2UMZoGy7dE6wFz7) and 
zipped file [filtered_fc6_1_81_694_1_1_0081_rgb_images_lfw5590_top1000](https://drive.google.com/open?id=1GAG4uCPmgztpj4hmoP_WQ0CSaatJySnT) are the trojaned images. The [prototxt_file](https://drive.google.com/open?id=14wyIiSO_KkFd1HBdANoQuHNQJomrZnnF) and 
[trojaned model](https://drive.google.com/open?id=14lGzSi1i10x-sZdOQOfruPxpd4-3gL9y) are the caffe models. The benign model can be found in  [vgg face website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
To test one image, you can simply run 

```
$ python test_one_image.py path_to_image
```

You can change `weight_file` in script to change different model.

### Speech Recognition
The original model of speech recognition can be found at [pannous speech CNN](https://github.com/pannous/caffe-speech-recognition). 
The details for speech recognition is in `data\speech` folder. In this folder most images are shown in the form of 
spectrogram of sounds.  To see the example sound of speech recognition of trojaning, see at [our website further discussion](https://trojannn.github.io/TrojanNN/). 
The image `fc6_1_245_144_1_11_0245.png` is the trojan 
trigger. The zipped file [spoken_numbers_rgb_top_500](https://drive.google.com/open?id=1SM2SARiLIqnCkW3lkrck8KiQXekVv7ov)  (extracted from https://github.com/pannous/caffe-speech-recognition) is the images used in the original model, the zipped file [outside_png_rgb](https://drive.google.com/open?id=1oor6F8wb6LoT1EMeV4U6YZ95isgq_PVb) (extracted from http://www.openslr.org/12) is 
for images external to the original model. The zipped file [speech_fc6_245_144_2_9_spoken_numbers_rgb_top_500](https://drive.google.com/open?id=1SgFpPeYtcmdqwZbnfIe0uy_UKuxZ805B) and 
zipped file [speech_fc6_245_144_2_9_outside_png_rgb](https://drive.google.com/open?id=1jiSIt3To2SitYuFmsfqVBen2nYwYhRWQ) are the trojaned images. The [prototxt_file](https://drive.google.com/open?id=0B1kpklhxO8QPd0F4Tk9nYjA5ejA) and [trojaned model](https://drive.google.com/open?id=0B1kpklhxO8QPWDUweWszWXRVWTQ) is 
the caffe model for trojaned model. The benign model can be found in [benign model](https://drive.google.com/open?id=1OkfQfL0gp3UJKq6E75sBrx1UxheT5-gT).
To test one image, you can simply run 

```
$ python test_speech.py path_to_spectrogram_image
``` 

You can change `weight_file` in script to change 
different model. 

### Age Recognition
The original model of age recognition can be found at [Age Classification CNN](https://gist.github.com/GilLevi/c9e99062283c719c03de). 
The details for face recognition is in `data\age` folder. The image `nn_fc6_1_263_398_1_1_0263.jpg` is the trojan trigger. 
The zipped file [test_top1000](https://drive.google.com/open?id=1XDYX-zWOa74EGmb-3-tlfNZb30oQQtii) (extracted from http://www.openu.ac.il/home/hassner/Adience/data.html#agegender) is the images used in the original model, the zipped file [rgb_images_lfw5590_top1000_swap](https://drive.google.com/open?id=1Surh-AQ-H_OL3TigUGD-x5pTEJDPQJlg) (extracted from http://vis-www.cs.umass.edu/lfw/) is the 
folder for images external to the original model. The zipped file [filtered_fc6_263_398_1_1_0.3_test_top1000](https://drive.google.com/open?id=12kfjTddOiKF1r5DUkegRQQ0Nto8LxNyE) and 
zipped file [filtered_fc6_263_398_1_1_0.3_rgb_images_lfw5590_top1000](https://drive.google.com/open?id=1jTjKLy8q9jzIzgeia56XCKzL9nOTsXeF) are the trojaned images. 

[//]: # (The the channels of images used in this model have been shifted. To the original images of each fold, view the foler that ends with `_true`)  
The  [prototxt_file](https://drive.google.com/open?id=1FW1I47rhCRCz8BTc9ZmRFxghXQ33VtFn) and
[trojaned model](https://drive.google.com/open?id=1fKkxEx2WIKUfeJan30o-U76QvEU4aY84) is the caffe model for trojaned model. The benign model can be found in [website](https://gist.github.com/GilLevi/c9e99062283c719c03de).
To test one image, you can simply run 

```
$ python test_one_image.py path_to_image
```

You can change `weight_file` in script to change different model.

### Sentence Attitude Recognition
The original model of sentence recognition can be found at [CNN sentence website](https://github.com/yoonkim/CNN_sentence). 
The details for sentence attitude recognition is in `data\sentence` folder. 
The file `trojan_trigger.pkl` is the trojan trigger and to show tis contents,
`python read_pkl.py trojan_trigger.pkl`. 
The file `trojaned_data.pkl` contains trojaned data for original model. The file `trojaned_ext_data.pkl` 
(extracted from https://www.cs.cornell.edu/people/pabo/movie-review-data/) contains trojaned external data.
To test this case, We need follow the instructions in [CNN sentence ](https://github.com/yoonkim/CNN_sentence). 
First download pre-trained word2vec  binary file from https://code.google.com/p/word2vec/
Then run,

```
$ python process_data.py path
```

 where path points to the word2vec binary file (i.e. GoogleNews-vectors-negative300.bin file). This will create a pickle object called mr.p in the same folder, which contains the dataset in the right format.
Then, we can test the model by run,
`python conv_net_sentence_mlp_test.py model_to_test.pkl`


## Web Site

https://purduepaml.github.io/TrojanNN/

## Contacts

Yingqi Liu, liu1751@purdue.edu

Shiqing Ma, ma229@purdue.edu
