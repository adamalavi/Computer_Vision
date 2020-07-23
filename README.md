# Computer_Vision
This repository includes three projects that use basic concepts of computer vision and deep neural networks. 

### Facial keypoints detection
In this project, a convolutional neural network is trained to detect a set of 68 points in an image of a face. Haar cascade is used to detect the faces in the images first and then the neural network is used to detect the keypoints on each of the faces. These keypoints can further be used for a variety of applications like facial tracking, facial pose recognition, facial filters, and emotion recognition. One such application is superimposing a png on to a face like a hat, sunglasses, moustache, etc which is done in the last notebook.

### Image captioning
The objective of this project is to develop a network that outputs a description of any image that is provided as input. The entire network can be divided into two seperate neural networks that is a convolutional neural network that extracts the features from an image which is then provided as input to a recurrent neural network made up of LSTM cells that generates an appropriate caption for the image.

### Visual SLAM
This is a basic implementation of 2D SLAM(Simultaneous localisation and mapping) that focusses on the math and concepts behind the technique.
