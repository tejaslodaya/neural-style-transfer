# Neural style transfer

Neural Style Transfer is an algorithm that given a content image C and a style image S can generate novel artistic image

#### A few examples
==================
* The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night) 
  <img src="https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/perspolis_vangogh.png">
* The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan 
  <img src="https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/pasargad_kashi.png">
* A scientific study of a turbulent fluid with the style of a abstract blue fluid painting
  <img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/circle_abstract.png">


#### Transfer Learning
=====================
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning. 

<img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/vgg19.jpg">

Following the original [NST paper](https://arxiv.org/abs/1508.06576), I have used the VGG network. Specifically, VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers)

#### Cost function
=================
Most of the algorithms optimize a cost function to get a set of parameter values. In NST, optimize a cost function to get pixel values!
1. Building the content cost function

    "Generated" image G should have similar content as the input image C. The most visually pleasing results will be generated if a layer is chosen in the middle of the network--neither too shallow nor too deep. Set the image C as the input to the pretrained VGG network, and run forward propagation. a(C) be the hidden layer activation in the layer you had chosen. Set G as the input, and run forward propagation. a(G) be the corresponding hidden layer activation. The cost function will be:
    <img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/equation1.png">
    
    When minimizing the content cost later, it helps make sure G has similar content as C
    
2. Building the style cost function

    Gram matrix (or style matrix) which serves as the basic building block of style cost function computes the correlation between filters. This matrix is of dimensions (nC,nC) where nC is the number of filters. The value G(i,j) measures how similar the activation of filter i are to the activation of filter j. The Style matrix G measures the style of an image. After generating the Style matrix (Gram matrix), goal is to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G
    
    <img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/equation2.png">
    
    where G(S) and G(G) are respectively the Gram matrices of the "style" image and the "generated" image, computed using the hidden layer activations for a particular hidden layer in the network.
    
    Better results are obtained when style costs from several different layers are merged. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient. λ[l] is the weights given to different layers
    
    <img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/equation3.png">

    Minimizing the style cost will cause the image G to follow the style of the image S

3. Putting it together to get total cost

    Cost function that minimizes both the style and the content cost
    
    <img src = "https://raw.githubusercontent.com/tejaslodaya/neural-style-transfer/master/nb_images/equation4.png">
    
    The total cost is a linear combination of the content cost and the style cost. α and β are hyperparameters that control the relative weighting between content and style
    
Now, reduce the cost function and "generated" image will be a combination of content of content image and style of style image
         
#### Steps
=========
The following steps are to be followed to synthesize new images. Find correlation between the steps mentioned below and `nst_main.py`
1. Create an Interactive tensorflow session
2. Load the content image
3. Load the style image
4. Randomly initialize the image to be generated
5. Load the pretrained VGG16 model
6. Build the TensorFlow graph:
    * Run the content image through the VGG16 model and compute the content cost
    * Run the style image through the VGG16 model and compute the style cost
    * Compute the total cost
    * Define the optimizer and the learning rate
7. Initialize the TensorFlow graph and run it for a large number of iterations(200 here), updating the generated image at every step

#### NOTE
========
1. Download pretrained VGG model from [here](http://www.vlfeat.org/matconvnet/pretrained/) and place it in `pretrained_model` folder. Change the `config.py` file to point to VGG19 model path 
2. Run the `nst_main.py` on different style and content images placed in `images` folder. Change the `config.py` accordingly
3. Content & style images can be found in `images` directory. Corresponding output images can be found in `output` directory

#### References
==============
* [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Harish Narayanan, Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
* [Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
* [Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
* [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/)

