import numpy as np

class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # Normalizing factor along 3 color channels
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Download from http://www.vlfeat.org/matconvnet/pretrained/
    STYLE_IMAGE = 'images/400/monet.jpg' # Style image to use.
    CONTENT_IMAGE = 'images/400/louvre_small.jpg' # Content image to use.
    OUTPUT_DIR = 'output/'
    CONTENT_LAYER = 'conv4_2'
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
    ALPHA = 10
    BETA = 40
    LEARNING_RATE = 2.0
    NUM_ITERATIONS = 200