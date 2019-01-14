import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from collections import OrderedDict
from random import randint
from skimage.util import random_noise
from scipy import ndimage, ndarray
from abc import abstractmethod, ABCMeta
import random

from scipy import ndimage, ndarray
from skimage.transform import rotate, resize
from skimage.util import random_noise

import glob

out_dir = "00_out"
sess = None
x = None

def maxpool2d(x, k=2, padding="VALID"):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding=padding)


def  LeNet1(x):
    y = tf.Variable(tf.truncated_normal([128, 43]))

    return y

def LeNet_3_3(x, conv_keep_prob=1, fc_keep_prob=1, depth=3, conv1_depth=6, conv2_depth=16, conv3_depth = 64, filter_size=5):
    # input_array = (x-128)/128
    input_array = x
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    width = 32
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x6.
    F_W1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, depth, conv1_depth], mean=mu, stddev=sigma), name="cc_weight1")
    F_b1 = tf.Variable(tf.zeros(conv1_depth))

    strides = [1, 1, 1, 1]
    padding = 'VALID'
    cc_out1 = tf.nn.conv2d(input_array, F_W1, strides, padding) + F_b1
    width = width - filter_size + 1

    # TODO: Activation.
    cc_out1 = tf.nn.relu(cc_out1)
    cc_out1 = tf.nn.dropout(cc_out1, conv_keep_prob)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    cc_out1 = maxpool2d(cc_out1, k=2)
    width = int(width/2)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    F_W2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, conv1_depth, conv2_depth], mean=mu, stddev=sigma), name="cc_weight2")
    F_b2 = tf.Variable(tf.zeros(conv2_depth))

    strides = [1, 1, 1, 1]
    padding = 'VALID'
    cc_out2 = tf.nn.conv2d(cc_out1, F_W2, strides, padding) + F_b2
    width = width - filter_size + 1

    # TODO: Activation.
    cc_out2 = tf.nn.relu(cc_out2)
    cc_out2 = tf.nn.dropout(cc_out2, conv_keep_prob)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    cc_out2 = maxpool2d(cc_out2, k=2)
    width = int(width/2)


    # TODO: Layer 3: Convolutional. Output = 5x5x16.
    F_W3 = tf.Variable(tf.truncated_normal([filter_size, filter_size, conv2_depth, conv3_depth], mean=mu, stddev=sigma), name="cc_weight3")
    F_b3 = tf.Variable(tf.zeros(conv3_depth))

    strides = [1, 1, 1, 1]
    padding = 'VALID'
    cc_out3 = tf.nn.conv2d(cc_out2, F_W3, strides, padding) + F_b3
    width = width - filter_size + 1

    # TODO: Activation.
    cc_out3 = tf.nn.relu(cc_out3)
    cc_out3 = tf.nn.dropout(cc_out3, conv_keep_prob)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    #cc_out3 = maxpool2d(cc_out3, k=2)
    #width = int(width/2)


    # TODO: Flatten. Input = 4*4*32. Output = 400.
    flat_inputs = flatten(cc_out3)
    INPUT_LENGTH = width*width*conv3_depth
    #flat_inputs = flatten(cc_out2)
    #INPUT_LENGTH = 4*4*16

    # TODO: Layer 4: Fully Connected. Input = 400. Output = 120.
    fc1_output_width = 200
    FULL_W1 = tf.Variable(tf.truncated_normal([INPUT_LENGTH, fc1_output_width], mean=mu, stddev=sigma), name="FULL_W1")
    FULL_B1 = tf.Variable(tf.truncated_normal([fc1_output_width]), name="FULL_B1")
    fc1 = tf.add(tf.matmul(flat_inputs, FULL_W1), FULL_B1)

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, fc_keep_prob)

    # fc1 = tf.nn.dropout(fc1, dropout)
    fc2_output_width = 100
    FULL_W2 = tf.Variable(tf.truncated_normal([fc1_output_width, fc2_output_width], mean=mu, stddev=sigma), name="FULL_W2")
    # TODO: Layer 5: Fully Connected. Input = 120. Output = 84.
    FULL_B2 = tf.Variable(tf.truncated_normal([fc2_output_width]), name="FULL_B2")
    fc2 = tf.add(tf.matmul(fc1, FULL_W2), FULL_B2)

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, fc_keep_prob)


    # TODO: Layer 6: Fully Connected. Input = 84. Output =
    fc3_output_width = 43

    FULL_W3 = tf.Variable(tf.truncated_normal([fc2_output_width, fc3_output_width], mean=mu, stddev=sigma), name="FULL_W3")
    FULL_B3 = tf.Variable(tf.truncated_normal([fc3_output_width]), name="FULL_B3")
    logits = tf.add(tf.matmul(fc2, FULL_W3), FULL_B3)

    return logits


def LeNet_2_4(x, conv_keep_prob=1, fc_keep_prob=1, depth=3, conv1_depth=6, conv2_depth=16, filter_size=5):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    width = 32
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, depth, conv1_depth), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_depth))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, conv_keep_prob)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, conv1_depth, conv2_depth), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(conv2_depth))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, conv_keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    INPUT_LENGTH = width*width*conv2_depth

    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_output_width = 300

    fc1_W = tf.Variable(tf.truncated_normal(shape=(INPUT_LENGTH, fc1_output_width), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(fc1_output_width))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, fc_keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_output_width = 200

    fc2_W = tf.Variable(tf.truncated_normal(shape=(fc1_output_width, fc2_output_width), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(fc2_output_width))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, fc_keep_prob)


    # SOLUTION: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc3_output_width = 100

    fc3_W = tf.Variable(tf.truncated_normal(shape=(fc2_output_width, fc3_output_width), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(fc3_output_width))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    # SOLUTION: Activation.
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, fc_keep_prob)


    # SOLUTION: Layer 6: Fully Connected. Input = 84. Output = 10.
    fc4_output_width = 43

    fc4_W = tf.Variable(tf.truncated_normal(shape=(fc3_output_width, fc4_output_width), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(fc4_output_width))
    logits = tf.matmul(fc3, fc4_W) + fc4_b

    return logits

def LeNet_2_5(x, conv_keep_prob=1, fc_keep_prob=1, depth=3, conv1_depth=6, conv2_depth=16, filter_size=5):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    width = 32
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, depth, conv1_depth), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_depth))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, conv_keep_prob)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, conv1_depth, conv2_depth), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(conv2_depth))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, conv_keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    INPUT_LENGTH = width*width*conv2_depth

    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_output_width = 320

    fc1_W = tf.Variable(tf.truncated_normal(shape=(INPUT_LENGTH, fc1_output_width), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(fc1_output_width))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, fc_keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_output_width = 240

    fc2_W = tf.Variable(tf.truncated_normal(shape=(fc1_output_width, fc2_output_width), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(fc2_output_width))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, fc_keep_prob)


    # SOLUTION: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc3_output_width = 160

    fc3_W = tf.Variable(tf.truncated_normal(shape=(fc2_output_width, fc3_output_width), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(fc3_output_width))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    # SOLUTION: Activation.
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, fc_keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc4_output_width = 80

    fc4_W = tf.Variable(tf.truncated_normal(shape=(fc3_output_width, fc4_output_width), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(fc4_output_width))
    fc4 = tf.matmul(fc3, fc4_W) + fc4_b

    # SOLUTION: Activation.
    fc4 = tf.nn.relu(fc4)
    fc4 = tf.nn.dropout(fc4, fc_keep_prob)


    # SOLUTION: Layer 6: Fully Connected. Input = 84. Output = 10.
    output_width = 43
    fc_W = tf.Variable(tf.truncated_normal(shape=(fc4_output_width, output_width), mean=mu, stddev=sigma))
    fc_b = tf.Variable(tf.zeros(output_width))
    logits = tf.matmul(fc4, fc_W) + fc_b

    return logits


def LeNet(x, conv_keep_prob=1, fc_keep_prob=1, depth=3, conv1_depth=6, conv2_depth=16, filter_size=5):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    width = 32
    global conv1, conv2, fc0, fc1, fc2, sess
    global conv1_W, conv1_b, conv2_W, conv2_b, fc1_W, fc2_W, fc3_W, fc1_b, fc2_b, fc3_b

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, depth, conv1_depth), mean=mu, stddev=sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(conv1_depth), name="conv1_b")
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, conv_keep_prob)


    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, conv1_depth, conv2_depth), mean=mu, stddev=sigma), name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(conv2_depth),  name="conv2_b")
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    width = width - filter_size + 1

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, conv_keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    width = int(width/2)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    flatten_length = np.int64(conv2_depth*width*width)
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_output_width = 200
    fc1_W = tf.Variable(tf.truncated_normal(shape=(flatten_length, fc1_output_width), mean=mu, stddev=sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(fc1_output_width), name="fc1_b")
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, fc_keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_output_width = 100
    fc2_W = tf.Variable(tf.truncated_normal(shape=(fc1_output_width, fc2_output_width), mean=mu, stddev=sigma), name="fc2_W")
    fc2_b = tf.Variable(tf.zeros(fc2_output_width), name="fc2_b")
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, fc_keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_output_width = 43
    fc3_W = tf.Variable(tf.truncated_normal(shape=(fc2_output_width, fc3_output_width), mean=mu, stddev=sigma), name="fc3_W")
    fc3_b = tf.Variable(tf.zeros(fc3_output_width), name="fc3_b")
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def feed_forward_(X, keep_prob=1, depth=3):
    NO_OF_HIDDEN_LAYERS = 3

    NO_OF_NODES_INPUT = 32 * 32 * depth

    NO_OF_NODES_Hidden_1 = 500
    NO_OF_NODES_Hidden_2 = 500
    NO_OF_NODES_Hidden_3 = 500

    NO_OF_NODES_OUTPUT = 43
    #print(tf.get_default_graph())
    global hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer
    X = tf.contrib.layers.flatten(X)
    hidden_1_layer = {'wight': tf.Variable(tf.random_normal([NO_OF_NODES_INPUT, NO_OF_NODES_Hidden_1]), name="hidden_1_layer_wight", dtype=tf.float32),
                      'biases': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_1]), name="hidden_1_layer_biases", dtype=tf.float32)}

    hidden_2_layer = {'wight': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_1, NO_OF_NODES_Hidden_2]),  name="hidden_2_layer_wight", dtype=tf.float32),
                      'biases': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_2]), name="hidden_2_layer_biases", dtype=tf.float32)}

    hidden_3_layer = {'wight': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_2, NO_OF_NODES_Hidden_3]),  name="hidden_3_layer_wight", dtype=tf.float32),
                      'biases': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_3]), name="hidden_3_layer_biases", dtype=tf.float32)}

    output_layer = {'wight': tf.Variable(tf.random_normal([NO_OF_NODES_Hidden_3, NO_OF_NODES_OUTPUT]),  name="output_layer_wight", dtype=tf.float32),
                      'biases': tf.Variable(tf.random_normal([NO_OF_NODES_OUTPUT]), name="output_layer_biases", dtype=tf.float32)}

    # layer output = Activation(layer input * wights + bias)
    layer_1_output = tf.add(tf.matmul(X, hidden_1_layer['wight']), hidden_1_layer['biases'], name="Out_L1_0")
    layer_1_output = tf.nn.relu(layer_1_output, name="Out_L1_1")
    #layer_1_output = tf.nn.dropout(layer_1_output, keep_prob)


    layer_2_output = tf.add(tf.matmul(layer_1_output, hidden_2_layer['wight']), hidden_2_layer['biases'],  name="Out_L2_0")
    layer_2_output = tf.nn.relu(layer_2_output, name="Out_L2_1")
    #layer_2_output = tf.nn.dropout(layer_2_output, keep_prob)


    layer_3_output = tf.add(tf.matmul(layer_2_output, hidden_3_layer['wight']), hidden_3_layer['biases'], name="Out_L3_0")
    layer_3_output = tf.nn.relu(layer_3_output, name="Out_L3_1")
    #layer_3_output = tf.nn.dropout(layer_3_output, keep_prob)

    output = tf.add(tf.matmul(layer_3_output, output_layer['wight']), output_layer['biases'])
    #output = tf.nn.relu(output)

    return output

def prep_data_augment(image):
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=63/255.0)
    #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image

def data_augment(input_tensor):
    output_tensor = tf.map_fn(prep_data_augment, input_tensor)
    return output_tensor

def convert_figure_to_array(fig0):
    fig0.canvas.draw()
    data = np.fromstring(fig0.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig0.canvas.get_width_height()
    result = data.reshape((h, w, 3))
    return result


def store_image(image, name, dir=None, img_form="jpg"):
    if not dir:
        dir = out_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = dir + "/" + name + "." + img_form
    print("Saving image: " + name)
    if np.max(image) == 1:
        image = image * 255

    im = Image.fromarray(image)
    if im.mode != 'RGB':
       im = im.convert('RGB')
    im.save(name)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text= "Epoch={:d}, accuracy={:.3f}".format(int(xmax), ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = min(y)
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)

def get_date_time():
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    return timestr

def write_json_file(data, file_name):
    """
    :param file_name: the file name to save
    :param data: data to store
    write data in a json file.
    :return:
    """
    # Common.print_system_info()
    # Get Configuration file path
    # ---------------------------
    ext = ".json"
    file_name += ext
    with open(file_name, 'w') as fp:
        print("Writing file : " + file_name)
        json.dump(data, fp, indent=4, sort_keys=True)


def Rotate(image_array, max_right_degree=10, max_left_degree=10):
    random_degree = random.uniform(-max_right_degree, max_left_degree)
    return rotate(image_array, random_degree)


def RandomNoise(image_array, mode='gaussian'):
    return random_noise(image_array, mode=mode)

def Blur(image_array, size=3):
    return ndimage.uniform_filter(image_array, size=size)

def evaluate_network():

    pass
def find_files(p=None, ext=".log", recursive=False):
    """
    Function to get all files of certain extensions within certain directory.
    :param p: path to scan
    :param ext: extension to search for.
    :return: list of files found.
    """

    ret = []
    if recursive:
        for root, dirs, files in os.walk(p):
            for loc_file in files:
                if loc_file.endswith(ext):
                    f = os.path.abspath(os.path.join(root, loc_file))
                    if os.path.isfile(f):
                        ret.append(f)
    else:
        ret = glob.glob(p+"/*"+ext)
    return ret




def augment_imgs(imgs, p):
    """
    Performs a set of augmentations with with a probability p
    """
    augs = iaa.SomeOf((2, 4),
                      [
                          iaa.Crop(px=(0, 4)),  # crop images from each side by 0 to 4px (randomly chosen)
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                          iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                          iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to +45 degrees)
                          iaa.Affine(shear=(-10, 10))  # shear by -10 to +10 degrees
                      ])

    seq = iaa.Sequential([iaa.Sometimes(p, augs)])

    return seq.augment_images(imgs)
