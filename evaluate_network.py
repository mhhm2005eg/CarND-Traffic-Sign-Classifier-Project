import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
from common import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.preprocessing import LabelBinarizer

EPOCHS = 20
BATCH_SIZE = 32
learning_rate = 0.001
feed_forward = LeNet  # feed_forward_#LeNet_1#LeNet
fc_keep_prob_value = .5
conv_keep_prob_value = .75
saver = None
conv1_depth_val = 6
conv2_depth_val = 16
conv_filter_size = 5
available_training_data_set = 1
image_depth = 1
acc_list = [0]
train_acc_list = [0]
loss_list = [1]
CONTINUE = False
TRAIN = True
TEST = True
model_folder = "2019_01_14-14_15_16"

def LeNet(x, conv_keep_prob=1, fc_keep_prob=1, depth=3, conv1_depth=6, conv2_depth=16, filter_size=5):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    width = 32
    global conv1, conv2, fc0, fc1, fc2, sess, logits
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



def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    global sess, x
    activation = tf_activation.eval(session=sess, feed_dict={x: image_input, fc_keep_prob: fc_keep_prob_value, conv_keep_prob: conv_keep_prob_value})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")


sess = tf.InteractiveSession()
sess.as_default()
#tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
y = tf.placeholder(tf.int32, (None, 43))
fc_keep_prob = tf.placeholder(tf.float32)
conv_keep_prob = tf.placeholder(tf.float32)
#img = np.ndarray()
#sess.run(fc_keep_prob, feed_dict={fc_keep_prob: fc_keep_prob_value})
#sess.run(conv_keep_prob, feed_dict={conv_keep_prob: conv_keep_prob_value})
logit= LeNet(x, conv_keep_prob, fc_keep_prob, depth=image_depth, conv1_depth=conv1_depth_val, conv2_depth=conv2_depth_val, filter_size=conv_filter_size)

op = sess.graph.get_operations()
print([m.values() for m in op][1])


#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('./00_out/'+model_folder+'/LeNet.meta')

saver.restore(sess, './00_out/' + model_folder + '/LeNet')
#saver.restore(sess, './00_out/2019_01_14-12_03_04/LeNet')

images = find_files("./01_test_images", ".png")

op = sess.graph.get_operations()
print([m.values() for m in op][1])

labels = [8, 15, 12, 25, 16]
encoder = LabelBinarizer()
encoder.fit(range(43))
labels = encoder.transform(labels)


img_gray_array = np.ndarray
i = -1
for image_path in images:
    i+=1
    img_rgb = mpimg.imread(image_path)
    img_gray = np.sum(img_rgb/3, axis=2, keepdims=True)
    #print(img_gray.shape)
    img_gray = cv2.resize(img_gray, (32, 32), interpolation=cv2.INTER_AREA)
    img_gray = np.reshape(img_gray, (1, 32, 32, 1))

    #print(img_gray.shape)

    #plt.imshow(img_gray[0,:, :,0])
    #plt.show()
    img = (img_gray - 128)/128
    #img_gray_array[0] = img_gray
    #saver.restore(sess, './00_out/2019_01_13-13_52_52/LeNet')

    #sess.run(LeNet(x, conv_keep_prob, fc_keep_prob, depth=image_depth, conv1_depth=conv1_depth_val, conv2_depth=conv2_depth_val, filter_size=conv_filter_size),
             #feed_dict={x:img, fc_keep_prob:fc_keep_prob_value, conv_keep_prob:conv_keep_prob_value})
    logit_val = sess.run(logit, feed_dict={x:img, fc_keep_prob:fc_keep_prob_value, conv_keep_prob:conv_keep_prob_value})
    class_val = sess.run(tf.nn.softmax_cross_entropy_with_logits(logits=logit_val, labels=labels[i]))
    print(class_val)
    outputFeatureMap(img_gray, conv1)
