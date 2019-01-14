import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
from common import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from scipy.ndimage import rotate
from random import randint
from skimage.util import random_noise
from scipy import ndimage, ndarray
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


#print(cv2.getBuildInformation())

X_train = None
y_train = None
X_valid = None
y_valid = None
X_test = None
y_test = None
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

Data_Augmentation_EqualizeHist = True
Data_Augmentation_Noise = True
Data_Augmentation_Blur = False
Data_Augmentation_Rotate = False
DISPLAY_DATA_SET_HISTOGRAM = False

def main():
    global X_train, y_train, X_valid, y_valid, X_test, y_test, EPOCHS, BATCH_SIZE, learning_rate, feed_forward,\
        fc_keep_prob_value, conv_keep_prob_value, saver, CONTINUE, conv1_depth_val, conv2_depth_val, conv_filter_size,\
    available_training_data_set, image_depth, acc_list, train_acc_list, loss_list, TRAIN, TEST


    global accuracy_operation, x, y, conv_keep_prob, fc_keep_prob

    # TODO: Fill this in based on where you saved the training and testing data

    training_file = "./traffic_signs_data/train.p"
    validation_file = "./traffic_signs_data/valid.p"
    testing_file = "./traffic_signs_data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    # TODO: Number of training examples
    n_train = len(y_train)

    # TODO: Number of validation examples
    n_validation = len(y_valid)

    # TODO: Number of testing examples.
    n_test = len(y_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = max(np.concatenate((y_train ,y_valid , y_test),  axis=0)) + 1

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    #DATA analysis
    if DISPLAY_DATA_SET_HISTOGRAM:
        n, bins, patches = plt.hist(y_train, n_classes, facecolor='blue', alpha=0.5)
        plt.show()

    #convert to grayscale
    if image_depth == 1:
        X_train = np.sum(X_train/3, axis=3, keepdims=True)
        X_valid = np.sum(X_valid/3, axis=3, keepdims=True)
        X_test = np.sum(X_test/3, axis=3, keepdims=True)
        #Histograms Equalization in OpenCV
        if Data_Augmentation_EqualizeHist:
            X_new = np.zeros_like(X_train)
            for i in range(X_train.shape[0]):
                X_new[i, :, :, 0] = cv2.equalizeHist(X_train[i].astype(np.uint8))
            add_images(X_new)
            for i in range(X_valid.shape[0]):
                X_valid[i, :, :, 0] = cv2.equalizeHist(X_valid[i].astype(np.uint8))
            for i in range(X_test.shape[0]):
                X_test[i, :, :, 0] = cv2.equalizeHist(X_test[i].astype(np.uint8))

    x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
    y = tf.placeholder(tf.int32, (None, 43))

    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()

    # Normalize  data
    X_train = (X_train - 128)/128
    X_valid = (X_valid - 128)/128
    X_test = (X_test - 128)/128

    index = random.randint(0, len(X_train))


    if Data_Augmentation_Rotate:
        print("Data Augmentation %s" %"Rotate")
        random_degree = random.uniform(-10, 10)
        x_rotate = rotate(X_train, random_degree)
        X_train = np.concatenate((X_train, x_rotate), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        available_training_data_set += 1

    if Data_Augmentation_Noise:
        print("Data Augmentation %s" %"Noise")

        x_noise = RandomNoise(X_train)
        #x_noise = x_noise / 2
        image_x = X_train[index].squeeze()
        image_noise = x_noise[index].squeeze()
        plt.figure(figsize=(1,2))
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image_x)
        ax[1].imshow(image_noise)
        plt.show()
        X_train = np.concatenate((X_train, x_noise), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        available_training_data_set += 1

    if Data_Augmentation_Blur:
        print("Data Augmentation %s" %"Blur")
        x_blur = Blur(X_train, size=3)
        image_x = X_train[index].squeeze()
        image_blur = x_blur[index].squeeze()
        plt.figure(figsize=(1,2))
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image_x)
        ax[1].imshow(image_blur)
        plt.show()
        X_train = np.concatenate((X_train, x_blur), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        available_training_data_set += 1


    X_train, y_train = shuffle(X_train, y_train)
    n_train *=available_training_data_set

    encoder = LabelBinarizer()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_valid = encoder.transform(y_valid)
    y_test = encoder.transform(y_test)

    if False:
        plt.figure(figsize=(1,1))
        plt.imshow(image)
        print(y_train_num[index])
        print(y_train[index])
    print('Labels One-Hot Encoded')

    plt.show()

    #@staticmethod
    #def define_cost_optimizer(self):
    fc_keep_prob = tf.placeholder(tf.float32)
    conv_keep_prob = tf.placeholder(tf.float32)

    logits = feed_forward(x, conv_keep_prob, fc_keep_prob, depth=image_depth, conv1_depth=conv1_depth_val, conv2_depth=conv2_depth_val, filter_size=conv_filter_size)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate )

    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if CONTINUE == True:
            saver.restore(sess, './00_out/2019_01_13-13_52_52/LeNet')
        else:
            sess.run(tf.global_variables_initializer())

        num_examples = int(n_train/BATCH_SIZE)*BATCH_SIZE

        if TRAIN:
            print("Training...")
            for i in range(EPOCHS):
                #print(type(y_train))
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    if end >= n_train:
                        end = n_train-1
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                    _, l = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, conv_keep_prob : conv_keep_prob_value, fc_keep_prob : fc_keep_prob_value})
                loss_list.append(l)

                validation_accuracy = evaluate(X_valid, y_valid)
                training_accuracy = evaluate(X_train, y_train)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print("Training Accuracy = {:.3f}".format(training_accuracy))
                acc_list.append(validation_accuracy)
                train_acc_list.append(training_accuracy)
                print()

            sub_folder_name = get_date_time()
            fig0 = plt.figure(0)
            fig0.clf()
            plt.plot(acc_list, label='validation_accuracy')
            plt.plot(train_acc_list, label='training_accuracy')
            plt.grid(True)
            plt.legend()
            annot_max(range(len(acc_list)),acc_list)
            print("Max accuracy %.3f at Epoch %d" %(max(acc_list), range(len(acc_list))[np.argmax(acc_list)]))
            arr = convert_figure_to_array(fig0)
            store_image(arr, "validation_accuracy", out_dir + "/" + sub_folder_name)

            fig0 = plt.figure(0)
            fig0.clf()
            plt.plot(loss_list, label='loss')
            plt.grid(True)
            plt.legend(loc=2)
            annot_min(range(len(loss_list)),loss_list)

            arr = convert_figure_to_array(fig0)
            store_image(arr, "loss", out_dir + "/" + sub_folder_name)

            data_list = ["EPOCHS", "BATCH_SIZE", "learning_rate", "feed_forward.__name__",
                         "fc_keep_prob_value", "conv_keep_prob_value", "CONTINUE", "image_depth",
                         "conv1_depth_val", "conv2_depth_val", "conv_filter_size",
                         "acc_list", "train_acc_list", "loss_list"]
            file_name =  out_dir + "/" + sub_folder_name + "/" + "configs"
            save_variables(file_name, data_list)

            saver.save(sess, "./" + out_dir + "/" + sub_folder_name + '/' + feed_forward.__name__)
            print("Model saved")

        if TEST:
            print("Testing ...")
            test_accuracy = evaluate(X_test, y_test)
            print("Testing Accuracy = {:.3f}".format(test_accuracy))


def evaluate(X_data, y_data):
    global accuracy_operation, x, y, conv_keep_prob, fc_keep_prob
    num_examples = int(len(X_data)/BATCH_SIZE)*BATCH_SIZE
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation , feed_dict={x: batch_x, y: batch_y, conv_keep_prob: 1, fc_keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def save_variables(file_name, data_list):
    global X_train, y_train, X_valid, y_valid, X_test, y_test, EPOCHS, BATCH_SIZE, learning_rate, feed_forward,\
        fc_keep_prob_value, conv_keep_prob_value, saver, CONTINUE, conv1_depth_val, conv2_depth_val, conv_filter_size, \
        image_depth, acc_list, train_acc_list, loss_list

    data_dic = OrderedDict()
    for var in data_list:
        data_dic[var] = str(eval(var))
    write_json_file(data_dic,  file_name)

def add_images(X_new):
    global X_train, y_train, available_training_data_set
    X_train = np.concatenate((X_train, X_new), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    available_training_data_set += 1



if __name__ == "__main__":

    main()

