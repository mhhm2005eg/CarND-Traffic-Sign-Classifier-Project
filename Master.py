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

class master(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
        self.learning_rate = 0.001
        self.feed_forward = LeNet
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None, 43))
        self.saver = None

        self.load_data()
        self.pre_process()

    def load_data(self):
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

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']
        # TODO: Number of training examples
        self.n_train = len(self.y_train)

        # TODO: Number of validation examples
        self.n_validation = len(self.y_valid)

        # TODO: Number of testing examples.
        self.n_test = len(self.y_test)

        # TODO: What's the shape of an traffic sign image?
        self.image_shape = self.X_train[0].shape

        # TODO: How many unique classes/labels there are in the dataset.
        #print(type(self.y_test))
        self.n_classes = max(np.concatenate((self.y_train ,self.y_valid , self.y_test),  axis=0)) + 1

        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)
        #print(self.X_train[0].shape)

    def pre_process(self):
        # shuffle training data
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

        #convert to gray scale images
        #self.X_train = tf.image.rgb_to_grayscale(self.X_train, name="X_train_gray")
        #self.X_valid = tf.image.rgb_to_grayscale(self.X_valid, name="X_valid_gray")
        #self.X_test = tf.image.rgb_to_grayscale(self.X_test, name="X_test_gray")
        # Normalize  data
        self.X_train = (self.X_train - 128)/128
        self.X_valid = (self.X_valid - 128)/128
        self.X_test = (self.X_test - 128)/128

        # convert labels to one-hot code
        #self.y_train_hot = tf.one_hot(self.y_train, self.n_classes)
        #self.y_valid_hot = tf.one_hot(self.y_valid, self.n_classes)
        #self.y_test_hot = tf.one_hot(self.y_test, self.n_classes)
        self.labels_encoding()

    #@staticmethod
    def define_cost_optimizer(self):
        logits = self.feed_forward(self.x)
        #print(self.y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def evaluate(self, X_data, y_data):
        num_examples = int(len(X_data)/self.BATCH_SIZE)*self.BATCH_SIZE
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + self.BATCH_SIZE], y_data[offset:offset + self.BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


    def labels_encoding(self):

        # Turn labels into numbers and apply One-Hot Encoding
        encoder = LabelBinarizer()
        encoder.fit(self.y_train)
        self.y_train = encoder.fit_transform(self.y_train)
        self.y_valid = encoder.fit_transform(self.y_valid)
        self.y_test = encoder.fit_transform(self.y_test)

        # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
        self.y_train = self.y_train.astype(np.float32)
        self.y_valid = self.y_valid.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

        print('Labels One-Hot Encoded')



    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = int(self.n_train/self.BATCH_SIZE)*self.BATCH_SIZE

            print("Training...")
            print()
            for i in range(self.EPOCHS):
                #print(type(self.y_train))
                X_train, y_train = (self.X_train, self.y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    if end >= self.n_train:
                        end = self.n_train-1
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    #print(type(self.x), type(batch_x))
                    #one_hot_y = tf.one_hot(batch_y, self.n_classes)
                    #print(i, offset, end)
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})
                #print(type(self.X_train))
                validation_accuracy = self.evaluate(self.X_valid, self.y_valid)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            self.saver.save(sess, './lenet')
            print("Model saved")
    @staticmethod
    def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
        # Here make sure to preprocess your image_input in a way your network expects
        # with size, normalization, ect if needed
        # image_input =
        # Note: x should be the same name as your network's tensorflow data placeholder variable
        # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
        activation = tf_activation.eval(session=sess, feed_dict={x: image_input})
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

    def main(self):
        self.define_cost_optimizer()
        self.run()


if __name__ == "__main__":

    Master = master()
    Master.main()

