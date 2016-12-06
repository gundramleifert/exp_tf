from __future__ import print_function
# from PIL import Image
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes
import numpy as np
import os

num_epochs = 3
filename = "./resources/list_all.txt"


class ImageLoader:
    def __init__(self, path, batchSize):
        self._path = path
        self._batchSize = batchSize
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def getBatchSize(self):
        return self._batchSize

    def getEpochsCompleted(self):
        return self._epochs_completed

    def read_labeled_image_list(self, image_list_file):

        """Reads a .txt file containing paths to the images
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """
        f = open(image_list_file, 'r')
        filenames = []
        labels = []
        for line in f:
            filenames.append(line[:-1])
            labelFile = line[:-1] + ".txt"
            tmp = open(labelFile, 'r')
            labels.append(tmp.readline())
        if tmp is not None:
            tmp.close()
        f.close()

        self._fileNames = filenames
        self._labels = labels

    def getNextBatchTF(self):

        fileNamesBatch = self._fileNames[self._index_in_epoch:self._index_in_epoch + self._batchSize]
        labelsBatch = self._labels[self._index_in_epoch:self._index_in_epoch + self._batchSize]
        imgList = tf.convert_to_tensor(fileNamesBatch, dtype=dtypes.string)
        labels = tf.convert_to_tensor(labelsBatch, dtype=dtypes.string)

        # Load images
        filenameQueue = tf.train.string_input_producer(imgList)

        reader = tf.WholeFileReader()
        images = []
        for _ in xrange(self._batchSize):
            imgKey, imgValue = reader.read(filenameQueue)
            img = tf.image.decode_jpeg(imgValue)
            images.append(img)

        self._index_in_epoch += self._batchSize

        # Epoch completed
        if self._index_in_epoch >= len(self._fileNames):
            self._epochs_completed += 1
            self._index_in_epoch = 0

            # Shuffle data
            tmpList = list(zip(self._fileNames, self._labels))
            np.random.shuffle(tmpList)
            self._fileNames, self._labels = zip(*tmpList)

        return images, labels

    def getNextBatch(self):

        fileNames = self._fileNames[self._index_in_epoch:self._index_in_epoch + self._batchSize]
        labels = self._labels[self._index_in_epoch:self._index_in_epoch + self._batchSize]

        # Load images
        images = []
        for file in fileNames:
            img = misc.imread(file)
            images.append(img)

        self._index_in_epoch += self._batchSize

        # Epoch completed
        if self._index_in_epoch >= len(self._fileNames):
            self._epochs_completed += 1
            self._index_in_epoch = 0

            # Shuffle data
            tmpList = list(zip(self._fileNames, self._labels))
            np.random.shuffle(tmpList)
            self._fileNames, self._labels = zip(*tmpList)

        return images, labels


if __name__ == '__main__':
    os.chdir("..")

    loader = ImageLoader(filename, 2)

    # Reads paths of images together with their labels
    loader.read_labeled_image_list(filename)

    while loader.getEpochsCompleted() < num_epochs:
        print("Epoch: " + str(loader._epochs_completed + 1) + "\tIndex: " + str(loader._index_in_epoch))
        images, labels = loader.getNextBatch()

        # for i in xrange(loader.getBatchSize()):
        #     print("\tLabel: " + str(labels[i]))
        #     print("\tShape: " + str(images[i].shape))

    plt.subplot(1,2,1)
    plt.imshow(images[0])
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(images[1])
    plt.axis('off')

    plt.show()

    # init_op = tf.initialize_all_variables()
    # sess = tf.Session()
    # with sess.as_default():
    #     while loader.getEpochsCompleted() < num_epochs:
    #         print("Epoch: " + str(loader._epochs_completed + 1) + "\tIndex: " + str(loader._index_in_epoch))
    #         images, labels = loader.getNextBatchTF()
    #
    #         sess.run(init_op)
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #         images1, labels1 = sess.run([images, labels])
    #
    #         for i in xrange(loader.getBatchSize()):
    #             print("\tLabel: " + str(labels1[i]))
    #             print("\tShape: " + str(images1[i].shape))
    #
    #             # for l in labels1:
    #             #     print(l)
    #
    #             # for i in range():  # length of your filename list
    #             #     image = my_img.eval()  # here is your image Tensor :)
    #             #
    #             # print(image.shape)
    #             # Image.show(Image.fromarray(np.asarray(image)))
    #
    #     coord.request_stop()
    #     coord.join(threads)
