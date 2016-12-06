from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import os

num_epochs = 10
filename = "./resources/list_all.txt"

class ImageLoader:

    def __init__(self, path, batchSize):
        self._path = path
        self._batchSize = batchSize
        self._noImages = 0
        self._index_in_epoch = 0
        self._epochs_completed = 0

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

        self._noImages = len(filenames)
        self._fileNames = filenames
        self._labels = labels

    def read_images_from_disk(self, input_queue):

        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_png(file_contents, channels=3)
        return example, label

    def getNextBatch(self):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # load #batchSize images & labels and return as tensors
        fileNamesBatch = self._fileNames[self._index_in_epoch:self._index_in_epoch + self._batchSize]
        labelsBatch = self._labels[self._index_in_epoch:self._index_in_epoch + self._batchSize]

        # need to load/convert images as tensors
        filenameQueue = tf.train.string_input_producer(fileNamesBatch)

        reader = tf.WholeFileReader()
        tmpList = []
        for _ in xrange(self._batchSize):
            imgKey, imgValue = reader.read(filenameQueue)
            img = tf.image.decode_jpeg(imgValue)
            tmpList.append(img)
        images = tf.convert_to_tensor(tmpList)
        labels = tf.convert_to_tensor(labelsBatch, dtype=tf.string)

        self._index_in_epoch += self._batchSize

        # Epoch completed
        if self._index_in_epoch >= self._noImages:
            self._epochs_completed += 1
            self._index_in_epoch = 0

            # Shuffle data
            perm = np.arange(self._noImages)
            np.random.shuffle(perm)
            self._fileNames = self._fileNames[perm]
            self._labels = self._labels[perm]

        return images, labels



if __name__ == '__main__':
    os.chdir("..")

    loader = ImageLoader(filename)

    # # Reads pathes of images together with their labels
    # image_list, label_list = loader.read_labeled_image_list(filename)
    #
    # images = tf.convert_to_tensor(image_list, dtype=tf.string)
    # labels = tf.convert_to_tensor(label_list, dtype=tf.string)
    #
    # # Makes an input queue
    # input_queue = tf.train.slice_input_producer([images, labels],
    #                                             num_epochs=num_epochs,
    #                                             shuffle=True)
    #
    image, label = loader.read_images_from_disk(input_queue)


    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range():  # length of your filename list
        image = my_img.eval()  # here is your image Tensor :)

    print(image.shape)
    Image.show(Image.fromarray(np.asarray(image)))

    coord.request_stop()
    coord.join(threads)


    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    # image = preprocess_image(image)
    # label = preprocess_label(label)

    # Optional Image and Label Batching
    # image_batch, label_batch = tf.train.batch([image, label],
    #                                           batch_size=batch_size)