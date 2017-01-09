# from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import STR2CTC
from util.CharacterMapper import get_cm_lp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  return images, label_batch

def get_input(image_list_file, cm):
    f = open(image_list_file, 'r')
    imageFileNames = []
    listOfTgtVals = []
    for line in f:
        imageFileNames.append(line[:-1])
        labelFile = line[:-1] + ".txt"
        tmp = open(labelFile, 'r')
        tgtStr = tmp.readline()
        tgtVals = STR2CTC.target_to_int_repr(tgtStr, cm)
        # listOfTgtVals.append(tgtVals[:3])
        listOfTgtVals.append(tgtVals)
        if tmp is not None:
            tmp.close()
    f.close()
    return imageFileNames, listOfTgtVals

def processing_image(image, width, height):
  imgG = tf.image.rgb_to_grayscale(image, name=None)
  imgW = tf.image.per_image_standardization(imgG)
  imgR = tf.image.resize_image_with_crop_or_pad(imgW, height, width)
  return imgR

def read_my_file_format_png(filename_queue):
  fName = filename_queue[0]
  label = filename_queue[1]
  file_contents = tf.read_file(fName)
  img = tf.image.decode_png(file_contents, 3)
  return img, label


def input_pipeline(image_list, batch_size, cm, width, height , min_queue_examples=160):
  imageFileNames, listOfTgtVals = get_input(image_list,cm)
  images = ops.convert_to_tensor(imageFileNames, dtype=dtypes.string)
  print(images.get_shape())
  labels = ops.convert_n_to_tensor(listOfTgtVals)
  labels = tf.pack(labels)
  print(labels.get_shape())
  input_queue = tf.train.slice_input_producer([images, labels])

  image, tgt = read_my_file_format_png(input_queue)
  pr_image = processing_image(image, width, height)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(pr_image, tgt,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

INPUT_PATH_TRAIN = './private/lists/lp_only_val.lst'
cm = get_cm_lp()
height = 48
width = 256

os.chdir("..")
####Load data
print('Loading data')
ex, lab = input_pipeline(INPUT_PATH_TRAIN, 16, cm, width, height)

init_op = tf.global_variables_initializer()
sess = tf.Session()
with sess.as_default():
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for i in range(11):
    ex1, lab1 = sess.run([ex, lab])
    print(lab1[0])
    plt.imshow(ex1[0,:,:,0], cmap=plt.cm.gray)
    plt.show()
  coord.request_stop()
  coord.join(threads)