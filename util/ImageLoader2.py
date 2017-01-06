import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import queue_runner
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

INPUT_PATH_TRAIN = './res/lp_tt.lst'
INPUT_PATH_VAL = './res/lp_val.lst'

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 1  #

def processing_image(image):
  imgG = tf.image.rgb_to_grayscale(image, name=None)
  print imgG.get_shape()[:-1]
  imgW = tf.image.per_image_whitening(imgG)
  imgR = tf.image.resize_image_with_crop_or_pad(imgW, 20, 100)
  return imgR

def read_my_file_format(filename_queue):
  fName = filename_queue[0]
  label = filename_queue[1]
  file_contents = tf.read_file(fName)
  img = tf.image.decode_png(file_contents, 3)
  return img, label

def get_input(image_list_file):
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

    return filenames, labels

def input_pipeline(path_to_list, batch_size, num_epochs=None):
  image_list, label_list = get_input(path_to_list)
  images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)
  input_queue = tf.train.slice_input_producer([images, labels],
                                              num_epochs=num_epochs,
                                              shuffle=True)
  image, label = read_my_file_format(input_queue)

  pr_image = processing_image(image)

  image_batch, label_batch = tf.train.batch([pr_image, label], batch_size=batch_size)

  ## Display the training images in the visualizer.
  #tensor_name = image.op.name
  #tf.image_summary(tensor_name + 'images', image_batch)
  return image_batch, label_batch







####Load data
print('Loading data')
#fileNames, labels  = get_input(INPUT_PATH_TRAIN)
#print fileNames
ex, lab = input_pipeline(INPUT_PATH_TRAIN, 2)

#rgb_ex = tf.image.grayscale_to_rgb(ex, name=None)

init_op = tf.initialize_all_variables()
sess = tf.Session()
with sess.as_default():
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for i in range(11):
    ex1, lab1 = sess.run([ex, lab])
    print lab1
  print lab1[1]
  print(ex1.shape)
  i = ex1.squeeze()
  print(i.shape)
  #print ii.shape
  ii = np.asarray(i[1])
  #Image._show(Image.fromarray(i))
  #print i
  plt.imshow(ii)
  plt.show()

  coord.request_stop()
  coord.join(threads)