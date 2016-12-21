# exp_tf
experimental - first tests with tensorflow

INSTALL THE LATEST VERSION OF TENSORFLOW
(see also https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#using-pip)

1. Go into your tensorflow environment (source activate tensorflow)
2. Type in:
   export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
   (For newer versions get the latest url)
3. Execute the following command for Python 2:
   pip install --ignore-installed --upgrade $TF_BINARY_URL


HOW TO LOAD DATA

http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_input.py

http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels ==> maybe best

https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html ==> suggestion from Max


GET STARTED WITH TENSORFLOW

1. Start with Python Tutorial: https://docs.python.org/2.7/tutorial/

2. Continue NumPy Tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

3. Start with: https://www.tensorflow.org/versions/master/get_started/basic_usage.html

4. Try tutorial: https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html

5. API of TensorFlow including contributions: https://www.tensorflow.org/api_docs/python/

Other Tutorials and examples:
https://github.com/sherrym/tf-tutorial
https://github.com/aymericdamien/TensorFlow-Examples.git

==> do not try to understand anything without reading these in advance :-)



DISTRIBUTED TRAINING

maybe interesting for HPCs: https://www.tensorflow.org/versions/master/how_tos/distributed/index.html


COMMON ERRORS:
do not find tensorflow-sources/packages:
File=>Settings...=> Project: exp_tf => set default project interpreter => tensorflow

TensorBoard does not show the experiments
bash: >source activate tensorflow

mothods of scipy do not work ("AttributeError: 'module' object has no attribute'<method>'":
source activate tensorflow
conda install pillow

out of Memory using GPUs:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
or:
export CUDA_VISIBLE_DEVICES=1
