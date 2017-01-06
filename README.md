# exp_tf
experimental - first tests with tensorflow

<b> INSTALL THE LATEST VERSION OF TENSORFLOW </b>

(see also https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#using-pip)
<ol>
    <li>
    Go into your tensorflow environment (source activate tensorflow)
    </li>
    <li>
    Type in:<br>
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl <br>
    (For newer versions get the latest url)
    </li>
    <li>
    Execute the following command for Python 2: <br>
    pip install --ignore-installed --upgrade $TF_BINARY_URL
    </li>
</ol>

<b> HOW TO LOAD DATA </b>

http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_input.py

http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels ==> maybe best

https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html ==> suggestion from Max

<b> GET STARTED WITH TENSORFLOW </b>
<ol>
    <li>
    Start with Python Tutorial: https://docs.python.org/2.7/tutorial/
    </li>
    <li>
    Continue NumPy Tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
    </li>
    <li>
    Start with: https://www.tensorflow.org/versions/master/get_started/basic_usage.html
    </li>
    <li>
    Try tutorial: https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html
    </li>
    <li>
    API of TensorFlow including contributions: https://www.tensorflow.org/api_docs/python/
    </li>
</ol>

<b> Other Tutorials and examples: </b>

https://github.com/sherrym/tf-tutorial

https://github.com/aymericdamien/TensorFlow-Examples.git

<b> Saving images in Tensorboard: </b>

http://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots

==> do not try to understand anything without reading these in advance :-)


<b> DISTRIBUTED TRAINING </b>

maybe interesting for HPCs: https://www.tensorflow.org/versions/master/how_tos/distributed/index.html


<b> COMMON ERRORS </b>

<ul>
    <li>
    Do not find tensorflow-sources/packages:
    File=>Settings...=> Project: exp_tf => set default project interpreter => tensorflow
    <li>
    TensorBoard does not show the experiments
    bash: >source activate tensorflow
    </li>
    <li>
    Methods of scipy do not work ("AttributeError: 'module' object has no attribute '&lt;method&gt;'"):
    <ul>
        <li> source activate tensorflow </li>
        <li> conda install pillow </li>
        <li> export PYTHON_PATH=<path_to_sources>:$PYTHON_PATH
    </ul>
    </li>
    <li>
    out of Memory using GPUs:
    <ul>
        <li> gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) </li>
        <li> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) </li>
    </ul>
    or:
    export CUDA_VISIBLE_DEVICES=1
    </li>
</ul>

<b> KEYWORDS FOR CLOSING ISSUES </b>

The following keywords will close an issue via commit message:
<ul>
    <li> close, closes , closed </li>
    <li> fix, fixes, fixed </li>
    <li> resolve, resolves, resolved </li>
</ul>

See also: https://help.github.com/articles/closing-issues-via-commit-messages/