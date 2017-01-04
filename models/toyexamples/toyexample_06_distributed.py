import tensorflow as tf
from toyexample_05_distributed import ModelHandler
from tensorflow.examples.tutorials.mnist import input_data
import toyexample_05_serverfile as serverfile

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    # if  len(ps_hosts)==0:
    if len(ps_hosts) == 1 and ps_hosts[0] == '':
        cluster = serverfile.get_cluster_spec()
        print("use predefined cluster specification")
    else:
        cluster_spec = {"ps": ps_hosts, "worker": worker_hosts}
        print(cluster_spec)
        cluster = tf.train.ClusterSpec(cluster_spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    print("job_name = {}".format(FLAGS.job_name))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        mnist = input_data.read_data_sets("private/resources/MNIST_data/", one_hot=True, reshape=False,
                                          validation_size=0)
        # initialization
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # placeholder for correct answers
        Y_ = tf.placeholder(tf.float32, [None, 10])

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # Data

            # Build model...
            mh = ModelHandler()
            global_step = tf.Variable(0)
            logit, Y = mh.inference(X)
            loss, acc = mh.loss(logit, Y, Y_, include_summary=True)
            # minimize = mh.training(loss)
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        print("is chief? {}".format(FLAGS.task_index == 0))
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        print("supervisor is set up")
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("session started...")
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 1000000:
                print("step {}".format(step))
                batch_X, batch_Y = mnist.train.next_batch(100)
                train_data = {X: batch_X, Y_: batch_Y}

                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                _, step = sess.run([train_op, global_step], feed_dict=train_data)

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
