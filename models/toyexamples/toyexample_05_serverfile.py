import tensorflow as tf
import sys


def get_cluster_spec():
    return tf.train.ClusterSpec({
        "worker": [
            # "139.30.31.218:2222",  # citlab
            "139.30.31.13:2222", # citlab 2
            "139.30.31.13:2224", # citlab 2
            # "139.30.31.187:8888:2222", # citlab 3
        ],
        "ps": [
            "139.30.31.13:2223", # citlab 2
            # "139.30.31.176:2222"  # gundram
        ]})


if __name__ == '__main__':
    spec = get_cluster_spec()
    for arg in sys.argv[1:]:
        print arg
    jobname = sys.argv[1]
    taskindex = int(sys.argv[2])
    print("job_name = '{}'".format(jobname))
    print("task_index = '{}'".format(taskindex))
    server = tf.train.Server(spec, job_name=jobname, task_index=taskindex)
