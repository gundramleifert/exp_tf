# -*- coding: utf-8 -*-
import tensorflow as tf
import os.path


class PrefixSaver:
    def __init__(self, prefix, dir_to_model, name='model', write_meta_graph=False):
        self.prefix = prefix
        if not dir_to_model:
            raise RuntimeError('no folder given where variables should be saved.')
        if dir_to_model[-1] != '/':
            dir_to_model += '/'
        self.dir_to_model = str(dir_to_model)
        self.name = str(name)
        # print(self.name)
        self.write_meta_graph = write_meta_graph
        self.saver = None

    def save(self, session, global_step=None):
        if not self.saver:
            self.saver = tf.train.Saver(get_op(self.prefix))
        if not os.path.isdir(self.dir_to_model):
            os.mkdir(self.dir_to_model)
        self.saver.save(session, os.path.join(self.dir_to_model, self.name), global_step)

    def restore(self, session):
        if not self.saver:
            self.saver = tf.train.Saver(get_op(self.prefix))
        to_restore = tf.train.get_checkpoint_state(self.dir_to_model)
        if not to_restore:
            raise RuntimeError("in folder '{}' no variables found named '{}'.".format(self.dir_to_model,self.name))
        self.saver.restore(session, to_restore.model_checkpoint_path)


def get_op(prefix=None):
    """
    Returns all variable of the default tensorflow graph with the given prefix. The return value is a dictionary 'NAME_OF_VARIABLE' => 'VARIABLE'. If a prefix is given, the prefix is deleted in 'NAME_OF_VARIABLE'.
    :rtype: dictionary 'string'=>tensor
    """
    dict = {}
    if prefix is not None and len(prefix) > 1:
        if prefix[-1] != '/':
            prefix += '/'
    res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
    for t in res:
        key = t.name
        key = key[len(prefix):]
        dict[str(key)] = t
    return dict
