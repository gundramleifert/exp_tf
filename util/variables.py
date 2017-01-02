from __future__ import print_function
import tensorflow as tf

def get_uninitialized_variables(variables=None):
    """Get uninitialized variables as a list.

    Parameters
    ----------
    variables : collections.Iterable[tf.Variable]
        Return only uninitialized variables within this collection.
        If not specified, will return all uninitialized variables.

    Returns
    -------
    list[tf.Variable]
    """
    sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    init_flag = sess.run(
        tf.pack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]