import tensorflow as tf


def denselayer(scope, x, out_dim, nonlinearity=tf.identity, w_initializer=None, w_regularizer=None):

    x_shape = x.get_shape().as_list()
    with tf.variable_scope(scope):
        w = tf.get_variable('w', shape=[x_shape[1], out_dim], initializer=w_initializer, regularizer=w_regularizer)
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0), regularizer=None)
        o = nonlinearity(tf.matmul(x, w) + b)

    return o, [w, b]
