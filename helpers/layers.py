import tensorflow as tf
import numpy as np

def denselayer(scope, x, out_dim, nonlinearity=tf.identity, w_initializer=None, w_regularizer=None):

    x_shape = x.get_shape().as_list()
    with tf.variable_scope(scope):
        w = tf.get_variable('w', shape=[x_shape[1], out_dim], initializer=w_initializer, regularizer=w_regularizer)
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0), regularizer=None)
        o = nonlinearity(tf.matmul(x, w) + b)

    return o, [w, b]

def noisy_denselayer(scope, x, out_dim, nonlinearity=tf.identity, factorized=False, init_sigma=0.5):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(scope):
        if factorized:
            in_mult = 1./np.sqrt(x_shape[1])
            w_mean = tf.get_variable('w_mean', shape=[x_shape[1], out_dim], initializer=tf.random_uniform_initializer(-in_mult, in_mult))
            w_sigma = tf.get_variable('w_sigma', shape=[x_shape[1], out_dim], initializer=tf.constant_initializer(init_sigma/in_mult))
            b_mean = tf.get_variable('b_mean', shape=[out_dim], initializer=tf.constant_initializer(0))
            b_sigma = tf.get_variable('b_sigma', shape=[out_dim], initializer=tf.constant_initializer(init_sigma/in_mult))
        else:
            in_mult = np.sqrt(3) / np.sqrt(x_shape[1])
            w_mean = tf.get_variable('w_mean', shape=[x_shape[1], out_dim],
                                     initializer=tf.random_uniform_initializer(-in_mult, in_mult))
            w_sigma = tf.get_variable('w_sigma', shape=[x_shape[1], out_dim],
                                      initializer=tf.constant_initializer(0.017))
            b_mean = tf.get_variable('b_mean', shape=[out_dim], initializer=tf.constant_initializer(0))
            b_sigma = tf.get_variable('b_sigma', shape=[out_dim],
                                      initializer=tf.constant_initializer(0.017))

        if factorized:
            noise_input = tf.random_normal(shape=[x_shape[1], 1])
            noise_output = tf.random_normal(shape=[1, out_dim])
            noise_w = noise_input * noise_output
            noise_b = tf.reshape(noise_output, [-1])
        else:
            noise_w = tf.random_normal(shape=[x_shape[1], out_dim])
            noise_b = tf.random_normal(shape=[out_dim])
        w = w_mean + w_sigma * noise_w
        b = b_mean + b_sigma * noise_b
        o = nonlinearity(tf.matmul(x, w) + b)

    return o, [w_mean, w_sigma, b_mean, b_sigma]