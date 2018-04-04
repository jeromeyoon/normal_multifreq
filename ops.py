import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *
import pdb
"""
class batch_norm(object):
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
"""

def batchnorm(input_):
    with tf.variable_scope('batchnorm_'):
        input_ = tf.identity(input_)
        channels = input_.get_shape()[3]
        offset = tf.get_variable('offset',[channels],dtype=tf.float32,initializer=tf.zeros_initializer)
        scale = tf.get_variable('scale',[channels],dtype=tf.float32,initializer=tf.random_normal_initializer(1.0,0.02))
	mean, variance = tf.nn.moments(input_,axes=[0,1,2],keep_dims=False)
        variance_epsilon =1e-5
	normalized = tf.nn.batch_normalization(input_,mean,variance,offset,scale,variance_epsilon=variance_epsilon)
	return normalized





def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=1, d_w=1, stddev=0.02,padding='SAME',
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_ch,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
	batch,in_hei,in_wid,in_ch = [int(d) for d in input_.get_shape()]
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_ch, in_ch],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[batch,in_hei*2,in_wid*2,output_ch],
                                strides=[1, d_h, d_w, 1],padding='SAME')

        #biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
