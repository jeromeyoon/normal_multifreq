import pdb
import numpy as np 
from numpy import inf
import tensorflow as tf

def ang_error(samples,gt_samples):

    [batch,h,w,c] = samples.get_shape().as_list()
    #valid_pixel = b*h*w
    output = l2_normalize(samples)
    gt_output = l2_normalize(gt_samples)
    tmp = tf.reduce_sum(tf.multiply(output,gt_output),3)
    output = tf.reduce_sum(tf.subtract(tf.ones_like(tmp,dtype=tf.float32),tmp),[1,2])
    output = output/(h*w)
    return tf.reduce_sum(output)/batch

def l2_normalize(input_):
    tmp1 = tf.square(input_)
    tmp2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(tmp1,3)),-1)
    tmp3 = tf.div(input_,tmp2)
    return tmp3


def ang_error_py(sample,gt):
    sample = np.clip(sample,np.exp(-10),1.0)
    gt = np.clip(gt,np.exp(-10),1.0)
    output = l2_normalize_py(sample)
    gt_output = l2_normalize_py(gt) 
    tmp  = (output * gt_output)
    tmp  = tmp.sum(axis =-1)
    tmp2  = np.subtract(np.ones((600,800),dtype=np.float),tmp)
    tmp2 = tmp2 * mask	 
    output = np.sum(output,dtype=np.float)
    output = output/nonzeros
    return output 
	  
def l2_normalize_py(input_):  
    tmp1 = np.square(input_)
    tmp2 = np.expand_dims(np.sqrt(np.sum(tmp1,axis=-1)),axis=-1) 
    tmp3 = input_/tmp2
    return tmp3

