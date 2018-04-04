import os,time
from glob import glob
import tensorflow as tf
from ops import *
from utils import *
from network import networks
class EVAL(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=1, num_block=1,ir_image_shape=[64, 64, 1],
                 df_dim=64,dataset_name='default',checkpoint_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
	self.num_block = num_block
        self.build_model()

    def build_model(self):

        
	self.nondetail_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='nondetail_images')
	self.detail_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='detail_images')

	net  = networks(64,64)
        self.low_G,self.high_G = net.generator(self.nondetail_images,self.detail_images)
        self.G = self.high_G[-1] + self.low_G[-1]	
        self.saver = tf.train.Saver()


    def load(self, checkpoint_dir,model):
        print(" [*] Reading checkpoints...")

        #model_dir = "%s_%s" % (self.dataset_name, 32)
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	#model_path = os.path.join(checkpoint_dir,model)
	if os.path.isfile(os.path.join(checkpoint_dir,model)):
	    print(' Success load network ')
	    self.saver.restore(self.sess, os.path.join(checkpoint_dir, model))
	    return True
	else:
	    print('Fail to load network')
	    return False
