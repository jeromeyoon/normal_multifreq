import os,time,pdb,argparse,threading,ang_loss,pickle
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
from ops import *
from utils import *
from random import shuffle
from network import networks
from scipy import ndimage
import scipy.io
################### ATTENTIION ################################################
####### This model contains non_detail, normal  Adversarial loss #####
###############################################################################
class DCGAN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=12,num_block=1,ir_image_shape=[256, 256,1], normal_image_shape=[256, 256, 3],\
	         light_shape=[64,64,3],df_dim=64,dataset_name='default',checkpoint_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
	self.num_block = num_block
        self.checkpoint_dir = checkpoint_dir
	self.ir_image_shape=[64,64,1]
	self.normal_image_shape=[64,64,3]
	self.use_queue = True
	self.mean_nir = -0.3313 #-1~1
	self.dropout =0.7
	self.loss ='L1'
	self.lambda_d = 1
	self.lambda_g_non = 1
	self.lambda_g_d  =1
	self.input_type = 'single' #multi frequency
        self.list_val = [11,16,21,22,33,36,38,53,59,92]
	self.pair = False
	self.build_model()
	
    def build_model(self):
       
        self.images = tf.placeholder(tf.float32,shape=[self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
        self.detail_images = tf.placeholder(tf.float32,shape=[self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
        self.nondetail_images = tf.placeholder(tf.float32,shape=[self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
	self.normal_images = tf.placeholder(tf.float32,shape=[self.batch_size,self.normal_image_shape[0],self.normal_image_shape[1],3])
	self.nondetail_normal = tf.placeholder(tf.float32,shape=[self.batch_size,self.normal_image_shape[0],self.normal_image_shape[1],3])
	self.keep_prob = tf.placeholder(tf.float32)
	net  = networks(64,self.df_dim)
       	self.nondetail_G,self.G = net.multi_freq_generator_skip(self.nondetail_images,self.detail_images) 
	self.G = self.g[-1]

	################ Discriminator Loss ######################
	if self.pair: 
            self.nondetail_D = net.discriminator_low(tf.concat(3,[self.nondetail_images,self.nondetail_normal]),self.keep_prob)
	    self.nondetail_D_  = net.discriminator_low(tf.concat(3,[self.nondetail_images,self.nondetail_G[-1]]),self.keep_prob,reuse=True)
            self.D = net.discriminator(self.normal_images,self.keep_prob)
	    self.D_  = net.discriminator(self.G,self.keep_prob,reuse=True)
	else:
	    self.nondetail_D = net.discriminator_low(self.nondetail_normal,self.keep_prob)
	    self.nondetail_D_  = net.discriminator_low(self.nondetail_G[-1],self.keep_prob,reuse=True)
            self.D = net.discriminator(self.normal_images,self.keep_prob)
	    self.D_  = net.discriminator(self.G,self.keep_prob,reuse=True)
	    
        #### nondetail resolution ####
	self.nondetail_d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.nondetail_D[-1].get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0), self.nondetail_D[-1])
        self.nondetail_d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.nondetail_D[-1].get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.nondetail_D_[-1])
        self.nondetail_d_loss = self.nondetail_d_loss_real + self.nondetail_d_loss_fake 

        self.nondetail_d_loss_real_sum = tf.summary.scalar("nondetail_d_loss_real",self.nondetail_d_loss_real)
        self.nondetail_d_loss_fake_sum = tf.summary.scalar("nondetail_d_loss_fake",self.nondetail_d_loss_fake)
        self.nondetail_d_loss_sum = tf.summary.scalar("nondetail_d_loss",self.nondetail_d_loss)
        
        #### detail resolution ####
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D[-1]), self.D[-1])
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_[-1]), self.D_[-1])
        self.d_loss = self.d_loss_real + self.d_loss_fake 
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real",self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake",self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss",self.d_loss)

	########################## Generative loss ################################
	self.ang_loss = ang_loss.ang_error(self.G,self.normal_images)
        self.ang_loss_sum = tf.summary.scalar("ang_loss",self.ang_loss)
	
	if self.loss == 'L1':
           self.nondetail_L_loss = tf.reduce_mean(tf.abs(tf.subtract(self.nondetail_G[-1],self.nondetail_normal)))
           self.L_loss = tf.reduce_mean(tf.abs(tf.subtract(self.G,self.normal_images)))
	   self.nondetail_L_loss_sum = tf.summary.scalar("nondetail_L1_loss",self.nondetail_L_loss)
	   self.L_loss_sum = tf.summary.scalar("L1_loss",self.L_loss)
        else:
           self.nondetail_L_loss = tf.reduce_mean(tf.square(self.nondetail_G[-1]-self.nondetail_normal))
           self.L_loss = tf.reduce_mean(tf.square(self.G-self.normal_images))

	   self.nondetail_L_loss_sum = tf.summary.scalar("nondetail_L2_loss",self.nondetail_L_loss)
	   self.L_loss_sum = tf.summary.scalar("L2_loss",self.L_loss)

        self.nondetail_g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.nondetail_D_[-1]), self.nondetail_D_[-1])
        self.nondetail_g_sum = tf.summary.scalar("low_g_loss",self.nondetail_g_loss)

        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_[-1]), self.D_[-1])
        self.g_loss_sum = tf.summary.scalar("g_loss",self.g_loss)
        self.nondetail_gen_loss = self.nondetail_g_loss +self.g_loss + (self.nondetail_L_loss+self.L_loss+self.ang_loss)*self.lambda_g_non
        self.nondetail_gen_loss_sum = tf.summary.scalar("nondetail_gen_loss",self.nondetail_gen_loss)
        self.gen_loss = self.g_loss + self.L_loss+self.ang_loss
        self.gen_loss_sum = tf.summary.scalar("gen_loss",self.gen_loss)

        t_vars = tf.trainable_variables()
        self.nondetail_d_vars =[var for var in t_vars if 'low_dis' in var.name]
        self.nondetail_g_vars =[var for var in t_vars if 'low_g' in var.name]
        self.detail_d_vars =[var for var in t_vars if 'high_dis' in var.name]
        self.detail_g_vars =[var for var in t_vars if 'high_g' in var.name]

	self.saver = tf.train.Saver(max_to_keep=20)
    def train(self, config):
        #####Train DCGAN####
        global_step1 = tf.Variable(0,name='global_step1',trainable=False)
        global_step2 = tf.Variable(0,name='global_step2',trainable=False)
        global_step3 = tf.Variable(0,name='global_step3',trainable=False)
        global_step4 = tf.Variable(0,name='global_step4',trainable=False)

  	nondetail_d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                         .minimize(self.nondetail_d_loss, global_step=global_step1,var_list=self.nondetail_d_vars)
        nondetail_g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.nondetail_gen_loss, global_step=global_step2,var_list=self.nondetail_g_vars)
	
	d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step3,var_list=self.detail_d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step4,var_list=self.detail_g_vars)

	self.g_sum = tf.summary.merge([self.g_loss_sum,self.nondetail_g_sum,self.L_loss_sum,self.ang_loss_sum,self.nondetail_gen_loss_sum])
	self.d_sum = tf.summary.merge([self.d_loss_sum,self.nondetail_d_loss_sum,self.nondetail_d_loss_real_sum,self.nondetail_d_loss_fake_sum,self.d_loss_fake_sum,self.d_loss_real_sum,self.d_loss_sum])


	self.writer = tf.summary.FileWriter("./logs_multifreq_skip3", self.sess.graph)
        try:
	    tf.global_variables_initializer().run()
	except:
	    tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path

        data = json.load(open("/research2/ECCV_journal/trainingdata_json/0326/traininput.json"))
        data_label = json.load(open("/research2/ECCV_journal/trainingdata_json/0326/traingt.json"))
	train_input =[data[idx] for idx in xrange(0,len(data))]
	train_gt =[data_label[idx] for idx in xrange(0,len(data))]

	counter = 1;
        for epoch in xrange(config.epoch):
	    shuffle = np.random.permutation(range(len(train_input)))
	    batch_idxs = min(len(train_input), config.train_size)/config.batch_size

            for idx in xrange(0,batch_idxs):
                batch_files = shuffle[idx*config.batch_size:(idx+1)*config.batch_size]
                batches = [get_image(train_input[batch_file], train_gt[batch_file],self.image_size,is_crop=self.is_crop) for batch_file in batch_files]
                batches = np.array(batches).astype(np.float)
		batch_images = np.array(batches[:,:,:,0]).astype(np.float32)
	        batch_images = batch_images.reshape([self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
                batch_high_images = np.array(batches[:,:,:,1]).astype(np.float32)
	        batch_high_images = batch_high_images.reshape([self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
                batch_low_images = np.array(batches[:,:,:,2]).astype(np.float32)
	        batch_low_images = batch_low_images.reshape([self.batch_size,self.ir_image_shape[0],self.ir_image_shape[1],1])
                batch_labels = np.array(batches[:,:,:,3:6]).astype(np.float32)
                batch_high_labels = np.array(batches[:,:,:,6:9]).astype(np.float32)
                batch_low_labels = np.array(batches[:,:,:,9:12]).astype(np.float32)
                start_time = time.time()
                
                _,_,summary,d_err = self.sess.run([nondetail_d_optim,d_optim,self.d_sum,self.d_loss],feed_dict={self.nondetail_images:batch_low_images,self.detail_images:batch_high_images,self.images:batch_images,\
			    self.nondetail_normal:batch_low_labels,self.detail_normal:batch_high_labels,self.normal_images:batch_labels,self.keep_prob:self.dropout})
		self.writer.add_summary(summary, counter)

       	        _,_,summary,g_err =self.sess.run([nondetail_g_optim,g_optim,self.g_sum,self.g_loss],feed_dict={self.nondetail_images:batch_low_images,self.detail_images:batch_high_images,self.images:batch_images,\
			    self.nondetail_normal:batch_low_labels,self.normal_images:batch_labels,self.keep_prob:self.dropout})
	           
		self.writer.add_summary(summary, counter)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f d_loss:%.4f \n" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,g_err,d_err))

                if np.mod(global_step1.eval(),4000) ==0 and global_step1 != 0:
	           self.save(config.checkpoint_dir,global_step1)

    		counter = counter+1
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

