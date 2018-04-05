import os,time,pdb,argparse,threading,ang_loss
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
from ops import *
from utils import *
from random import shuffle
from network import networks
from scipy import ndimage
class DCGAN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=32,num_block=1,ir_image_shape=[256, 256,1], normal_image_shape=[256, 256, 3],\
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
	self.input_type = 'multi' #multi frequency
        self.list_val = [11,16,21,22,33,36,38,53,59,92]
	self.pair = False
	self.build_model()
	
    def build_model(self):
	
	if not self.use_queue:

        	self.low_ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_ir_image_shape,
                                    name='low_ir_images')
        	self.low_normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_normal_image_shape,
                                    name='low_normal_images')
	else:
		print ' using queue loading'
		if self.input_type == 'single':
		    self.image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
	 	    self.normal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
                    q = tf.RandomShuffleQueue(1000,100,[tf.float32,tf.float32],[[self.ir_image_shape[0],self.ir_image_shape[1],1],[normal_image_shape[0],self.normal_image_shape[1],3]])
		    self.enqueue_op = q.enqueue([self.image_single,self.normal_image_single])
		    self.images,self.normal_images = q.dequeue_many(self.batch_size)

		else:
		    self.nondetail_image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
		    self.detail_image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
		    self.nondetailnormal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
		    self.detailnormal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
	 	    self.normal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
	
		    q = tf.RandomShuffleQueue(1000,100,[tf.float32,tf.float32,tf.float32,tf.float32 ,tf.float32],[[self.ir_image_shape[0],self.ir_image_shape[1],1],[self.ir_image_shape[0],self.ir_image_shape[1],1],[self.normal_image_shape[0],self.normal_image_shape[1],3],[self.normal_image_shape[0],self.normal_image_shape[1],3],[self.normal_image_shape[0],self.normal_image_shape[1],3]])
		    self.enqueue_op = q.enqueue([self.nondetail_image_single,self.detail_image_single,self.nondetailnormal_image_single,self.detailnormal_image_single,self.normal_image_single])
		    self.nondetail_images,self.detail_images,self.nondetail_normal,self.detail_normal,self.normal_images = q.dequeue_many(self.batch_size)

	self.keep_prob = tf.placeholder(tf.float32)
	net  = networks(64,self.df_dim)
	if self.input_type is 'single':
       	    self.G = net.generator(self.images) 
	    self.G = self.G[-1]
	else:
       	    self.nondetail_G,self.detail_G = net.multi_freq_generator(self.nondetail_images,self.detail_images) 
	    self.G = self.nondetail_G[-1] + self.detail_G[-1]

	######## evaluation #######

	if self.input_type == 'single':
	    self.sample_G = tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler') 

	else:
	    self.sample_low= tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler_low')
	    self.sample_high= tf.placeholder(tf.float32,shape=[1,600,800,1],name='sampler_high')
	    self.sample_low_G,self.sample_high_G =net.multi_freq_sampler(self.sample_low,self.sample_high)
	    self.sample_G = self.sample_low_G[-1] + self.sample_high_G[-1]	
	################ Discriminator Loss ######################
        if self.input_type=='single':
            self.D = net.discriminator(tf.concat(3,[self.images,self.normal]),self.keep_prob)
	    self.D_  = net.discriminator(tf.concat(3,[self.images,self.G[-1]]),self.keep_prob,reuse=True)


        else:
	    if self.pair: 
                self.nondetail_D = net.discriminator_low(tf.concat(3,[self.nondetail_images,self.nondetail_normal]),self.keep_prob)
	        self.nondetail_D_  = net.discriminator_low(tf.concat(3,[self.nondetail_images,self.nondetail_G[-1]]),self.keep_prob,reuse=True)
	        self.detail_D = net.discriminator_high(tf.concat(3,[self.detail_images,self.detail_normal]),self.keep_prob)
	        self.detail_D_  = net.discriminator_high(tf.concat(3,[self.detail_images,self.detail_G[-1]]),self.keep_prob,reuse=True)

	    else:
	        self.nondetail_D = net.discriminator_low(self.nondetail_normal,self.keep_prob)
	        self.nondetail_D_  = net.discriminator_low(self.nondetail_G[-1],self.keep_prob,reuse=True)
	        self.detail_D = net.discriminator_high(self.detail_normal,self.keep_prob)
	        self.detail_D_  = net.discriminator_high(self.detail_G[-1],self.keep_prob,reuse=True)

	#### entire resolution ####
	if self.input_type=='single':
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones.like(self.D[-1]), self.D[-1])
            self.loss_fake = binary_cross_entropy_with_logits(tf.zeros.like(self.D_[-1]), self.D_[-1])
            self.d_loss = self.d_loss_real + self.d_loss_fake 

	else:
	    #### nondetail resolution ####
	    self.nondetail_d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.nondetail_D[-1].get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0), self.nondetail_D[-1])
            self.nondetail_d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.nondetail_D[-1].get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.nondetail_D_[-1])
            self.nondetail_d_loss = self.nondetail_d_loss_real + self.nondetail_d_loss_fake 

            #### detail resolution ####
	    self.detail_d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.detail_D[-1].get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0),self.detail_D[-1])
            self.detail_d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.detail_D[-1].get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.detail_D_[-1])
            self.detail_d_loss = self.detail_d_loss_real + self.detail_d_loss_fake 

	
	########################## Generative loss ################################
	self.ang_loss = ang_loss.ang_error(self.G,self.normal_images)
	
	if self.loss == 'L1':
            if self.input_type == 'single':
                self.L_loss = tf.reduce_mean(tf.abs(tf.sub(self.G,self.normal_images)))

	    else:
                self.nondetail_L_loss = tf.reduce_mean(tf.abs(tf.sub(self.nondetail_G[-1],self.nondetail_normal)))
                self.detail_L_loss = tf.reduce_mean(tf.abs(tf.sub(self.detail_G[-1],self.detail_normal)))
                self.L_loss = tf.reduce_mean(tf.abs(tf.sub(self.G,self.normal_images)))
	else:
            if self.input_type == 'single':
                self.L_loss = tf.reduce_mean(tf.square(self.G-self.normal_images))
            else:
                self.nondetail_L_loss = tf.reduce_mean(tf.square(self.nondetail_G[-1]-self.nondetail_normal))
                self.detail_L_loss =tf.reduce_mean(tf.square(self.detail_G[-1]-self.detail_normal))
                self.L_loss = tf.reduce_mean(tf.square(self.G-self.normal_images))

	if self.input_type == 'single':
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_[-1]), self.D_[-1])
            self.gen_loss = self.g_loss + (self.L_loss+self.ang_loss)*100
	    t_vars = tf.trainable_variables()
            
	    t_vars = tf.trainable_variables()
	    self.g_vars =[var for var in t_vars if 'g' in var.name]
	    self.d_vars =[var for var in t_vars if 'dis' in var.name]

	else:	
            self.nondetail_g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.nondetail_D_[-1]), self.nondetail_D_[-1])
            self.detail_g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.detail_D_[-1]), self.detail_D_[-1])
            self.nondetail_gen_loss = self.nondetail_g_loss + (self.nondetail_L_loss+self.L_loss+self.ang_loss)*100
            self.detail_gen_loss = self.detail_g_loss + (self.detail_L_loss)*1000+(self.L_loss+self.ang_loss)*1000
	
            t_vars = tf.trainable_variables()
	    self.nondetail_d_vars =[var for var in t_vars if 'low_dis' in var.name]
	    self.nondetail_g_vars =[var for var in t_vars if 'low_g' in var.name]
	    self.detail_d_vars =[var for var in t_vars if 'high_dis' in var.name]
	    self.detail_g_vars =[var for var in t_vars if 'high_g' in var.name]

	self.saver = tf.train.Saver(max_to_keep=20)
    def train(self, config):
        #####Train DCGAN####

	if self.input_type == 'single':
            global_step1 = tf.Variable(0,name='global_step1',trainable=False)
            global_step2 = tf.Variable(0,name='global_step2',trainable=False)
            d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step1,var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step2,var_list=self.g_vars)

	else:
            global_step1 = tf.Variable(0,name='global_step1',trainable=False)
            global_step2 = tf.Variable(0,name='global_step2',trainable=False)
            global_step3 = tf.Variable(0,name='global_step3',trainable=False)
            global_step4 = tf.Variable(0,name='global_step4',trainable=False)

  	    nondetail_d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.nondetail_d_loss, global_step=global_step1,var_list=self.nondetail_d_vars)
            nondetail_g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.nondetail_gen_loss, global_step=global_step2,var_list=self.nondetail_g_vars)
	
	    detail_d_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.detail_d_loss, global_step=global_step3,var_list=self.detail_d_vars)
            detail_g_optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1) \
                          .minimize(self.detail_gen_loss, global_step=global_step4,var_list=self.detail_g_vars)


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

	shuf = range(len(data))
	random.shuffle(shuf)

	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =1
	    if self.input_type =='single':
                for i in range(num_thread):
 	            t = threading.Thread(target=self.load_and_enqueue_single,args=(coord,train_input,train_gt,shuf,i,num_thread))
	 	    t.start()

	    else:
                for i in range(num_thread):
 	            t = threading.Thread(target=self.load_and_enqueue_multi,args=(coord,train_input,train_gt,shuf,i,num_thread))
	 	    t.start()

	if self.use_queue:
	    for epoch in xrange(config.epoch):
	        #shuffle = np.random.permutation(range(len(data)))
	        batch_idxs = min(len(data), config.train_size)/config.batch_size

		if self.input_type is 'single':
                    for idx in xrange(0,batch_idxs):
        	        start_time = time.time()
		        _,d_err =self.sess.run([d_optim,self.d_loss],feed_dict={self.keep_prob:self.dropout})
		        _,g_err,L_err,ang_err =self.sess.run([g_optim,self.g_loss,self.L_loss,self.ang_loss])
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L: %.6f d_loss:%.4f ang_loss:%.6f" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,g_err,L_err,ang_err))

                        if np.mod(global_step1.eval(),4000) ==0 and global_step1 != 0:
	                     self.save(config.checkpoint_dir,global_step1)


                else:
  		    for idx in xrange(0,batch_idxs):
        	        start_time = time.time()
		        _,nondetail_d_err =self.sess.run([nondetail_d_optim,self.nondetail_d_loss],feed_dict={self.keep_prob:self.dropout})
		        _,nondetail_g_err,nondetail_L_err =self.sess.run([nondetail_g_optim,self.nondetail_g_loss,self.nondetail_L_loss])
	
		        _,detail_d_err =self.sess.run([detail_d_optim,self.detail_d_loss],feed_dict={self.keep_prob:self.dropout})
		        _,detail_g_err,detail_L_err,L_err,ang_err = self.sess.run([detail_g_optim,self.detail_g_loss,self.detail_L_loss,self.L_loss,self.ang_loss])

		        print("Epoch: [%2d] [%4d/%4d] time: %4.4f nondetail_g_loss: %.6f nondetail_L: %.6f nondetail_d_loss:%.4f detail_g_loss:%.6f detail_d_loss:%.6f detail_L_loss: %.6f, L_loss: %.6f ang_loss:%.6f" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,nondetail_g_err,nondetail_L_err,nondetail_d_err,detail_g_err,detail_d_err,detail_L_err,L_err,ang_err))

                        if np.mod(global_step1.eval(),4000) ==0 and global_step1 != 0:
	                     self.save(config.checkpoint_dir,global_step1)
	else:
            print('Only multi-thread support \n') 
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

	
    def load_and_enqueue_single(self,coord,file_list,label_list,shuf,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	rot=[0,90,180,270]
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
	    r = random.randint(0,3)
            input_img = scipy.misc.imread(file_list[shuf[i]][0].encode("utf-8")).reshape([224,224,1]).astype(np.float32)
	    gt_img = scipy.misc.imread(label_list[shuf[i]][0].encode("utf-8")).reshape([224,224,3]).astype(np.float32)
	    input_img = input_img/127.5 -1.
	    gt_img = gt_img/127.5 -1.
	
	    rand_x = np.random.randint(64,224-64)
	    rand_y = np.random.randint(64,224-64)
	    ipt =  input_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    label = gt_img[rand_y:rand_y+64,rand_x:rand_x+64,:]

            self.sess.run(self.enqueue_op,feed_dict={self.image_single:ipt,self.normal_image_single:label})
	    count +=1
    
    def load_and_enqueue_multi(self,coord,file_list,label_list,shuf,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	rot=[0,90,180,270]
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
	    r = random.randint(0,3)
            input_img = scipy.misc.imread(file_list[shuf[i]][0].encode("utf-8")).reshape([224,224,1]).astype(np.float32)
	    gt_img = scipy.misc.imread(label_list[shuf[i]][0].encode("utf-8")).reshape([224,224,3]).astype(np.float32)
            low_input = ndimage.gaussian_filter(input_img,sigma=(1,1,0),order=0)	    
            low_gt = ndimage.gaussian_filter(gt_img,sigma=(1,1,0),order=0)	    
	    input_img = input_img/127.5 -1.
	    gt_img = gt_img/127.5 -1.
	    low_input = low_input/127.5 -1.
	    low_gt = low_gt/127.5 -1.
	    high_input = input_img - low_input
	    high_gt = gt_img - low_gt
	
	    rand_x = np.random.randint(64,224-64)
	    rand_y = np.random.randint(64,224-64)
	    ipt =  input_img[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    label = gt_img[rand_y:rand_y+64,rand_x:rand_x+64,:]

	    low_ipt =  low_input[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    low_label = low_gt[rand_y:rand_y+64,rand_x:rand_x+64,:]

	    high_ipt =  high_input[rand_y:rand_y+64,rand_x:rand_x+64,:]
	    high_label = high_gt[rand_y:rand_y+64,rand_x:rand_x+64,:]
            self.sess.run(self.enqueue_op,feed_dict={self.nondetail_image_single:low_ipt,self.detail_image_single:high_ipt,self.nondetailnormal_image_single:low_label,self.detailnormal_image_single:high_label,self.normal_image_single:label})
	    count +=1
		
