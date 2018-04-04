import numpy as np
import os
import tensorflow as tf
import random,time,json,pdb,scipy.misc,glob
from model_queue import DCGAN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
from numpy import inf
from sorting import natsorted
import matplotlib.image as mpimg
from scipy import ndimage
flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "0422", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
flags.DEFINE_integer("num_block", 3, "The number of block for generator model")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
    width_size = 905
    height_size = 565
    #width_size = 1104
    #height_size = 764
    #width_size = 1123
    #height_size = 900
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    if not os.path.exists(os.path.join('./logs',time.strftime('%d%m'))):
    	os.makedirs(os.path.join('./logs',time.strftime('%d%m')))

    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    #with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,\
	    num_block = FLAGS.num_block,dataset_name=FLAGS.dataset,is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
	    dcgan = EVAL(sess, batch_size=1,num_block=FLAGS.num_block,ir_image_shape=[None,None,1],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
	    print('deep model test \n')

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            list_val = [11,16,21,22,33,36,38,53,59,92]
	    print '1: Estimating Normal maps from arbitary obejcts \n'
            print '2: EStimating Normal maps according to only object tilt angles(Light direction is fixed(EX:3) \n'
	    print '3: Estimating Normal maps according to Light directions and object tilt angles \n'
	    x = input('Selecting a Evaluation mode:')
            VAL_OPTION = int(x)

            if VAL_OPTION ==1: # arbitary dataset 
                print("Computing arbitary dataset ")
		trained_models = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		trained_models  = natsorted(trained_models)
		model = trained_models[6]
		model = model.split('/')
		model = model[-1]
		print('Load trained network: %s\n' %model)
		dcgan.load(FLAGS.checkpoint_dir,model)
		datapath = '/research3/datain/gmchoe_normal/0403/IR_0.25'
		#datapath = '/research3/dataout/ECCV_2018/2130/centerview'
		#datapath = '/research2/proposal_linux/dataset/coin'
                savepath = datapath
		mean_nir = -0.3313
		img_files = glob.glob(os.path.join(datapath,'*.png'))
		img_files = natsorted(img_files)
		pdb.set_trace()
		#listdir = natsorted(os.listdir(datapath))
		#fulldatapath = natsorted(fulldatapath)
                for idx in xrange(0,len(img_files)):
		    print('Processing %d/%d \n' %(len(img_files),idx))
		    #img_file = glob.glob(os.path.join(datapath,'nir.jpg'))
		    #img_file = glob.glob(os.path.join(datapath,listdir[idx]))
		    input_= scipy.misc.imread(img_files[idx],'F').astype(float)
		    height_size = input_.shape[0]
		    width_size = input_.shape[1]
	            #input_ = scipy.misc.imresize(input_,[565,905])
	            input_ = np.reshape(input_,(height_size,width_size,1)) # LF size:383 x 552 
		    #input_ = np.power(input_,0.6)
		    nondetail_input_ = ndimage.gaussian_filter(input_,sigma=(1,1,0),order=0)	   
		    input_ = input_/127.5 -1.0 
	            nondetail_input_  = nondetail_input_/127.5 -1.0 # normalize -1 ~1
	            detail_input_ = input_ - nondetail_input_
	            nondetail_input_ = np.reshape(nondetail_input_,(1,height_size,width_size,1)) # LF size:383 x 552 
	            detail_input_ = np.reshape(detail_input_,(1,height_size,width_size,1))
	            #detail_input_  = detail_input_/127.5 -1.0 # normalize -1 ~1
	            start_time = time.time() 
	            sample = sess.run(dcgan.G, feed_dict={dcgan.nondetail_images: nondetail_input_,dcgan.detail_images:detail_input_})
		    print('time: %.8f' %(time.time()-start_time))     
	            sample = np.squeeze(sample).astype(np.float32)

                    # normalization #
		    output = np.sqrt(np.sum(np.power(sample,2),axis=2))
		    output = np.expand_dims(output,axis=-1)
		    output = sample/output
		    output = (output+1.)/2.
		    """
		    if not os.path.exists(os.path.join(savepath,'%s/%s/%s' %(FLAGS.dataset,model,listdir[idx]))):
		        os.makedirs(os.path.join(savepath,'%s/%s/%s' %(FLAGS.dataset,model,listdir[idx])))
                    """
		    savename = os.path.join(savepath,'result/%s.bmp' % (img_files[idx][-10:]))
		    #savename = os.path.join(savepath,'single_normal_%02d.bmp' % (idx+1))
		    #savename = os.path.join(savepath,'%s/%s/%s/single_normal.bmp' % (FLAGS.dataset,model,listdir[idx]))
		    scipy.misc.imsave(savename, output)

	    elif VAL_OPTION ==2: # light source fixed
                list_val = [11,16,21,22,33,36,38,53,59,92]
		save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		save_files  = natsorted(save_files)
		savepath ='./RMSS_ang_scale_loss_result'
		for model_idx in range(0,len(save_files),2):
		    model = save_files[model_idx]
		    model = model.split('/')
		    model = model[-1]
		    dcgan.load(FLAGS.checkpoint_dir,model)
            	    for idx in range(len(list_val)):
			if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		            os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		        for idx2 in range(1,10): 
			    print("Selected material %03d/%d" % (list_val[idx],idx2))
			    img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
			    input_ = scipy.misc.imread(img+'/3.bmp').astype(float)
			    gt_ = scipy.misc.imread('/research2/IR_normal_small/save016/1/12_Normal.bmp').astype(float)
			    input_ = scipy.misc.imresize(input_,[300,400])

			    input_  = (input_/127.5)-1. # normalize -1 ~1
			    gt_ = scipy.misc.imresize(gt_,[600,800])
			    gt_ = np.reshape(gt_,(1,600,800,3)) 
			    gt_ = np.array(gt_).astype(np.float32)
			    input_ = np.reshape(input_,(1,600,800,1)) 
			    input_ = np.array(input_).astype(np.float32)
			    start_time = time.time() 
			    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
			    print('time: %.8f' %(time.time()-start_time))     
			    # normalization #
			    sample = np.squeeze(sample).astype(np.float32)
			    output = np.zeros((600,800,3)).astype(np.float32)
			    output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
			    output[output ==inf] = 0.0
			    sample = (output+1.)/2.
			    if not os.path.exists(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2))):
			        os.makedirs(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2)))
			    savename = os.path.join(savepath, '%03d/%d/single_normal_%s.bmp' % (list_val[idx],idx2,model))
			    scipy.misc.imsave(savename, sample)

	    elif VAL_OPTION ==3: # depends on light sources 
                list_val = [11,16,21,22,33,36,38,53,59,92]
		mean_nir = -0.3313 #-1~1
		save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		save_files  = natsorted(save_files)
		savepath ='./low_result'
		if not os.path.exists(os.path.join(savepath)):
		    os.makedirs(os.path.join(savepath))
		selec_model=[0,2,4,6,8,10,12,14,16,18]
		#[selec_model.append(ii) for ii in range(0,len(save_files),2)]
                for m in range(len(selec_model)):
		    model = save_files[selec_model[m]]
		    model = model.split('/')
		    model = model[-1]
		    dcgan.load(FLAGS.checkpoint_dir,model)
	            for idx in range(len(list_val)):
		        if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		            os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		        for idx2 in range(1,10): #tilt angles 1~9 
		            for idx3 in range(1,13): # light source 
			        print("Selected material %03d/%d" % (list_val[idx],idx2))
			        #img = '/research2/ECCV_dataset_resized/save%03d/%d' % (list_val[idx],idx2)
			        img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
			        input_ = scipy.misc.imread(img+'/%d.bmp' %idx3).astype(np.float32) #input NIR image
			        input_ = scipy.misc.imresize(input_,[600,800],'nearest')
			        input_ = np.reshape(input_,(600,800,1))

				nondetail_input_ = ndimage.gaussian_filter(input_,sigma=(1,1,0),order=0)	   
				input_ = input_/127.5 -1.0 
			        nondetail_input_  = nondetail_input_/127.5 -1.0 # normalize -1 ~1
			        detail_input_ = input_ - nondetail_input_
			        nondetail_input_ = np.reshape(nondetail_input_,(1,600,800,1))
			        detail_input_ = np.reshape(detail_input_,(1,600,800,1))
			        #detail_input_  = detail_input_/127.5 -1.0 # normalize -1 ~1
			        start_time = time.time() 
	                        sample,_ = sess.run([dcgan.low_G,dcgan.high_G], feed_dict={dcgan.nondetail_images: nondetail_input_,dcgan.detail_images:detail_input_})
			        #sample = sess.run(dcgan.G, feed_dict={dcgan.nondetail_images: nondetail_input_,dcgan.detail_images:detail_input_})
				#sample = np.squeeze(sample).astype(np.float32)
				sample = np.squeeze(sample[-1]).astype(np.float32)
				
			        print('time: %.8f' %(time.time()-start_time))     
			        # normalization #
			        output = np.sqrt(np.sum(np.power(sample,2),axis=2))
			        output = np.expand_dims(output,axis=-1)
			        output = sample/output
			        output = (output+1.)/2.
			        if not os.path.exists(os.path.join(savepath,'%s/%s/%03d/%d' %(FLAGS.dataset,model,list_val[idx],idx2))):
			            os.makedirs(os.path.join(savepath,'%s/%s/%03d/%d' %(FLAGS.dataset,model,list_val[idx],idx2)))
			        savename = os.path.join(savepath,'%s/%s/%03d/%d/single_normal_%03d.bmp' % (FLAGS.dataset,model,list_val[idx],idx2,idx3))
				scipy.misc.imsave(savename, output)


if __name__ == '__main__':
    tf.app.run()
