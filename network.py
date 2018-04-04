from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,gf_dim,df_dim):
	self.df_dim = df_dim
	self.gf_dim = gf_dim
 


    def multi_freq_generator(self,low_nir,high_nir):

        low_layers=[]
	high_layers=[]
        layers_spec=[4,2]
	
	with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	    net =lrelu(conv2d(low_nir,self.gf_dim*2))
	    low_layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	        net =lrelu(batchnorm(conv2d(low_layers[-1],self.gf_dim*maps)))
	        low_layers.append(net)
    
        with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	    net  =conv2d(net,3)
	    net = tf.tanh(net)
	    low_layers.append(net)

	with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
	    net =lrelu(conv2d(high_nir,self.gf_dim*2))
	    high_layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
		low = low_layers[len(high_layers)-1]
	    	net = high_layers[-1]+low
		net = lrelu(batchnorm(conv2d(net,self.gf_dim*maps)))
                high_layers.append(net)

	with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
	    net = tf.tanh(conv2d(high_layers[-1],3))
	    high_layers.append(net)
	
	return low_layers,high_layers

    def generator(self,nir):

        layers=[]
	layers_spec=[4,2]
	
	with tf.variable_scope('g%d' %(len(layers)+1)):
	    net =lrelu(conv2d(nir,self.gf_dim*2))
	    layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('g%d' %(len(layers)+1)):
	        net =lrelu(batchnorm(conv2d(layers[-1],self.gf_dim*maps)))
	        layers.append(net)
    
        with tf.variable_scope('g%d' %(len(layers)+1)):
	    net  =conv2d(net,3)
	    net = tf.tanh(net)
	    layers.append(net)

	return layers

    def multi_freq_sampler(self,low_nir,high_nir):

        low_layers=[]
	high_layers=[]
        layers_spec=[4,2]
	
	tf.get_variable_scope().reuse_variables()	
	with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	    net =lrelu(conv2d(low_nir,self.gf_dim*2))
	    low_layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	        net =lrelu(batchnorm(conv2d(low_layers[-1],self.gf_dim*maps)))
	        low_layers.append(net)
    
        with tf.variable_scope('low_g%d' %(len(low_layers)+1)):
	    net  =conv2d(net,3)
	    net = tf.tanh(net)
	    low_layers.append(net)

	with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
	    net =lrelu(conv2d(high_nir,self.gf_dim*2))
	    high_layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
		low = low_layers[len(high_layers)-1]
	    	net = high_layers[-1]+low
		net = lrelu(batchnorm(conv2d(net,self.gf_dim*maps)))
                high_layers.append(net)

	with tf.variable_scope('high_g%d' %(len(high_layers)+1)):
	    net = tf.tanh(conv2d(high_layers[-1],3))
	    high_layers.append(net)
	
	return low_layers,high_layers

    def sampler(self,nir):

	layers=[]
        layers_spec=[4,2]
	
	tf.get_variable_scope().reuse_variables()	
	with tf.variable_scope('g%d' %(len(layers)+1)):
	    net =lrelu(conv2d(nir,self.gf_dim*2))
	    layers.append(net)

        for maps in layers_spec:
	    with tf.variable_scope('g%d' %(len(layers)+1)):
	        net =lrelu(batchnorm(conv2d(layers[-1],self.gf_dim*maps)))
	        layers.append(net)
    
        with tf.variable_scope('g%d' %(len(layers)+1)):
	    net  =conv2d(net,3)
	    net = tf.tanh(net)
	    layers.append(net)

	return layers


    def discriminator_low(self, image,keep_prob, reuse=False): 
	layers=[]
	layers_spec=[2,4]
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
            net = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2)) 
	    layers.append(net)
            for i in layers_spec:
                with tf.variable_scope('low_dis%d' %(len(layers)+1)):
                    net = lrelu(conv2d(layers[-1],self.df_dim*i,d_h=2,d_w=2)) 
	            layers.append(net)
            with tf.variable_scope('low_dis%d' %(len(layers)+1)):
                net= tf.nn.sigmoid(conv2d(layers[-1], 1, d_h=1,d_w=1))
                layers.append(net)
            return layers

    def discriminator_high(self, image,keep_prob, reuse=False): 
	layers=[]
	layers_spec=[2,4,8]
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
            with tf.variable_scope('high_dis%d' %(len(layers)+1)):
                net = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2)) 
	        layers.append(net)
            for i in layers_spec:
                with tf.variable_scope('high_dis%d' %(len(layers)+1)):
                    net = lrelu(conv2d(layers[-1],self.df_dim*i,d_h=2,d_w=2)) 
	            layers.append(net)
            with tf.variable_scope('high_dis%d' %(len(layers)+1)):
                net= tf.nn.sigmoid(conv2d(layers[-1], 1, d_h=1,d_w=1))
                layers.append(net)
            return layers

    def discriminator(self, image,keep_prob, reuse=False): 
	layers=[]
	layers_spec=[2,4]
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
            net = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2)) 
	    layers.append(net)
            for i in layers_spec:
                with tf.variable_scope('dis%d' %(len(layers)+1)):
                    net = lrelu(conv2d(layers[-1],self.df_dim*i,d_h=2,d_w=2)) 
	            layers.append(net)
            with tf.variable_scope('dis%d' %(len(layers)+1)):
                net= tf.nn.sigmoid(conv2d(layers[-1], 1, d_h=1,d_w=1))
                layers.append(net)
            return layers

