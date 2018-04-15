"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import json
import pprint
import scipy.misc
import numpy as np
import pdb
import random
from scipy import ndimage
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def create_mask(images):
    mask = [images >-1.][0]*1.
    return mask


def get_image(image_path,gt_path,image_size,npx=64,is_crop=True):
    gt =imread(gt_path)
    low_gt = ndimage.gaussian_filter(gt,sigma=(1,1,0),order=0)	    
    gt = gt/127.5 -1.0
    low_gt = low_gt/127.5-1.0
    high_gt = gt - low_gt
    input_ = imread(image_path)
    low_input_ = ndimage.gaussian_filter(input_,sigma=(1,1,0),order=0)	    
    input_ = input_/127.5 -1.0
    low_input_ = low_input_/127.5-1.0
    high_input_ = input_ - low_input_

    randx = np.random.randint(gt.shape[1]-npx)
    randy = np.random.randint(gt.shape[0]-npx)
    output = np.concatenate((input_,high_input_,low_input_,gt,high_gt,low_gt),axis=2)
    output = output[randy:randy+npx,randx:randx+npx,:]
    return output

def surface_normal(surface):
    surface = surface/np.expand_dims(np.sqrt(np.sum(np.power(surface,2),axis=2)),-1)
    return surface


def get_image_original(image_path,gray=True):
    if gray:
	return imread_gray(image_path)
    else:
	return imread_(image_path)

def get_image_normal(image_path, image_size,randx,randy, is_crop=True):
    return np.array(normalize(imread(image_path)))
    #return transform_normal(normalize(imread(image_path)), image_size, randx,randy,is_crop)
#def get_image(image_path, image_size, is_crop=True):
#    return transform(imread(image_path), image_size, is_crop)

def get_image_eval(image_path):
    return transform_eval(imread(image_path))

#def save_images(images, size, image_path):
#    return imsave(inverse_normalize(images), size, image_path)

def save_normal(image,image_path):
    normal = inverse_transform(image)
    return scipy.misc.imsave(image_path,normal)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread_gray(path):
    return scipy.misc.imread(path,1).astype(np.float)


def imread(path):
    path = path[0].encode("utf-8")
    tmp = scipy.misc.imread(path).astype(np.float)
    if tmp.shape[-1] != 3:
       return np.reshape(tmp, (tmp.shape[0],tmp.shape[1],1))
    return np.reshape(tmp,(tmp.shape[0],tmp.shape[1],3))
       
"""
def imread(path,gray):
    if gray:
        return scipy.misc.imread(path,1).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)
"""
def merge_images(images, size):
    return inverse_transform(images)

def normalize(images):
    images = images/255.0 # 0~1
    images_ = (images * 2.0) -1.0 # -1~1
    z = images[:, :, -1]
    z.clip(np.exp(-100), z.max(),out=z)
    isZero = (z == np.exp(-100)).astype(int)

    yy = images_[:,:,0]/z
    yy[isZero ==1] = 0.0
    xx = images_[:,:,1]/z
    xx[isZero ==1] = 0.0

    tmp = np.squeeze(np.dstack((yy,xx)))
    return tmp



def inverse_normalize(images):
    batchnum  = images.shape[0]
    inv_ = np.zeros((images.shape[0],images.shape[1],images.shape[2],3)).astype(float)
    for batch in range(batchnum):
        y = images[batch,:,:,0]
        x = images[batch,:,:,1]
        z = np.ones((images.shape[1],images.shape[2])).astype(float)
        is_zero = (x == -1).astype(int)
        norm = np.sqrt(np.power(x,2)+np.power(y,2)+1.)
        yy = y/norm
        xx = x/norm
        zz = z/norm

        inv = np.dstack((yy,xx,zz))
        inv = (inv*2.0)+1.
        inv[is_zero ==1]= 0.0
        inv_[batch,:,:,:] = inv      
    return inv_




def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[-1] == 3:
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
    else:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w] = np.squeeze(image)

        
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def random_crop(x,npx,randx,randy):
    npx =64
    return x[randy:randy+npx, randx:randx+npx]

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform_normal(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    mean = 1.0
    std = 0.05
    cropped_image = cropped_image * np.random.normal(mean,std)
    max_val = np.max(cropped_image)
    cropped_image = cropped_image /max_val
    #scipy.misc.imshow(cropped_image)
    #print('cropped image dim:',cropped_image.shape)
    #print('x:%d y:%d' % (randx,randy))
    return np.array(cropped_image)*2. -1.


def transform(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    image = image/127.5 - 1.0
    low_image = ndimage.gaussian_filter(image,sigma=(1,1,0),order=0)	    
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
	low_cropped_image = random_crop(low_image,npx,rands,randy)
	high_cropped_image = cropped_image - low_cropped_image
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return cropped_image,high_cropped_image,low_cropped_image

def transform2(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)*2. - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def load_all_images(data,label,bathch_size,train_size):
     #batch_idxs = min(len(data),train_size)/batch_size
    num_data = len(data)
    datalist =[', '.join(data[idx]) for idx in xrange(0,num_data)]
    labellist =[', '.join(label[idx]) for idx in xrange(0,num_data)]

    ir_batch = [get_image_original(datalist[batch_file],gray=True) for batch_file in range(len(data))]
    normal_batchlabel = [get_image_original(labellist[batch_file],gray=False) for batch_file in range(len(data))]
    batch_images = np.array(ir_batch).astype(np.float32)
    batchlabel_images = np.array(normal_batchlabel).astype(np.float32)

    print('ir image size',batch_images.shape)
    print('normal size',batchlabel_images.shape)
     
    return batch_images, batchlabel_images


def accos_(input_):
    tmp  = np.arccos(input_)
    
    return np.sum(tmp,dtype=np.float32)

def rotation_Light(L,ang):
    rad = ang * math.pi/180.0
    rot =[[math.cos(rad),math.sin(rad),0.0],[-math.sin(rad),math.cos(rad),0.0],[0.0,0.0,1.0]]
    return np.array(np.mat(L)*np.mat(rot))
    
