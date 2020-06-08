#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras.layers as tk
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
import glob
from numpy import array
import tqdm
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from PIL import ImageEnhance, ImageFilter
import pytesseract
import string as s
import re
from matplotlib import pyplot as plt


# In[70]:


def crop(img,full_path):
    im=Image.open(full_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.ones((5,5),np.uint8)
    edges=cv2.Canny(gray,0,200,apertureSize=3)
    edges=cv2.dilate(edges,kernel)
    _,contours,heirarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    area=[cv2.contourArea(c) for c in contours]
    Ar_max=np.argmax(area)
    cMax=contours[Ar_max]
    x,y,w,h=cv2.boundingRect(cMax)
    im=im.crop((x,y,x+w,y+h))
    im.save("cropped.jpeg")
    cropped=cv2.imread("cropped.jpeg")
    os.remove("cropped.jpeg")
    plt.imshow(cropped)
    return cropped


# In[57]:


def pre_p(string):
    im=Image.open(string)
    img=cv2.imread(string)
    im.convert('L')
    im=ImageEnhance.Contrast(im)
    im=im.enhance(5)
    im=im.filter(ImageFilter.EDGE_ENHANCE)
    im.save("temp.jpg")
    im=cv2.imread("temp.jpg")
    os.remove("temp.jpg")
    return im


# In[22]:


def res_block(x_in, filters, scaling):
    x = tk.Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = tk.Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = tk.add([x_in, x])
    return x


# In[23]:


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = tk.Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


# In[24]:


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5


def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN


# In[25]:


def load_image(path):
    return np.array(Image.open(path))


# In[26]:


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


# In[27]:


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = tk.Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = tk.Conv2D(num_filters, 3, padding='same')(b)
    x = tk.add([x, b])

    x = upsample(x, scale, num_filters)
    x = tk.Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


# In[28]:


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    return x / 255.0


def normalize_m11(x):  
    return x / 127.5 - 1


def denormalize_m11(x):
    return (x + 1) * 127.5

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


# In[63]:


def resolve_and_plot(model_fine_tuned, lr_image_path):
    lr = load_image(lr_image_path)
    
    #sr_pt = resolve_single(model_pre_trained, lr)
    sr_ft = resolve_single(model_fine_tuned, lr)
    
    plt.figure(figsize=(20, 20))
    
    model_name = model_fine_tuned.name.upper()
    images = [lr,sr_ft]
    titles = ['LR', f'SR ({model_name}, pixel loss)', f'SR ({model_name}, perceptual loss)']
    positions = [1,3]
    
    #for i, (image, title, position) in enumerate(zip(images, titles, positions)):
     #   plt.subplot(2, 2, position)
      #  plt.imshow(image)
       # plt.title(title)
        #plt.xticks([])
        #plt.yticks([])
    return sr_ft


# In[60]:


def Image_enhancer(file_name):
    input_dir="E:\\python\\images"
    path_to_file=os.path.join(input_dir,file_name)
    image_save_dir="E:\python\Output_files"
    weight_dir='E:\\python\\weights\\article\\weights-edsr-16-x4-fine-tuned.h5'
    edsr_fine_tuned = edsr(scale=4, num_res_blocks=16)
    edsr_fine_tuned.load_weights(weight_dir)
    pre_im=pre_p(path_to_file)
    #cropped=cv2.imread(path_to_file)
    cv2.imwrite("temp.jpg",pre_im)
    im=Image.open("temp.jpg")
    im=im.resize((384,384)) #given due to performance constraints of pc
    im.save("temp.jpg")
    enhanced_image=resolve_and_plot(edsr_fine_tuned, 'temp.jpg')
    new_img=os.path.join(image_save_dir,file_name)
    plt.imsave(new_img,np.uint8(enhanced_image))
    return enhanced_image


# In[80]:


def Contours(gray,full_path):
    cropped=crop(gray,full_path)
    gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    rec,thresh=cv2.threshold(gray,0,100,cv2.THRESH_BINARY_INV)
    dilation=cv2.dilate(thresh,kernel,iterations=1)
    dilation=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel)
    _,contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,gray,dilation

def OCR(file_name):
    im=Image.open(file_name)
    im=im.convert('L')
    im.save("tmp.jpg")
    gray=cv2.imread("tmp.jpg")
    os.remove("tmp.jpg")
    pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    contours,cropped,dilation=Contours(gray,file_name)
    ret,cropped=cv2.threshold(cropped,0,255,cv2.THRESH_OTSU)
    #cropped = cv2.GaussianBlur(cropped,(5,5),cv2.BORDER_DEFAULT)
    f=open("E:\\python\\output_files\\text.txt","w+")
    pattern="\d+"
    text=pytesseract.image_to_string(cropped,config='-c tessedit_char_whitelist=0123456789')
    print(text)
    f.write(text)
    f.close()
    cv2.drawContours(cropped,contours,-1,(10,255,5),2)
    return dilation,gray,contours

def main(file_name):
    os.getcwd()
    path="E:/python/output_files"
    full_path=path+'/'+file_name
    dilation,op,contours=OCR(full_path)
    f=open("text2.txt")
    lines=f.readlines()
    f.close
    if len(lines)>1:
        for line in lines:
            if len(line)!=12:
                lines.remove(line)
    f=open("text.txt","w+")
    text=lines[0]
    f.write(lines[0])
    f.close
    return op,contours,text


# In[73]:


def Main_func(file_name):
    Image_enhancer(file_name)
    op,contours,text=main(file_name)


# In[78]:


Main_func("aad10.jpg")


# In[51]:


x=main("aad10.jpg")


# In[140]:


x=tf.image.encode_jpeg(op)


# In[195]:


im=Image.open("E:\\python\\output_files\\aad7.jpeg")
im=im.convert('L')
im.save("temp.jpg")
x=cv2.imread("temp.jpg")
plt.imshow(x)


# In[264]:


dialation,cropped,contours=OCR("E:\\python\\output_files\\aadhar.jpeg")


# In[191]:





# In[265]:


clear


# In[266]:


clear


# In[ ]:




